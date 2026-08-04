"""Metal Shading Language source for the variable-radius density splat.

Compiled at runtime via ``torch.mps.compile_shader`` (see ``compile.py``). One GPU thread
per atom (``threads=[n_atoms]``): each sizes its own cubic bounding box from ``r2cut``
and the inverse-cell row norms, iterates it, truncates to the per-atom sphere
(``r2 <= r2cut``), evaluates the 5-term ITC92 Gaussian, and accumulates via a portable
compare-exchange float atomic-add that works on every Metal GPU family (Apple7 / M1
onward), not just Metal-3 native ``atomic_float``. Because each thread owns one atom, the
backward kernels need no atomics.

The math mirrors ``kernels/cuda/variable_radius.py`` and is validated against the
portable CPU reference. Coordinates are Cartesian offsets: ``frac`` is
fractional->Cartesian (columns are a, b, c), ``inv_frac`` its inverse, and for voxel
offset ``o`` from the atom's grid anchor ``w = frac @ (o/n - residual)`` with
``r2 = w.w``. Constants match the reference bit-for-bit.
"""

# NOTE: kept as a single translation unit so one compile_shader call yields all
# kernels. Backward (iso/aniso) and anisotropic kernels are appended below as
# each is validated.
MSL_SOURCE = r"""
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant float PI_1P5 = 5.568327996831708f;   // pi^1.5
constant float PI_SQ  = 9.869604401089358f;   // pi^2

// Portable float atomic-add via compare-exchange on uint. Works on EVERY Metal
// GPU family (incl. Apple7 / M1), not just Metal-3 native atomic_float. Measured
// identical to native on Apple8/M2 -- the splat's per-atom locality keeps
// contention low, so the CAS loop essentially never retries.
inline void atomic_add_f(device atomic_uint* addr, float val) {
    uint old = atomic_load_explicit(addr, memory_order_relaxed), nxt;
    do { nxt = as_type<uint>(as_type<float>(old) + val); }
    while (!atomic_compare_exchange_weak_explicit(
        addr, &old, nxt, memory_order_relaxed, memory_order_relaxed));
}

// ---------------------------------------------------------------------------
// Isotropic forward: accumulate 5-Gaussian density into grid (atomic add).
// ---------------------------------------------------------------------------
kernel void iso_splat_fwd(
    device atomic_uint*  grid       [[buffer(0)]],   // (nx*ny*nz,) accumulator
    device const float*  xyz        [[buffer(1)]],   // (n,3) Cartesian
    device const float*  adp        [[buffer(2)]],   // (n,)  isotropic B
    device const float*  occ        [[buffer(3)]],   // (n,)
    device const float*  A          [[buffer(4)]],   // (n,5) ITC92 amplitudes
    device const float*  B          [[buffer(5)]],   // (n,5) ITC92 widths
    device const float*  r2cut      [[buffer(6)]],   // (n,)  squared radius
    device const float*  mask       [[buffer(7)]],   // (n,5) per-Gaussian mask
    device const float*  inv_frac   [[buffer(8)]],   // 9 (row-major 3x3)
    device const float*  frac       [[buffer(9)]],   // 9 (row-major 3x3)
    constant int&        n_atoms    [[buffer(10)]],
    constant int&        nx         [[buffer(11)]],
    constant int&        ny         [[buffer(12)]],
    constant int&        nz         [[buffer(13)]],
    uint a [[thread_position_in_grid]])
{
    if (a >= (uint)n_atoms) return;

    // Cell geometry (frac columns are cell vectors a,b,c).
    float f0=frac[0],f1=frac[1],f2=frac[2],
          f3=frac[3],f4=frac[4],f5=frac[5],
          f6=frac[6],f7=frac[7],f8=frac[8];
    float i0=inv_frac[0],i1=inv_frac[1],i2=inv_frac[2],
          i3=inv_frac[3],i4=inv_frac[4],i5=inv_frac[5],
          i6=inv_frac[6],i7=inv_frac[7],i8=inv_frac[8];
    float fnx=(float)nx, fny=(float)ny, fnz=(float)nz;

    // Per-axis Cartesian voxel step vectors (cell vector / n).
    float uax=f0/fnx, uay=f3/fnx, uaz=f6/fnx;
    float ubx=f1/fny, uby=f4/fny, ubz=f7/fny;
    float ucx=f2/fnz, ucy=f5/fnz, ucz=f8/fnz;

    float rc2 = r2cut[a];
    float r   = sqrt(rc2);
    // Triclinic-correct half-widths: max|off_axis| = n_axis * r * ||inv_frac row||.
    int bhx = (int)ceil(r*fnx*sqrt(i0*i0+i1*i1+i2*i2));
    int bhy = (int)ceil(r*fny*sqrt(i3*i3+i4*i4+i5*i5));
    int bhz = (int)ceil(r*fnz*sqrt(i6*i6+i7*i7+i8*i8));

    // Atom -> fractional -> wrap into [0,1) -> nearest grid node + residual.
    float ax=xyz[3*a+0], ay=xyz[3*a+1], az=xyz[3*a+2];
    float fx=ax*i0+ay*i1+az*i2;
    float fy=ax*i3+ay*i4+az*i5;
    float fz=ax*i6+ay*i7+az*i8;
    fx-=floor(fx); fy-=floor(fy); fz-=floor(fz);
    int cix=(int)rint(fx*fnx), ciy=(int)rint(fy*fny), ciz=(int)rint(fz*fnz);
    float sx=fx-(float)cix/fnx, sy=fy-(float)ciy/fny, sz=fz-(float)ciz/fnz;
    float w0x=f0*sx+f1*sy+f2*sz;   // residual -> Cartesian
    float w0y=f3*sx+f4*sy+f5*sz;
    float w0z=f6*sx+f7*sy+f8*sz;

    // Per-Gaussian amplitude / width.
    float occa=occ[a], b_iso=adp[a];
    float Bt[5], An[5];
    for (int g=0; g<5; ++g) {
        float Bt_g = max((B[5*a+g]+b_iso)*0.25f, 0.1f);
        Bt[g] = Bt_g;
        An[g] = mask[5*a+g]*A[5*a+g]*occa*PI_1P5/(Bt_g*sqrt(Bt_g));
    }

    for (int ox=-bhx; ox<=bhx; ++ox) {
      float fox=(float)ox;
      for (int oy=-bhy; oy<=bhy; ++oy) {
        float foy=(float)oy;
        for (int oz=-bhz; oz<=bhz; ++oz) {
            float foz=(float)oz;
            float wx=fox*uax+foy*ubx+foz*ucx - w0x;
            float wy=fox*uay+foy*uby+foz*ucy - w0y;
            float wz=fox*uaz+foy*ubz+foz*ucz - w0z;
            float r2=wx*wx+wy*wy+wz*wz;
            if (r2 > rc2) continue;
            float dens=0.0f;
            for (int g=0; g<5; ++g) dens += An[g]*fast::exp(-PI_SQ*r2/Bt[g]);
            int vix=cix+ox; vix=((vix%nx)+nx)%nx;
            int viy=ciy+oy; viy=((viy%ny)+ny)%ny;
            int viz=ciz+oz; viz=((viz%nz)+nz)%nz;
            int idx=(vix*ny+viy)*nz+viz;
            atomic_add_f(&grid[idx], dens);
        }
      }
    }
}

// ---------------------------------------------------------------------------
// Isotropic backward: analytic grads to xyz, adp, occ. One thread per atom, so
// each thread owns its atom's gradient slots -> no atomics. Recomputes the
// forward geometry and gathers the upstream grad from the grid.
// ---------------------------------------------------------------------------
kernel void iso_splat_bwd(
    device float*        grad_xyz   [[buffer(0)]],   // (n,3) out
    device float*        grad_adp   [[buffer(1)]],   // (n,)  out
    device float*        grad_occ   [[buffer(2)]],   // (n,)  out
    device const float*  grad_out   [[buffer(3)]],   // (nx*ny*nz,) upstream grad
    device const float*  xyz        [[buffer(4)]],
    device const float*  adp        [[buffer(5)]],
    device const float*  occ        [[buffer(6)]],
    device const float*  A          [[buffer(7)]],
    device const float*  B          [[buffer(8)]],
    device const float*  r2cut      [[buffer(9)]],
    device const float*  mask       [[buffer(10)]],
    device const float*  inv_frac   [[buffer(11)]],
    device const float*  frac       [[buffer(12)]],
    constant int&        n_atoms    [[buffer(13)]],
    constant int&        nx         [[buffer(14)]],
    constant int&        ny         [[buffer(15)]],
    constant int&        nz         [[buffer(16)]],
    uint a [[thread_position_in_grid]])
{
    if (a >= (uint)n_atoms) return;

    float f0=frac[0],f1=frac[1],f2=frac[2],
          f3=frac[3],f4=frac[4],f5=frac[5],
          f6=frac[6],f7=frac[7],f8=frac[8];
    float i0=inv_frac[0],i1=inv_frac[1],i2=inv_frac[2],
          i3=inv_frac[3],i4=inv_frac[4],i5=inv_frac[5],
          i6=inv_frac[6],i7=inv_frac[7],i8=inv_frac[8];
    float fnx=(float)nx, fny=(float)ny, fnz=(float)nz;
    float uax=f0/fnx, uay=f3/fnx, uaz=f6/fnx;
    float ubx=f1/fny, uby=f4/fny, ubz=f7/fny;
    float ucx=f2/fnz, ucy=f5/fnz, ucz=f8/fnz;

    float rc2 = r2cut[a];
    float r   = sqrt(rc2);
    int bhx = (int)ceil(r*fnx*sqrt(i0*i0+i1*i1+i2*i2));
    int bhy = (int)ceil(r*fny*sqrt(i3*i3+i4*i4+i5*i5));
    int bhz = (int)ceil(r*fnz*sqrt(i6*i6+i7*i7+i8*i8));

    float ax=xyz[3*a+0], ay=xyz[3*a+1], az=xyz[3*a+2];
    float fx=ax*i0+ay*i1+az*i2;
    float fy=ax*i3+ay*i4+az*i5;
    float fz=ax*i6+ay*i7+az*i8;
    fx-=floor(fx); fy-=floor(fy); fz-=floor(fz);
    int cix=(int)rint(fx*fnx), ciy=(int)rint(fy*fny), ciz=(int)rint(fz*fnz);
    float sx=fx-(float)cix/fnx, sy=fy-(float)ciy/fny, sz=fz-(float)ciz/fnz;
    float w0x=f0*sx+f1*sy+f2*sz;
    float w0y=f3*sx+f4*sy+f5*sz;
    float w0z=f6*sx+f7*sy+f8*sz;

    float occa=occ[a], b_iso=adp[a];
    float Bt[5], An[5], clampf[5];
    for (int g=0; g<5; ++g) {
        float raw = (B[5*a+g]+b_iso)*0.25f;
        float Bt_g = max(raw, 0.1f);
        Bt[g] = Bt_g;
        An[g] = mask[5*a+g]*A[5*a+g]*occa*PI_1P5/(Bt_g*sqrt(Bt_g));
        clampf[g] = (raw > 0.1f) ? 1.0f : 0.0f;   // d(clamped Bt)/d(adp)=0 in clamp region
    }

    float gx=0.0f, gy=0.0f, gz=0.0f, gb=0.0f, go=0.0f;
    for (int ox=-bhx; ox<=bhx; ++ox) {
      float fox=(float)ox;
      for (int oy=-bhy; oy<=bhy; ++oy) {
        float foy=(float)oy;
        for (int oz=-bhz; oz<=bhz; ++oz) {
            float foz=(float)oz;
            float wx=fox*uax+foy*ubx+foz*ucx - w0x;
            float wy=fox*uay+foy*uby+foz*ucy - w0y;
            float wz=fox*uaz+foy*ubz+foz*ucz - w0z;
            float r2=wx*wx+wy*wy+wz*wz;
            if (r2 > rc2) continue;
            int vix=cix+ox; vix=((vix%nx)+nx)%nx;
            int viy=ciy+oy; viy=((viy%ny)+ny)%ny;
            int viz=ciz+oz; viz=((viz%nz)+nz)%nz;
            float g_out = grad_out[(vix*ny+viy)*nz+viz];
            float dens=0.0f, coeff_xyz=0.0f, db_sum=0.0f;
            for (int g=0; g<5; ++g) {
                float e = fast::exp(-PI_SQ*r2/Bt[g]);
                float Ae = An[g]*e;
                dens += Ae;
                coeff_xyz += Ae/Bt[g];
                db_sum += Ae*(-1.5f/Bt[g] + PI_SQ*r2/(Bt[g]*Bt[g]))*clampf[g];
            }
            float sxyz = g_out*2.0f*PI_SQ*coeff_xyz;
            gx += sxyz*wx; gy += sxyz*wy; gz += sxyz*wz;
            gb += g_out*0.25f*db_sum;
            go += g_out*dens;
        }
      }
    }
    grad_xyz[3*a+0]=gx; grad_xyz[3*a+1]=gy; grad_xyz[3*a+2]=gz;
    grad_adp[a]=gb;
    grad_occ[a]=(occa!=0.0f) ? go/occa : 0.0f;
}

constant float TWO_PI_SQ = 19.739208802178716f;   // 2*pi^2  (= 8*pi^2 / 4)

// ---------------------------------------------------------------------------
// Anisotropic forward. Per Gaussian g: M_g = (B_g*I + 8*pi^2*U)/4 (symmetric
// 3x3), inverted analytically; A_norm_g = mask*A*occ*pi^1.5/sqrt(det M_g);
// q_g = w^T Minv_g w; density = sum_g A_norm_g fast::exp(-pi^2 q_g). Truncation
// is a per-axis bounding box (from r2cut + inv-cell row norms) with a sphere
// cull (r2 <= r2cut) -- far tighter than a full cube on anisotropic high-res
// cells. Uses metal::fast::exp (accurate to ~1e-4, well under the grid floor).
// ---------------------------------------------------------------------------
kernel void aniso_splat_fwd(
    device atomic_uint*  grid       [[buffer(0)]],
    device const float*  xyz        [[buffer(1)]],   // (n,3)
    device const float*  u          [[buffer(2)]],   // (n,6) U11,U22,U33,U12,U13,U23
    device const float*  occ        [[buffer(3)]],
    device const float*  A          [[buffer(4)]],   // (n,5)
    device const float*  B          [[buffer(5)]],   // (n,5)
    device const float*  r2cut      [[buffer(6)]],   // (n,) squared truncation radius
    device const float*  mask       [[buffer(7)]],   // (n,5)
    device const float*  inv_frac   [[buffer(8)]],
    device const float*  frac       [[buffer(9)]],
    constant int&        n_atoms    [[buffer(10)]],
    constant int&        nx         [[buffer(11)]],
    constant int&        ny         [[buffer(12)]],
    constant int&        nz         [[buffer(13)]],
    uint a [[thread_position_in_grid]])
{
    if (a >= (uint)n_atoms) return;

    float f0=frac[0],f1=frac[1],f2=frac[2],
          f3=frac[3],f4=frac[4],f5=frac[5],
          f6=frac[6],f7=frac[7],f8=frac[8];
    float i0=inv_frac[0],i1=inv_frac[1],i2=inv_frac[2],
          i3=inv_frac[3],i4=inv_frac[4],i5=inv_frac[5],
          i6=inv_frac[6],i7=inv_frac[7],i8=inv_frac[8];
    float fnx=(float)nx, fny=(float)ny, fnz=(float)nz;
    float uax=f0/fnx, uay=f3/fnx, uaz=f6/fnx;
    float ubx=f1/fny, uby=f4/fny, ubz=f7/fny;
    float ucx=f2/fnz, ucy=f5/fnz, ucz=f8/fnz;

    float ax=xyz[3*a+0], ay=xyz[3*a+1], az=xyz[3*a+2];
    float fx=ax*i0+ay*i1+az*i2, fy=ax*i3+ay*i4+az*i5, fz=ax*i6+ay*i7+az*i8;
    fx-=floor(fx); fy-=floor(fy); fz-=floor(fz);
    int cix=(int)rint(fx*fnx), ciy=(int)rint(fy*fny), ciz=(int)rint(fz*fnz);
    float sx=fx-(float)cix/fnx, sy=fy-(float)ciy/fny, sz=fz-(float)ciz/fnz;
    float w0x=f0*sx+f1*sy+f2*sz, w0y=f3*sx+f4*sy+f5*sz, w0z=f6*sx+f7*sy+f8*sz;

    float occa=occ[a];
    float u11=u[6*a+0],u22=u[6*a+1],u33=u[6*a+2],u12=u[6*a+3],u13=u[6*a+4],u23=u[6*a+5];
    float p00[5],p11[5],p22[5],p01[5],p02[5],p12[5],An[5];
    for (int g=0; g<5; ++g) {
        float Bg=B[5*a+g];
        float ma=0.25f*Bg+TWO_PI_SQ*u11, mb=0.25f*Bg+TWO_PI_SQ*u22, mc=0.25f*Bg+TWO_PI_SQ*u33;
        float md=TWO_PI_SQ*u12, me=TWO_PI_SQ*u13, mf=TWO_PI_SQ*u23;
        float det=ma*(mb*mc-mf*mf)-md*(md*mc-me*mf)+me*(md*mf-me*mb);
        float inv=1.0f/det;
        p00[g]=(mb*mc-mf*mf)*inv; p11[g]=(ma*mc-me*me)*inv; p22[g]=(ma*mb-md*md)*inv;
        p01[g]=(me*mf-md*mc)*inv; p02[g]=(md*mf-me*mb)*inv; p12[g]=(md*me-ma*mf)*inv;
        An[g]=mask[5*a+g]*A[5*a+g]*occa*PI_1P5/sqrt(max(det,1e-10f));
    }

    float rc2=r2cut[a], r=sqrt(rc2);
    int bhx=(int)ceil(r*fnx*sqrt(i0*i0+i1*i1+i2*i2));
    int bhy=(int)ceil(r*fny*sqrt(i3*i3+i4*i4+i5*i5));
    int bhz=(int)ceil(r*fnz*sqrt(i6*i6+i7*i7+i8*i8));
    for (int ox=-bhx; ox<=bhx; ++ox) {
      float fox=(float)ox;
      for (int oy=-bhy; oy<=bhy; ++oy) {
        float foy=(float)oy;
        for (int oz=-bhz; oz<=bhz; ++oz) {
            float foz=(float)oz;
            float wx=fox*uax+foy*ubx+foz*ucx - w0x;
            float wy=fox*uay+foy*uby+foz*ucy - w0y;
            float wz=fox*uaz+foy*ubz+foz*ucz - w0z;
            if (wx*wx+wy*wy+wz*wz > rc2) continue;
            float dens=0.0f;
            for (int g=0; g<5; ++g) {
                float q=p00[g]*wx*wx+p11[g]*wy*wy+p22[g]*wz*wz
                       +2.0f*(p01[g]*wx*wy+p02[g]*wx*wz+p12[g]*wy*wz);
                dens += An[g]*fast::exp(-PI_SQ*q);
            }
            int vix=cix+ox; vix=((vix%nx)+nx)%nx;
            int viy=ciy+oy; viy=((viy%ny)+ny)%ny;
            int viz=ciz+oz; viz=((viz%nz)+nz)%nz;
            atomic_add_f(&grid[(vix*ny+viy)*nz+viz], dens);
        }
      }
    }
}

// ---------------------------------------------------------------------------
// Anisotropic backward: grads to xyz, U (6), occ. v_g = Minv_g w; dg_g =
// A_norm_g exp(-pi^2 w.v_g). grad_xyz = go*2pi^2*sum dg v; grad_U diag =
// go*2pi^2*sum dg(-0.5 p_ii + pi^2 v_i^2); grad_U offdiag = go*4pi^2*sum
// dg(-0.5 p_ij + pi^2 v_i v_j); grad_occ = go*sum dg / occ. One thread/atom.
// ---------------------------------------------------------------------------
kernel void aniso_splat_bwd(
    device float*        grad_xyz   [[buffer(0)]],   // (n,3)
    device float*        grad_u     [[buffer(1)]],   // (n,6)
    device float*        grad_occ   [[buffer(2)]],   // (n,)
    device const float*  grad_out   [[buffer(3)]],
    device const float*  xyz        [[buffer(4)]],
    device const float*  u          [[buffer(5)]],
    device const float*  occ        [[buffer(6)]],
    device const float*  A          [[buffer(7)]],
    device const float*  B          [[buffer(8)]],
    device const float*  r2cut      [[buffer(9)]],
    device const float*  mask       [[buffer(10)]],
    device const float*  inv_frac   [[buffer(11)]],
    device const float*  frac       [[buffer(12)]],
    constant int&        n_atoms    [[buffer(13)]],
    constant int&        nx         [[buffer(14)]],
    constant int&        ny         [[buffer(15)]],
    constant int&        nz         [[buffer(16)]],
    uint a [[thread_position_in_grid]])
{
    if (a >= (uint)n_atoms) return;

    float f0=frac[0],f1=frac[1],f2=frac[2],
          f3=frac[3],f4=frac[4],f5=frac[5],
          f6=frac[6],f7=frac[7],f8=frac[8];
    float i0=inv_frac[0],i1=inv_frac[1],i2=inv_frac[2],
          i3=inv_frac[3],i4=inv_frac[4],i5=inv_frac[5],
          i6=inv_frac[6],i7=inv_frac[7],i8=inv_frac[8];
    float fnx=(float)nx, fny=(float)ny, fnz=(float)nz;
    float uax=f0/fnx, uay=f3/fnx, uaz=f6/fnx;
    float ubx=f1/fny, uby=f4/fny, ubz=f7/fny;
    float ucx=f2/fnz, ucy=f5/fnz, ucz=f8/fnz;

    float ax=xyz[3*a+0], ay=xyz[3*a+1], az=xyz[3*a+2];
    float fx=ax*i0+ay*i1+az*i2, fy=ax*i3+ay*i4+az*i5, fz=ax*i6+ay*i7+az*i8;
    fx-=floor(fx); fy-=floor(fy); fz-=floor(fz);
    int cix=(int)rint(fx*fnx), ciy=(int)rint(fy*fny), ciz=(int)rint(fz*fnz);
    float sx=fx-(float)cix/fnx, sy=fy-(float)ciy/fny, sz=fz-(float)ciz/fnz;
    float w0x=f0*sx+f1*sy+f2*sz, w0y=f3*sx+f4*sy+f5*sz, w0z=f6*sx+f7*sy+f8*sz;

    float occa=occ[a];
    float u11=u[6*a+0],u22=u[6*a+1],u33=u[6*a+2],u12=u[6*a+3],u13=u[6*a+4],u23=u[6*a+5];
    float p00[5],p11[5],p22[5],p01[5],p02[5],p12[5],An[5];
    for (int g=0; g<5; ++g) {
        float Bg=B[5*a+g];
        float ma=0.25f*Bg+TWO_PI_SQ*u11, mb=0.25f*Bg+TWO_PI_SQ*u22, mc=0.25f*Bg+TWO_PI_SQ*u33;
        float md=TWO_PI_SQ*u12, me=TWO_PI_SQ*u13, mf=TWO_PI_SQ*u23;
        float det=ma*(mb*mc-mf*mf)-md*(md*mc-me*mf)+me*(md*mf-me*mb);
        float inv=1.0f/det;
        p00[g]=(mb*mc-mf*mf)*inv; p11[g]=(ma*mc-me*me)*inv; p22[g]=(ma*mb-md*md)*inv;
        p01[g]=(me*mf-md*mc)*inv; p02[g]=(md*mf-me*mb)*inv; p12[g]=(md*me-ma*mf)*inv;
        An[g]=mask[5*a+g]*A[5*a+g]*occa*PI_1P5/sqrt(max(det,1e-10f));
    }

    float gx=0,gy=0,gz=0,gu0=0,gu1=0,gu2=0,gu3=0,gu4=0,gu5=0,go=0;
    float rc2=r2cut[a], r=sqrt(rc2);
    int bhx=(int)ceil(r*fnx*sqrt(i0*i0+i1*i1+i2*i2));
    int bhy=(int)ceil(r*fny*sqrt(i3*i3+i4*i4+i5*i5));
    int bhz=(int)ceil(r*fnz*sqrt(i6*i6+i7*i7+i8*i8));
    for (int ox=-bhx; ox<=bhx; ++ox) {
      float fox=(float)ox;
      for (int oy=-bhy; oy<=bhy; ++oy) {
        float foy=(float)oy;
        for (int oz=-bhz; oz<=bhz; ++oz) {
            float foz=(float)oz;
            float wx=fox*uax+foy*ubx+foz*ucx - w0x;
            float wy=fox*uay+foy*uby+foz*ucy - w0y;
            float wz=fox*uaz+foy*ubz+foz*ucz - w0z;
            if (wx*wx+wy*wy+wz*wz > rc2) continue;
            int vix=cix+ox; vix=((vix%nx)+nx)%nx;
            int viy=ciy+oy; viy=((viy%ny)+ny)%ny;
            int viz=ciz+oz; viz=((viz%nz)+nz)%nz;
            float g_out=grad_out[(vix*ny+viy)*nz+viz];
            float sxx=0,syy=0,szz=0,dens=0;
            float su0=0,su1=0,su2=0,su3=0,su4=0,su5=0;
            for (int g=0; g<5; ++g) {
                float vx=p00[g]*wx+p01[g]*wy+p02[g]*wz;
                float vy=p01[g]*wx+p11[g]*wy+p12[g]*wz;
                float vz=p02[g]*wx+p12[g]*wy+p22[g]*wz;
                float q=wx*vx+wy*vy+wz*vz;
                float dg=An[g]*fast::exp(-PI_SQ*q);
                dens+=dg; sxx+=dg*vx; syy+=dg*vy; szz+=dg*vz;
                su0+=dg*(-0.5f*p00[g]+PI_SQ*vx*vx);
                su1+=dg*(-0.5f*p11[g]+PI_SQ*vy*vy);
                su2+=dg*(-0.5f*p22[g]+PI_SQ*vz*vz);
                su3+=dg*(-0.5f*p01[g]+PI_SQ*vx*vy);
                su4+=dg*(-0.5f*p02[g]+PI_SQ*vx*vz);
                su5+=dg*(-0.5f*p12[g]+PI_SQ*vy*vz);
            }
            float s=g_out*2.0f*PI_SQ;
            gx+=s*sxx; gy+=s*syy; gz+=s*szz;
            gu0+=s*su0; gu1+=s*su1; gu2+=s*su2;
            gu3+=g_out*4.0f*PI_SQ*su3; gu4+=g_out*4.0f*PI_SQ*su4; gu5+=g_out*4.0f*PI_SQ*su5;
            go+=g_out*dens;
        }
      }
    }
    grad_xyz[3*a+0]=gx; grad_xyz[3*a+1]=gy; grad_xyz[3*a+2]=gz;
    grad_u[6*a+0]=gu0; grad_u[6*a+1]=gu1; grad_u[6*a+2]=gu2;
    grad_u[6*a+3]=gu3; grad_u[6*a+4]=gu4; grad_u[6*a+5]=gu5;
    grad_occ[a]=(occa!=0.0f) ? go/occa : 0.0f;
}
"""
