"""Fused per-atom spherical-cutoff density splat for CPU (C++), with autograd.

The production CPU path for :func:`build_electron_density`, and a deliberate
transliteration of the Metal kernels in ``kernels/mps/_shaders.py``: CPU, CUDA and Metal
implement *one* truncation contract, so ``torchref.sigma_cutoff_ed`` means the same thing
on every device. Canonically:

    voxel v receives atom i's full 5-Gaussian density iff  ||w||^2 <= r_i^2,

where ``w`` is the minimum-image **Cartesian atom->voxel** vector (sphere centred on the
atom, not on its nearest grid node) and ``r_i`` is the raw ``radius_policy`` radius,
enumerated over the triclinic-correct per-axis half-width
``ceil(r_i * n_axis * ||inv_frac row_axis||)``. No grid-dependent requantization of the
radius, no diagonal metric.

Forward partitions the **output** by x-plane via ``at::parallel_for`` and backward
partitions over **atoms**, so neither needs atomics. float32 uses a branchless
``fast_exp`` (the CPU analogue of ``metal::fast::exp``), whose disagreement with
``std::exp`` is far below the amplitude-truncation floor at the default cutoff; float64
uses ``std::exp``, a float64 caller being precision-motivated by definition.

Gradients flow to ``xyz``, ``adp``/``u`` and ``occ`` with identity to the incoming
``density_map``; ``A``/``B`` and the cell matrices get none, as in the CUDA and Metal
kernels. **Backward is first-order only**; a ``create_graph=True`` backward is re-derived
through the portable splat (:func:`_double_backward_vjp`). Unlike the CUDA and Metal
entry points this kernel takes no per-Gaussian ``coeff_mask`` -- that argument is
all-ones at every call site, so it is omitted rather than allocated.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from torchref.base.electron_density.kernels.cpu._cpp_build import build_extension

_CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// exp: fast branchless variant for float, libm for double.
//
// exp(x) = 2^(x*log2(e)): the integer part goes straight into the IEEE exponent
// field, the fractional part through a degree-5 minimax polynomial on [0,1).
// No branches, so the innermost voxel loop stays vectorizable. This mirrors
// metal::fast::exp, which the Metal kernels already use.
// ---------------------------------------------------------------------------
static inline float fast_exp(float x) {
    x = x < -87.0f ? -87.0f : x;                 // below this, exp underflows
    const float t = x * 1.44269504088896341f;    // log2(e)
    const float n = std::floor(t);
    const float f = t - n;
    const float p = 1.0f + f * (0.6931471805f + f * (0.2402265069f
                  + f * (0.0555041087f + f * (0.0096181291f + f * 0.0013333558f))));
    const int32_t bits = (int32_t)((n + 127.0f) * 8388608.0f) & 0x7f800000;
    float scale;
    std::memcpy(&scale, &bits, sizeof(scale));
    return p * scale;
}

// Overloads rather than explicit specializations: the templated kernels below
// call kexp() unqualified and overload resolution picks by scalar_t.
static inline float  kexp(float x)  { return fast_exp(x); }
static inline double kexp(double x) { return std::exp(x); }

template <typename scalar_t> struct K {
    static constexpr scalar_t PI_1P5   = (scalar_t)5.568327996831708;   // pi^1.5
    static constexpr scalar_t PI_SQ    = (scalar_t)9.869604401089358;   // pi^2
    static constexpr scalar_t TWO_PI_SQ= (scalar_t)19.739208802178716;  // 2*pi^2
};

// Cell geometry + this atom's anchor, shared by every kernel below.
template <typename scalar_t>
struct Cell {
    scalar_t uax,uay,uaz, ubx,uby,ubz, ucx,ucy,ucz;   // Cartesian voxel steps
    scalar_t inva,invb,invc;                          // inv_frac row norms
    scalar_t Auc;                                     // |u_c|^2
    scalar_t fnx,fny,fnz;
    int nx,ny,nz;
};

template <typename scalar_t>
static Cell<scalar_t> make_cell(const scalar_t* fm, const scalar_t* im,
                                int nx, int ny, int nz) {
    Cell<scalar_t> c;
    c.nx=nx; c.ny=ny; c.nz=nz;
    c.fnx=(scalar_t)nx; c.fny=(scalar_t)ny; c.fnz=(scalar_t)nz;
    // frac columns are the cell vectors a,b,c; divided by n gives the voxel step.
    c.uax=fm[0]/c.fnx; c.uay=fm[3]/c.fnx; c.uaz=fm[6]/c.fnx;
    c.ubx=fm[1]/c.fny; c.uby=fm[4]/c.fny; c.ubz=fm[7]/c.fny;
    c.ucx=fm[2]/c.fnz; c.ucy=fm[5]/c.fnz; c.ucz=fm[8]/c.fnz;
    c.inva=std::sqrt(im[0]*im[0]+im[1]*im[1]+im[2]*im[2]);
    c.invb=std::sqrt(im[3]*im[3]+im[4]*im[4]+im[5]*im[5]);
    c.invc=std::sqrt(im[6]*im[6]+im[7]*im[7]+im[8]*im[8]);
    c.Auc=c.ucx*c.ucx+c.ucy*c.ucy+c.ucz*c.ucz;
    return c;
}

template <typename scalar_t>
struct Anchor {
    int cix,ciy,ciz, bhx,bhy,bhz;
    scalar_t w0x,w0y,w0z, rc2;
};

template <typename scalar_t>
static inline Anchor<scalar_t> make_anchor(
    const Cell<scalar_t>& c, const scalar_t* xyz, const scalar_t* fm,
    const scalar_t* im, int64_t a, scalar_t rc2)
{
    Anchor<scalar_t> g;
    g.rc2 = rc2;
    const scalar_t r = std::sqrt(rc2);
    // Per-axis bounding box of the Cartesian r-sphere in index space; the sphere
    // test culls the corners. Same formula as the Triton / Metal kernels.
    g.bhx = (int)std::ceil(r*c.fnx*c.inva);
    g.bhy = (int)std::ceil(r*c.fny*c.invb);
    g.bhz = (int)std::ceil(r*c.fnz*c.invc);
    const scalar_t ax=xyz[3*a+0], ay=xyz[3*a+1], az=xyz[3*a+2];
    scalar_t fx=ax*im[0]+ay*im[1]+az*im[2];
    scalar_t fy=ax*im[3]+ay*im[4]+az*im[5];
    scalar_t fz=ax*im[6]+ay*im[7]+az*im[8];
    fx-=std::floor(fx); fy-=std::floor(fy); fz-=std::floor(fz);   // wrap to [0,1)
    g.cix=(int)std::nearbyint(fx*c.fnx);
    g.ciy=(int)std::nearbyint(fy*c.fny);
    g.ciz=(int)std::nearbyint(fz*c.fnz);
    // Sub-voxel residual, taken to Cartesian: the sphere is centred HERE, on the
    // atom, not on the anchor node.
    const scalar_t sx=fx-(scalar_t)g.cix/c.fnx;
    const scalar_t sy=fy-(scalar_t)g.ciy/c.fny;
    const scalar_t sz=fz-(scalar_t)g.ciz/c.fnz;
    g.w0x=fm[0]*sx+fm[1]*sy+fm[2]*sz;
    g.w0y=fm[3]*sx+fm[4]*sy+fm[5]*sz;
    g.w0z=fm[6]*sx+fm[7]*sy+fm[8]*sz;
    return g;
}

static inline int wrap_idx(int i, int n) { return ((i % n) + n) % n; }

// Does any x-plane this atom touches fall in [xlo, xhi)? The plane set is the
// wrapped interval [cix-bhx, cix+bhx] mod nx, so at most two pieces.
static inline bool touches_x(int cix, int bhx, int nx, int64_t xlo, int64_t xhi) {
    if (2*bhx+1 >= nx) return true;
    const int lo = wrap_idx(cix-bhx, nx), hi = lo + 2*bhx;
    if (hi < nx) return (lo < xhi) && (hi >= xlo);
    return (lo < xhi) || ((hi - nx) >= xlo);
}

// In-sphere oz run for one (ox,oy) column. r2(oz) is a convex quadratic in oz, so
// the in-sphere set is one contiguous run; solving for it keeps the innermost loop
// branchless. Widened by one voxel each way, with the exact r2 <= rc2 test retained
// as a 0/1 multiply, so the accepted voxel set is bit-identical to the GPU kernels'
// straight comparison over the full box.
template <typename scalar_t>
static inline bool oz_run(const Cell<scalar_t>& c, scalar_t qx, scalar_t qy,
                          scalar_t qz, scalar_t rc2, int bhz, int& zlo, int& zhi) {
    const scalar_t Bq = (scalar_t)2*(qx*c.ucx+qy*c.ucy+qz*c.ucz);
    const scalar_t Cq = qx*qx+qy*qy+qz*qz - rc2;
    const scalar_t disc = Bq*Bq - (scalar_t)4*c.Auc*Cq;
    if (disc < 0) return false;
    const scalar_t sq = std::sqrt(disc), inv2A = (scalar_t)0.5/c.Auc;
    zlo = (int)std::ceil((-Bq-sq)*inv2A)  - 1;
    zhi = (int)std::floor((-Bq+sq)*inv2A) + 1;
    if (zlo < -bhz) zlo = -bhz;
    if (zhi >  bhz) zhi =  bhz;
    return zlo <= zhi;
}

// ===========================================================================
// ISOTROPIC
// ===========================================================================
template <typename scalar_t>
static void iso_fwd_impl(scalar_t* out, const scalar_t* xyz, const scalar_t* adp,
                         const scalar_t* occ, const scalar_t* Ac, const scalar_t* Bc,
                         const scalar_t* r2cut, const scalar_t* im, const scalar_t* fm,
                         int64_t n_at, int nx, int ny, int nz)
{
    const Cell<scalar_t> c = make_cell(fm, im, nx, ny, nz);
    // Output partitioned by x-plane: disjoint writes, so no atomics.
    at::parallel_for(0, nx, 1, [&](int64_t xlo, int64_t xhi) {
      for (int64_t a = 0; a < n_at; ++a) {
        const Anchor<scalar_t> g = make_anchor(c, xyz, fm, im, a, r2cut[a]);
        if (!touches_x(g.cix, g.bhx, nx, xlo, xhi)) continue;
        scalar_t Bt[5], An[5];
        const scalar_t bi=adp[a], oc=occ[a];
        for (int k=0;k<5;++k) {
            scalar_t bt=(Bc[5*a+k]+bi)*(scalar_t)0.25;
            bt = bt > (scalar_t)0.1 ? bt : (scalar_t)0.1;
            Bt[k]=bt;
            An[k]=Ac[5*a+k]*oc*K<scalar_t>::PI_1P5/(bt*std::sqrt(bt));
        }
        for (int ox=-g.bhx; ox<=g.bhx; ++ox) {
          const int vix = wrap_idx(g.cix+ox, nx);
          if (vix < xlo || vix >= xhi) continue;
          const scalar_t fox=(scalar_t)ox;
          const scalar_t px=fox*c.uax-g.w0x, py=fox*c.uay-g.w0y, pz=fox*c.uaz-g.w0z;
          for (int oy=-g.bhy; oy<=g.bhy; ++oy) {
            const scalar_t foy=(scalar_t)oy;
            const scalar_t qx=px+foy*c.ubx, qy=py+foy*c.uby, qz=pz+foy*c.ubz;
            int zlo, zhi;
            if (!oz_run(c, qx, qy, qz, g.rc2, g.bhz, zlo, zhi)) continue;
            scalar_t* row = out + ((int64_t)vix*ny + wrap_idx(g.ciy+oy, ny))*(int64_t)nz;
            for (int oz=zlo; oz<=zhi; ++oz) {
                const scalar_t foz=(scalar_t)oz;
                const scalar_t wx=qx+foz*c.ucx, wy=qy+foz*c.ucy, wz=qz+foz*c.ucz;
                const scalar_t r2=wx*wx+wy*wy+wz*wz;
                const scalar_t keep = r2 <= g.rc2 ? (scalar_t)1 : (scalar_t)0;
                scalar_t dens=0;
                for (int k=0;k<5;++k) dens += An[k]*kexp(-K<scalar_t>::PI_SQ*r2/Bt[k]);
                row[wrap_idx(g.ciz+oz, nz)] += keep*dens;
            }
          }
        }
      }
    });
}

template <typename scalar_t>
static void iso_bwd_impl(scalar_t* g_xyz, scalar_t* g_adp, scalar_t* g_occ,
                         const scalar_t* go, const scalar_t* xyz, const scalar_t* adp,
                         const scalar_t* occ, const scalar_t* Ac, const scalar_t* Bc,
                         const scalar_t* r2cut, const scalar_t* im, const scalar_t* fm,
                         int64_t n_at, int nx, int ny, int nz)
{
    const Cell<scalar_t> c = make_cell(fm, im, nx, ny, nz);
    // Partitioned over atoms: each thread owns its atoms' gradient slots.
    at::parallel_for(0, n_at, 1, [&](int64_t a0, int64_t a1) {
      for (int64_t a = a0; a < a1; ++a) {
        const Anchor<scalar_t> g = make_anchor(c, xyz, fm, im, a, r2cut[a]);
        scalar_t Bt[5], An[5], clampf[5];
        const scalar_t bi=adp[a], oc=occ[a];
        for (int k=0;k<5;++k) {
            const scalar_t raw=(Bc[5*a+k]+bi)*(scalar_t)0.25;
            const scalar_t bt = raw > (scalar_t)0.1 ? raw : (scalar_t)0.1;
            Bt[k]=bt;
            An[k]=Ac[5*a+k]*oc*K<scalar_t>::PI_1P5/(bt*std::sqrt(bt));
            // in the clamp region d(Bt)/d(adp) = 0
            clampf[k] = raw > (scalar_t)0.1 ? (scalar_t)1 : (scalar_t)0;
        }
        scalar_t gx=0,gy=0,gz=0,gb=0,gocc=0;
        for (int ox=-g.bhx; ox<=g.bhx; ++ox) {
          const int vix = wrap_idx(g.cix+ox, nx);
          const scalar_t fox=(scalar_t)ox;
          const scalar_t px=fox*c.uax-g.w0x, py=fox*c.uay-g.w0y, pz=fox*c.uaz-g.w0z;
          for (int oy=-g.bhy; oy<=g.bhy; ++oy) {
            const scalar_t foy=(scalar_t)oy;
            const scalar_t qx=px+foy*c.ubx, qy=py+foy*c.uby, qz=pz+foy*c.ubz;
            int zlo, zhi;
            if (!oz_run(c, qx, qy, qz, g.rc2, g.bhz, zlo, zhi)) continue;
            const scalar_t* row = go + ((int64_t)vix*ny + wrap_idx(g.ciy+oy, ny))*(int64_t)nz;
            for (int oz=zlo; oz<=zhi; ++oz) {
                const scalar_t foz=(scalar_t)oz;
                const scalar_t wx=qx+foz*c.ucx, wy=qy+foz*c.ucy, wz=qz+foz*c.ucz;
                const scalar_t r2=wx*wx+wy*wy+wz*wz;
                if (r2 > g.rc2) continue;
                const scalar_t gout = row[wrap_idx(g.ciz+oz, nz)];
                scalar_t dens=0, coeff=0, dbs=0;
                for (int k=0;k<5;++k) {
                    const scalar_t Ae = An[k]*kexp(-K<scalar_t>::PI_SQ*r2/Bt[k]);
                    dens  += Ae;
                    coeff += Ae/Bt[k];
                    dbs   += Ae*(-(scalar_t)1.5/Bt[k]
                                 + K<scalar_t>::PI_SQ*r2/(Bt[k]*Bt[k]))*clampf[k];
                }
                const scalar_t s = gout*(scalar_t)2*K<scalar_t>::PI_SQ*coeff;
                gx += s*wx; gy += s*wy; gz += s*wz;
                gb += gout*(scalar_t)0.25*dbs;
                gocc += gout*dens;
            }
          }
        }
        g_xyz[3*a+0]=gx; g_xyz[3*a+1]=gy; g_xyz[3*a+2]=gz;
        g_adp[a]=gb;
        g_occ[a]= oc != (scalar_t)0 ? gocc/oc : (scalar_t)0;
      }
    });
}

// ===========================================================================
// ANISOTROPIC.  M_g = (B_g*I + 8*pi^2*U)/4, inverted analytically; density uses
// the Mahalanobis form q = w^T Minv w, the cutoff stays the Euclidean sphere.
// ===========================================================================
template <typename scalar_t>
static inline void aniso_minv(const scalar_t* Bc, const scalar_t* u, int64_t a,
                              const scalar_t* Ac, scalar_t oc,
                              scalar_t* p00, scalar_t* p11, scalar_t* p22,
                              scalar_t* p01, scalar_t* p02, scalar_t* p12,
                              scalar_t* An)
{
    const scalar_t T = K<scalar_t>::TWO_PI_SQ;   // 8*pi^2 / 4
    const scalar_t u11=u[6*a+0],u22=u[6*a+1],u33=u[6*a+2],
                   u12=u[6*a+3],u13=u[6*a+4],u23=u[6*a+5];
    const scalar_t md=T*u12, me=T*u13, mf=T*u23;
    for (int k=0;k<5;++k) {
        const scalar_t Bg=Bc[5*a+k];
        const scalar_t ma=(scalar_t)0.25*Bg+T*u11;
        const scalar_t mb=(scalar_t)0.25*Bg+T*u22;
        const scalar_t mc=(scalar_t)0.25*Bg+T*u33;
        const scalar_t det=ma*(mb*mc-mf*mf)-md*(md*mc-me*mf)+me*(md*mf-me*mb);
        const scalar_t inv=(scalar_t)1/det;
        p00[k]=(mb*mc-mf*mf)*inv; p11[k]=(ma*mc-me*me)*inv; p22[k]=(ma*mb-md*md)*inv;
        p01[k]=(me*mf-md*mc)*inv; p02[k]=(md*mf-me*mb)*inv; p12[k]=(md*me-ma*mf)*inv;
        const scalar_t d = det > (scalar_t)1e-10 ? det : (scalar_t)1e-10;
        An[k]=Ac[5*a+k]*oc*K<scalar_t>::PI_1P5/std::sqrt(d);
    }
}

template <typename scalar_t>
static void aniso_fwd_impl(scalar_t* out, const scalar_t* xyz, const scalar_t* u,
                           const scalar_t* occ, const scalar_t* Ac, const scalar_t* Bc,
                           const scalar_t* r2cut, const scalar_t* im, const scalar_t* fm,
                           int64_t n_at, int nx, int ny, int nz)
{
    const Cell<scalar_t> c = make_cell(fm, im, nx, ny, nz);
    at::parallel_for(0, nx, 1, [&](int64_t xlo, int64_t xhi) {
      for (int64_t a = 0; a < n_at; ++a) {
        const Anchor<scalar_t> g = make_anchor(c, xyz, fm, im, a, r2cut[a]);
        if (!touches_x(g.cix, g.bhx, nx, xlo, xhi)) continue;
        scalar_t p00[5],p11[5],p22[5],p01[5],p02[5],p12[5],An[5];
        aniso_minv(Bc, u, a, Ac, occ[a], p00,p11,p22,p01,p02,p12,An);
        for (int ox=-g.bhx; ox<=g.bhx; ++ox) {
          const int vix = wrap_idx(g.cix+ox, nx);
          if (vix < xlo || vix >= xhi) continue;
          const scalar_t fox=(scalar_t)ox;
          const scalar_t px=fox*c.uax-g.w0x, py=fox*c.uay-g.w0y, pz=fox*c.uaz-g.w0z;
          for (int oy=-g.bhy; oy<=g.bhy; ++oy) {
            const scalar_t foy=(scalar_t)oy;
            const scalar_t qx=px+foy*c.ubx, qy=py+foy*c.uby, qz=pz+foy*c.ubz;
            int zlo, zhi;
            if (!oz_run(c, qx, qy, qz, g.rc2, g.bhz, zlo, zhi)) continue;
            scalar_t* row = out + ((int64_t)vix*ny + wrap_idx(g.ciy+oy, ny))*(int64_t)nz;
            for (int oz=zlo; oz<=zhi; ++oz) {
                const scalar_t foz=(scalar_t)oz;
                const scalar_t wx=qx+foz*c.ucx, wy=qy+foz*c.ucy, wz=qz+foz*c.ucz;
                if (wx*wx+wy*wy+wz*wz > g.rc2) continue;
                const scalar_t xx=wx*wx, yy=wy*wy, zz=wz*wz;
                const scalar_t xy=wx*wy, xz=wx*wz, yz=wy*wz;
                scalar_t dens=0;
                for (int k=0;k<5;++k) {
                    const scalar_t q = p00[k]*xx+p11[k]*yy+p22[k]*zz
                        + (scalar_t)2*(p01[k]*xy+p02[k]*xz+p12[k]*yz);
                    dens += An[k]*kexp(-K<scalar_t>::PI_SQ*q);
                }
                row[wrap_idx(g.ciz+oz, nz)] += dens;
            }
          }
        }
      }
    });
}

template <typename scalar_t>
static void aniso_bwd_impl(scalar_t* g_xyz, scalar_t* g_u, scalar_t* g_occ,
                           const scalar_t* go, const scalar_t* xyz, const scalar_t* u,
                           const scalar_t* occ, const scalar_t* Ac, const scalar_t* Bc,
                           const scalar_t* r2cut, const scalar_t* im, const scalar_t* fm,
                           int64_t n_at, int nx, int ny, int nz)
{
    const Cell<scalar_t> c = make_cell(fm, im, nx, ny, nz);
    at::parallel_for(0, n_at, 1, [&](int64_t a0, int64_t a1) {
      for (int64_t a = a0; a < a1; ++a) {
        const Anchor<scalar_t> g = make_anchor(c, xyz, fm, im, a, r2cut[a]);
        const scalar_t oc = occ[a];
        scalar_t p00[5],p11[5],p22[5],p01[5],p02[5],p12[5],An[5];
        aniso_minv(Bc, u, a, Ac, oc, p00,p11,p22,p01,p02,p12,An);
        scalar_t gx=0,gy=0,gz=0,gu0=0,gu1=0,gu2=0,gu3=0,gu4=0,gu5=0,gocc=0;
        for (int ox=-g.bhx; ox<=g.bhx; ++ox) {
          const int vix = wrap_idx(g.cix+ox, nx);
          const scalar_t fox=(scalar_t)ox;
          const scalar_t px=fox*c.uax-g.w0x, py=fox*c.uay-g.w0y, pz=fox*c.uaz-g.w0z;
          for (int oy=-g.bhy; oy<=g.bhy; ++oy) {
            const scalar_t foy=(scalar_t)oy;
            const scalar_t qx=px+foy*c.ubx, qy=py+foy*c.uby, qz=pz+foy*c.ubz;
            int zlo, zhi;
            if (!oz_run(c, qx, qy, qz, g.rc2, g.bhz, zlo, zhi)) continue;
            const scalar_t* row = go + ((int64_t)vix*ny + wrap_idx(g.ciy+oy, ny))*(int64_t)nz;
            for (int oz=zlo; oz<=zhi; ++oz) {
                const scalar_t foz=(scalar_t)oz;
                const scalar_t wx=qx+foz*c.ucx, wy=qy+foz*c.ucy, wz=qz+foz*c.ucz;
                if (wx*wx+wy*wy+wz*wz > g.rc2) continue;
                const scalar_t gout = row[wrap_idx(g.ciz+oz, nz)];
                scalar_t dens=0,sx=0,sy=0,sz=0,s0=0,s1=0,s2=0,s3=0,s4=0,s5=0;
                for (int k=0;k<5;++k) {
                    const scalar_t vx=p00[k]*wx+p01[k]*wy+p02[k]*wz;
                    const scalar_t vy=p01[k]*wx+p11[k]*wy+p12[k]*wz;
                    const scalar_t vz=p02[k]*wx+p12[k]*wy+p22[k]*wz;
                    const scalar_t q = wx*vx+wy*vy+wz*vz;
                    const scalar_t dg = An[k]*kexp(-K<scalar_t>::PI_SQ*q);
                    dens+=dg; sx+=dg*vx; sy+=dg*vy; sz+=dg*vz;
                    const scalar_t P = K<scalar_t>::PI_SQ;
                    s0+=dg*(-(scalar_t)0.5*p00[k]+P*vx*vx);
                    s1+=dg*(-(scalar_t)0.5*p11[k]+P*vy*vy);
                    s2+=dg*(-(scalar_t)0.5*p22[k]+P*vz*vz);
                    s3+=dg*(-(scalar_t)0.5*p01[k]+P*vx*vy);
                    s4+=dg*(-(scalar_t)0.5*p02[k]+P*vx*vz);
                    s5+=dg*(-(scalar_t)0.5*p12[k]+P*vy*vz);
                }
                const scalar_t s2pi = gout*(scalar_t)2*K<scalar_t>::PI_SQ;
                const scalar_t s4pi = gout*(scalar_t)4*K<scalar_t>::PI_SQ;
                gx+=s2pi*sx; gy+=s2pi*sy; gz+=s2pi*sz;
                gu0+=s2pi*s0; gu1+=s2pi*s1; gu2+=s2pi*s2;   // diagonal U
                gu3+=s4pi*s3; gu4+=s4pi*s4; gu5+=s4pi*s5;   // off-diagonal U
                gocc+=gout*dens;
            }
          }
        }
        g_xyz[3*a+0]=gx; g_xyz[3*a+1]=gy; g_xyz[3*a+2]=gz;
        g_u[6*a+0]=gu0; g_u[6*a+1]=gu1; g_u[6*a+2]=gu2;
        g_u[6*a+3]=gu3; g_u[6*a+4]=gu4; g_u[6*a+5]=gu5;
        g_occ[a]= oc != (scalar_t)0 ? gocc/oc : (scalar_t)0;
      }
    });
}

// ===========================================================================
// Bindings. Tensors arrive contiguous and pre-validated from Python.
// ===========================================================================
#define PTR(T, t) (t).data_ptr<T>()

void iso_fwd(torch::Tensor out, torch::Tensor xyz, torch::Tensor adp,
             torch::Tensor occ, torch::Tensor A, torch::Tensor B,
             torch::Tensor r2cut, torch::Tensor inv_frac, torch::Tensor frac,
             int64_t nx, int64_t ny, int64_t nz)
{
    AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "iso_fwd", [&] {
        iso_fwd_impl<scalar_t>(PTR(scalar_t,out), PTR(scalar_t,xyz), PTR(scalar_t,adp),
            PTR(scalar_t,occ), PTR(scalar_t,A), PTR(scalar_t,B), PTR(scalar_t,r2cut),
            PTR(scalar_t,inv_frac), PTR(scalar_t,frac), xyz.size(0), nx, ny, nz);
    });
}

void iso_bwd(torch::Tensor g_xyz, torch::Tensor g_adp, torch::Tensor g_occ,
             torch::Tensor go, torch::Tensor xyz, torch::Tensor adp,
             torch::Tensor occ, torch::Tensor A, torch::Tensor B,
             torch::Tensor r2cut, torch::Tensor inv_frac, torch::Tensor frac,
             int64_t nx, int64_t ny, int64_t nz)
{
    AT_DISPATCH_FLOATING_TYPES(go.scalar_type(), "iso_bwd", [&] {
        iso_bwd_impl<scalar_t>(PTR(scalar_t,g_xyz), PTR(scalar_t,g_adp),
            PTR(scalar_t,g_occ), PTR(scalar_t,go), PTR(scalar_t,xyz),
            PTR(scalar_t,adp), PTR(scalar_t,occ), PTR(scalar_t,A), PTR(scalar_t,B),
            PTR(scalar_t,r2cut), PTR(scalar_t,inv_frac), PTR(scalar_t,frac),
            xyz.size(0), nx, ny, nz);
    });
}

void aniso_fwd(torch::Tensor out, torch::Tensor xyz, torch::Tensor u,
               torch::Tensor occ, torch::Tensor A, torch::Tensor B,
               torch::Tensor r2cut, torch::Tensor inv_frac, torch::Tensor frac,
               int64_t nx, int64_t ny, int64_t nz)
{
    AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "aniso_fwd", [&] {
        aniso_fwd_impl<scalar_t>(PTR(scalar_t,out), PTR(scalar_t,xyz), PTR(scalar_t,u),
            PTR(scalar_t,occ), PTR(scalar_t,A), PTR(scalar_t,B), PTR(scalar_t,r2cut),
            PTR(scalar_t,inv_frac), PTR(scalar_t,frac), xyz.size(0), nx, ny, nz);
    });
}

void aniso_bwd(torch::Tensor g_xyz, torch::Tensor g_u, torch::Tensor g_occ,
               torch::Tensor go, torch::Tensor xyz, torch::Tensor u,
               torch::Tensor occ, torch::Tensor A, torch::Tensor B,
               torch::Tensor r2cut, torch::Tensor inv_frac, torch::Tensor frac,
               int64_t nx, int64_t ny, int64_t nz)
{
    AT_DISPATCH_FLOATING_TYPES(go.scalar_type(), "aniso_bwd", [&] {
        aniso_bwd_impl<scalar_t>(PTR(scalar_t,g_xyz), PTR(scalar_t,g_u),
            PTR(scalar_t,g_occ), PTR(scalar_t,go), PTR(scalar_t,xyz),
            PTR(scalar_t,u), PTR(scalar_t,occ), PTR(scalar_t,A), PTR(scalar_t,B),
            PTR(scalar_t,r2cut), PTR(scalar_t,inv_frac), PTR(scalar_t,frac),
            xyz.size(0), nx, ny, nz);
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("iso_fwd",   &iso_fwd,   "Fused isotropic spherical-cutoff splat (forward)");
    m.def("iso_bwd",   &iso_bwd,   "Fused isotropic spherical-cutoff splat (backward)");
    m.def("aniso_fwd", &aniso_fwd, "Fused anisotropic spherical-cutoff splat (forward)");
    m.def("aniso_bwd", &aniso_bwd, "Fused anisotropic spherical-cutoff splat (backward)");
}
"""

# ---------------------------------------------------------------------------
# Lazy compilation, attempted at most once per process. Mirrors
# ``scatter.py::_get_module``: any failure returns None so the caller degrades to
# the portable plain splat rather than dying.
# ---------------------------------------------------------------------------
_module = None
_module_failed = False
_module_error: Optional[Tuple[str, str]] = None


def _get_module():
    """The compiled fused-splat extension, or None if it could not be built."""
    global _module, _module_failed, _module_error
    if _module is not None:
        return _module
    if _module_failed:
        return None
    _module, _module_error = build_extension("sphere_splat", _CPP_SRC)
    if _module is None:
        _module_failed = True
    return _module


def why_unavailable() -> Optional[str]:
    """``None`` if the fused CPU splat is usable, else why it is not.

    The single availability probe for this backend, consumed by
    :mod:`torchref.utils.backends`. A missing compiler and a compile error are different
    problems, and the captured diagnostic is the only thing separating them.
    """
    if _get_module() is not None:
        return None
    reason = _module_error[0] if _module_error else "unknown reason"
    return (
        f"the fused CPU sphere_splat extension is not available ({reason}); see "
        "torchref.base.electron_density.kernels.cpu.sphere_splat.last_error()"
    )


def sphere_splat_available() -> bool:
    """Whether the fused CPU splat compiled and is ready to dispatch.

    Derived from :func:`why_unavailable` rather than re-testing, so there is one
    availability check here, not two that can drift.
    """
    return why_unavailable() is None




def warmup() -> bool:
    """Eagerly compile, to move the one-time cost off the first refinement step."""
    return _get_module() is not None


def clear_cache() -> None:
    """Forget the compiled module and failure state (rebuilt on next use)."""
    global _module, _module_failed, _module_error
    _module = None
    _module_failed = False
    _module_error = None


def last_error() -> Optional[Tuple[str, str]]:
    """The ``(message, traceback)`` of the last build failure, if any."""
    return _module_error


def _require_module():
    mod = _get_module()
    if mod is None:
        err = _module_error[0] if _module_error else "unknown reason"
        raise RuntimeError(
            f"fused CPU sphere_splat extension not available ({err}). See "
            "torchref.base.electron_density.kernels.cpu.sphere_splat.last_error()."
        )
    return mod


def _double_backward_vjp(plain_fn, ctx, grad_out, leaves, statics, r2cut):
    """Recompute this VJP through the portable differentiable splat.

    The C++ backward is a closed-form first-order formula with no autograd graph, so it
    cannot supply a second derivative -- the same limitation the CUDA and Metal kernels
    have.
    Rather than lose double backward on the CPU default path (a *silently wrong* Hessian was
    a real bug here), the double-backward context is detected and the identical VJP
    re-derived from the portable splat, which is built from differentiable ops.

    ``torch.is_grad_enabled()`` is the detector: autograd runs ``backward`` under
    ``no_grad``
    unless the caller passed ``create_graph=True``. ``grad_out.requires_grad`` is **not**
    usable -- at the top of a ``create_graph=True`` backward it is a plain ``ones`` tensor.
    Gradients are taken w.r.t. the **saved** leaves, not detached copies, so the VJP stays
    connected to the caller's graph; detaching would silently drop the second-order term.
    Both paths share the truncation contract, so first-order values agree to float noise and
    the only cost is that a Hessian workflow runs at the portable splat's speed.
    """
    A, B, inv_frac, frac = statics
    # Only the leaves that actually require grad may be differentiated; asking for
    # the others raises regardless of allow_unused.
    wanted = [t for t in leaves if t.requires_grad]
    out = [None] * len(leaves)
    if wanted:
        zeros = torch.zeros(
            ctx.grid_shape, dtype=grad_out.dtype, device=grad_out.device
        )
        with torch.enable_grad():
            dm = plain_fn(zeros, *leaves, A, B, inv_frac, frac, r2cut.sqrt())
        grads = torch.autograd.grad(
            dm, wanted, grad_out, create_graph=True, allow_unused=True
        )
        it = iter(grads)
        out = [next(it) if t.requires_grad else None for t in leaves]
    # forward returned density_map + splat -> grad wrt density_map is the identity
    return (grad_out, *out) + (None,) * 5


def _prep(density_map, xyz, radius_per_atom, *tensors):
    """Shared validation + contiguity for both entry points."""
    if density_map.device.type != "cpu":
        raise ValueError(
            f"sphere_splat is a CPU kernel; got device {density_map.device}"
        )
    dtype = density_map.dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"sphere_splat supports float32/float64, got {dtype}")
    for t in (xyz, radius_per_atom) + tensors:
        if t.dtype != dtype:
            raise ValueError(
                f"every input must match density_map.dtype ({dtype}); got {t.dtype}"
            )
    r2cut = (radius_per_atom * radius_per_atom).detach().contiguous()
    return dtype, r2cut


class _FusedIsoSplat(torch.autograd.Function):
    """Isotropic fused CPU splat: returns ``density_map + splat``."""

    @staticmethod
    def forward(ctx, density_map, xyz, adp, occ, A, B, r2cut, inv_frac, frac):
        mod = _require_module()
        nx, ny, nz = (int(s) for s in density_map.shape)
        out = density_map.contiguous().clone()
        if xyz.shape[0] > 0:
            mod.iso_fwd(
                out.view(-1),
                xyz.detach().contiguous(), adp.detach().contiguous(),
                occ.detach().contiguous(), A.contiguous(), B.contiguous(),
                r2cut, inv_frac.contiguous().view(-1), frac.contiguous().view(-1),
                nx, ny, nz,
            )
        ctx.save_for_backward(xyz, adp, occ, A, B, r2cut, inv_frac, frac)
        ctx.grid_shape = (nx, ny, nz)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        xyz, adp, occ, A, B, r2cut, inv_frac, frac = ctx.saved_tensors
        nx, ny, nz = ctx.grid_shape
        if torch.is_grad_enabled():  # create_graph=True -> need a differentiable VJP
            from torchref.base.electron_density.kernels.cpu.variable_radius import (
                add_isotropic_plain_var,
            )

            return _double_backward_vjp(
                add_isotropic_plain_var, ctx, grad_out,
                (xyz, adp, occ), (A, B, inv_frac, frac), r2cut,
            )
        g_xyz = torch.zeros_like(xyz)
        g_adp = torch.zeros_like(adp)
        g_occ = torch.zeros_like(occ)
        if xyz.shape[0] > 0:
            _require_module().iso_bwd(
                g_xyz.view(-1), g_adp, g_occ,
                grad_out.contiguous().view(-1),
                xyz.detach().contiguous(), adp.detach().contiguous(),
                occ.detach().contiguous(), A.contiguous(), B.contiguous(),
                r2cut, inv_frac.contiguous().view(-1), frac.contiguous().view(-1),
                nx, ny, nz,
            )
        # out = density_map + splat, so grad wrt density_map is the identity.
        return (grad_out, g_xyz, g_adp, g_occ, None, None, None, None, None)


class _FusedAnisoSplat(torch.autograd.Function):
    """Anisotropic fused CPU splat: returns ``density_map + splat``."""

    @staticmethod
    def forward(ctx, density_map, xyz, u, occ, A, B, r2cut, inv_frac, frac):
        mod = _require_module()
        nx, ny, nz = (int(s) for s in density_map.shape)
        out = density_map.contiguous().clone()
        if xyz.shape[0] > 0:
            mod.aniso_fwd(
                out.view(-1),
                xyz.detach().contiguous(), u.detach().contiguous(),
                occ.detach().contiguous(), A.contiguous(), B.contiguous(),
                r2cut, inv_frac.contiguous().view(-1), frac.contiguous().view(-1),
                nx, ny, nz,
            )
        ctx.save_for_backward(xyz, u, occ, A, B, r2cut, inv_frac, frac)
        ctx.grid_shape = (nx, ny, nz)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        xyz, u, occ, A, B, r2cut, inv_frac, frac = ctx.saved_tensors
        nx, ny, nz = ctx.grid_shape
        if torch.is_grad_enabled():  # see _FusedIsoSplat.backward
            from torchref.base.electron_density.kernels.cpu.variable_radius import (
                add_anisotropic_plain_var,
            )

            return _double_backward_vjp(
                add_anisotropic_plain_var, ctx, grad_out,
                (xyz, u, occ), (A, B, inv_frac, frac), r2cut,
            )
        g_xyz = torch.zeros_like(xyz)
        g_u = torch.zeros_like(u)
        g_occ = torch.zeros_like(occ)
        if xyz.shape[0] > 0:
            _require_module().aniso_bwd(
                g_xyz.view(-1), g_u.view(-1), g_occ,
                grad_out.contiguous().view(-1),
                xyz.detach().contiguous(), u.detach().contiguous(),
                occ.detach().contiguous(), A.contiguous(), B.contiguous(),
                r2cut, inv_frac.contiguous().view(-1), frac.contiguous().view(-1),
                nx, ny, nz,
            )
        return (grad_out, g_xyz, g_u, g_occ, None, None, None, None, None)


def add_isotropic_cpu_sphere_var(
    density_map, xyz, adp, occ, A, B, inv_frac_matrix, frac_matrix, radius_per_atom
):
    """Fused isotropic spherical-cutoff splat; returns ``density_map + splat``.

    Parameters
    ----------
    density_map : torch.Tensor
        Running map, shape ``(nx, ny, nz)``, CPU float32/float64. **Not mutated.**
    xyz, adp, occ : torch.Tensor
        Cartesian positions ``(n, 3)``, isotropic B-factors and occupancies ``(n,)``.
    A, B : torch.Tensor
        ITC92 amplitudes / widths, shape ``(n, 5)``.
    inv_frac_matrix, frac_matrix : torch.Tensor
        Cartesian<->fractional, shape ``(3, 3)``. The truncation box comes from these, so
        there is no ``voxel_size`` argument.
    radius_per_atom : torch.Tensor
        Per-atom cutoff in Angstrom, ``(n,)``, from
        :func:`radius_policy.per_atom_radius_iso`. Used raw -- no grid-dependent
        requantization, so the cutoff means the same thing at any sampling.
    """
    _, r2cut = _prep(density_map, xyz, radius_per_atom, adp, occ, A, B)
    return _FusedIsoSplat.apply(
        density_map, xyz, adp, occ, A, B, r2cut, inv_frac_matrix, frac_matrix
    )


def add_anisotropic_cpu_sphere_var(
    density_map, xyz, u, occ, A, B, inv_frac_matrix, frac_matrix, radius_per_atom
):
    """Fused anisotropic spherical-cutoff splat; adds into ``density_map``.

    Identical contract to :func:`add_isotropic_cpu_sphere_var`, but ``u`` carries the 6
    components ``[U11, U22, U33, U12, U13, U23]`` and the density is the full 3D Gaussian
    ``exp(-pi^2 w^T Minv w)`` with ``M_g = (B_g*I + 8*pi^2*U)/4``. The *cutoff* stays the
    Euclidean sphere at ``radius_per_atom`` (the ellipsoid's isotropic bounding radius),
    matching the CUDA and Metal kernels, which likewise cull on Euclidean distance and
    evaluate the Mahalanobis form.
    """
    _, r2cut = _prep(density_map, xyz, radius_per_atom, u, occ, A, B)
    return _FusedAnisoSplat.apply(
        density_map, xyz, u, occ, A, B, r2cut, inv_frac_matrix, frac_matrix
    )
