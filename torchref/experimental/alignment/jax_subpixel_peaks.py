"""
JAX sub-voxel peak refinement for the experimental ball-harmonic MR engine.

Experimental / unstable API: part of ``torchref.experimental.alignment``,
the opt-in ball-harmonic MR engine. The production MR entry point is
``torchref.alignment`` (the consolidated FRF engine). Signatures and behavior
may change without notice.
"""

import jax
import jax.numpy as jnp


@jax.jit
def refine_peaks_subvoxel(
    grid: jax.Array,
    peak_indices: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Refine discrete peaks to sub-voxel accuracy using separable quadratic fitting.

    Parameters
    ----------
    grid : jax.Array
        Array of values, shape (n0, n1, n2).
    peak_indices : jax.Array
        Integer indices of discrete maxima, shape (n_peaks, 3).

    Returns
    -------
    refined_coords : jax.Array
        Fractional coordinates, shape (n_peaks, 3).
    refined_values : jax.Array
        Interpolated peak heights, shape (n_peaks,).
    """
    shape = jnp.array(grid.shape)
    
    def refine_single_peak(idx):
        i0, i1, i2 = idx[0], idx[1], idx[2]
        
        def get_val(d0, d1, d2):
            # Periodic boundary conditions
            ii0 = (i0 + d0) % shape[0]
            ii1 = (i1 + d1) % shape[1]
            ii2 = (i2 + d2) % shape[2]
            return grid[ii0, ii1, ii2]
        
        f0 = get_val(0, 0, 0)
        
        # Axis 0
        fm = get_val(-1, 0, 0)
        fp = get_val(1, 0, 0)
        c0 = (fp - 2*f0 + fm) / 2
        b0 = (fp - fm) / 2
        delta0 = jnp.where(jnp.abs(c0) > 1e-12, jnp.clip(-b0 / (2*c0), -1.0, 1.0), 0.0)
        
        # Axis 1
        fm = get_val(0, -1, 0)
        fp = get_val(0, 1, 0)
        c1 = (fp - 2*f0 + fm) / 2
        b1 = (fp - fm) / 2
        delta1 = jnp.where(jnp.abs(c1) > 1e-12, jnp.clip(-b1 / (2*c1), -1.0, 1.0), 0.0)
        
        # Axis 2
        fm = get_val(0, 0, -1)
        fp = get_val(0, 0, 1)
        c2 = (fp - 2*f0 + fm) / 2
        b2 = (fp - fm) / 2
        delta2 = jnp.where(jnp.abs(c2) > 1e-12, jnp.clip(-b2 / (2*c2), -1.0, 1.0), 0.0)
        
        # Refined coordinates
        refined = jnp.array([i0 + delta0, i1 + delta1, i2 + delta2])
        
        # Refined value: f(δ) ≈ f0 - b²/(4c) at peak
        peak_val = f0
        peak_val -= jnp.where(jnp.abs(c0) > 1e-12, b0**2 / (4*c0), 0.0)
        peak_val -= jnp.where(jnp.abs(c1) > 1e-12, b1**2 / (4*c1), 0.0)
        peak_val -= jnp.where(jnp.abs(c2) > 1e-12, b2**2 / (4*c2), 0.0)
        
        return refined, peak_val
    
    return jax.vmap(refine_single_peak)(peak_indices)