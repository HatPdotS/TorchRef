That simplifies things significantly. Let me revise the plan to build on your existing infrastructure.

---

## Revised Implementation Plan: ML Orientation Recovery

### What You Have

- Symmetry handling
- Reflection data management
- Structure factor calculation pipeline

### What We Need to Build

The implementation reduces to three main components:

---

### Component 1: Likelihood Functions

#### 1.1 Numerical Primitives

```python
# distributions.py

def stable_log_bessel_i0(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(I_0(x)) for Rice distribution."""
    pass

def rice_log_likelihood(
    F_obs: torch.Tensor,      # observed amplitudes
    F_mean: torch.Tensor,     # |<F>| expected amplitude
    variance: torch.Tensor,   # σ²
) -> torch.Tensor:
    """Log-likelihood for acentric reflections."""
    pass

def woolfson_log_likelihood(
    F_obs: torch.Tensor,
    F_mean: torch.Tensor,
    variance: torch.Tensor,
) -> torch.Tensor:
    """Log-likelihood for centric reflections."""
    pass
```

#### 1.2 Variance and D Factor Estimation

```python
# scaling.py

def compute_d_factors(
    resolution: torch.Tensor,  # d-spacing per reflection
    rms_error: float,          # estimated coordinate error (Å)
    f_p: float = 1.0,          # fraction modeled
    f_sol: float = 0.95,       # solvent parameters
    b_sol: float = 300.0,
) -> torch.Tensor:
    """Resolution-dependent Luzzati D factors (equation 19)."""
    pass

def estimate_variance(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    D: torch.Tensor,
    epsilon: torch.Tensor,     # expected intensity factors
) -> torch.Tensor:
    """Compute σ² = ε(N - D²<|F_calc|²>) for MLTF."""
    pass
```

#### 1.3 Rotation Function (Sim MLRF)

```python
# rotation_functions.py

def sim_mlrf(
    F_obs: torch.Tensor,
    F_j_magnitudes: list[torch.Tensor],  # |F| from each symmetry copy
    D: torch.Tensor,
    epsilon: torch.Tensor,
    centric_flags: torch.Tensor,
    N_expected: torch.Tensor,
) -> torch.Tensor:
    """
    Sim maximum likelihood rotation function (equation 8).
    
    F_j_magnitudes: list of |F_calc| for each symmetry-related copy
                    of the model at the trial orientation.
    
    Returns scalar log-likelihood gain.
    """
    # Variance from random walk of symmetry copies
    # σ_S = Σ_j D²|F_j|² - D²_big|F_big|² + (N - Σ_j D²<|F_j|²>)
    pass
```

#### 1.4 Translation Function (MLTF)

```python
# translation_functions.py

def mltf(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,      # complex, coherent sum over symmetry
    D: torch.Tensor,
    epsilon: torch.Tensor,
    centric_flags: torch.Tensor,
) -> torch.Tensor:
    """
    Maximum likelihood translation function (equation 10).
    
    F_calc includes symmetry expansion - this is a coherent sum
    where phases are determined by the translation.
    
    Returns scalar log-likelihood gain.
    """
    pass
```

---

### Component 2: Differentiable Rigid Body Transform

This connects your existing SF calculation to the likelihood targets.

```python
# rigid_body.py

class RigidBodyTransform(nn.Module):
    """
    Applies rotation and translation to model coordinates,
    calculates structure factors, and evaluates likelihood.
    """
    
    def __init__(
        self,
        atom_coords: torch.Tensor,           # (N_atoms, 3) fractional
        atom_scattering: torch.Tensor,       # scattering factors
        F_obs: torch.Tensor,
        hkl: torch.Tensor,
        space_group,                          # your symmetry handler
        cell,                                 # your cell handler
        rms_error: float = 1.0,
    ):
        super().__init__()
        # Store fixed data as buffers
        self.register_buffer('atom_coords', atom_coords)
        # ... etc
        
        # Precompute D factors, epsilon, centric flags
        self.register_buffer('D', compute_d_factors(...))
        self.register_buffer('centric', self._compute_centric_flags())
        
    def forward(
        self, 
        rotation_quat: torch.Tensor,  # (4,) normalized quaternion
        translation: torch.Tensor,    # (3,) fractional
    ) -> torch.Tensor:
        """
        Returns negative log-likelihood (for minimization).
        """
        # 1. Apply rotation and translation to coordinates
        R = quaternion_to_matrix(rotation_quat)
        coords_transformed = self.atom_coords @ R.T + translation
        
        # 2. Expand by symmetry and calculate F_calc
        #    (uses your existing SF pipeline)
        F_calc = self._calculate_structure_factors_with_symmetry(
            coords_transformed
        )
        
        # 3. Evaluate MLTF
        ll = mltf(
            self.F_obs, 
            F_calc, 
            self.D, 
            self.epsilon,
            self.centric,
        )
        
        return -ll  # negative for minimization
    
    def _calculate_structure_factors_with_symmetry(
        self, 
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Coherent sum of F_calc over all symmetry operations.
        
        F_total(h) = Σ_sym F(h, sym_op · coords)
        """
        F_total = torch.zeros(len(self.hkl), dtype=torch.complex64)
        
        for sym_op in self.space_group.operations():
            coords_sym = self._apply_symmetry(coords, sym_op)
            F_sym = self._calculate_structure_factors(coords_sym)
            F_total = F_total + F_sym
            
        return F_total
```

---

### Component 3: Optimization Loop

```python
# optimize.py

def refine_orientation(
    atom_coords: torch.Tensor,
    atom_scattering: torch.Tensor,
    F_obs: torch.Tensor,
    hkl: torch.Tensor,
    space_group,
    cell,
    init_rotation: torch.Tensor | None = None,   # quaternion
    init_translation: torch.Tensor | None = None,
    rms_error: float = 1.0,
    n_iterations: int = 100,
    tolerance: float = 1e-6,
) -> dict:
    """
    Refine rotation and translation against MLTF target.
    
    Returns dict with:
        - rotation: refined quaternion
        - translation: refined fractional translation
        - llg: final log-likelihood gain
        - converged: bool
    """
    # Initialize target function
    target = RigidBodyTransform(
        atom_coords, atom_scattering, F_obs, hkl,
        space_group, cell, rms_error
    )
    
    # Initialize parameters
    if init_rotation is None:
        init_rotation = torch.tensor([1., 0., 0., 0.])
    if init_translation is None:
        init_translation = torch.zeros(3)
    
    rotation = nn.Parameter(init_rotation.clone())
    translation = nn.Parameter(init_translation.clone())
    
    # LBFGS works well for this (matches Phaser's BFGS)
    optimizer = torch.optim.LBFGS(
        [rotation, translation],
        lr=1.0,
        max_iter=20,
        line_search_fn='strong_wolfe',
    )
    
    prev_loss = float('inf')
    
    for iteration in range(n_iterations):
        def closure():
            optimizer.zero_grad()
            
            # Normalize quaternion for forward pass
            quat_normalized = rotation / rotation.norm()
            loss = target(quat_normalized, translation)
            loss.backward()
            
            return loss
        
        loss = optimizer.step(closure)
        
        # Renormalize quaternion after step
        with torch.no_grad():
            rotation.data = rotation.data / rotation.data.norm()
        
        # Check convergence
        if abs(prev_loss - loss.item()) < tolerance:
            break
        prev_loss = loss.item()
    
    return {
        'rotation': rotation.detach(),
        'translation': translation.detach(),
        'llg': -loss.item(),
        'converged': iteration < n_iterations - 1,
        'n_iterations': iteration + 1,
    }
```

---

### Module Structure (Simplified)

```
ml_orientation/
├── __init__.py
├── distributions.py      # Rice, Woolfson, stable log-Bessel
├── scaling.py            # D factors, variance estimation
├── likelihood.py         # MLRF, MLTF implementations
├── rigid_body.py         # differentiable transform + target
├── quaternion.py         # quaternion <-> matrix conversions
└── optimize.py           # LBFGS refinement loop
```

---

### Implementation Order

**Step 1**: `distributions.py` + `quaternion.py`
- Stable log-Bessel
- Rice and Woolfson log-likelihoods
- Quaternion utilities (if not already in your codebase)

**Step 2**: `scaling.py`
- D factor calculation
- Variance estimation

**Step 3**: `likelihood.py`
- MLTF implementation
- Unit tests against simple known cases

**Step 4**: `rigid_body.py`
- Connect your SF calculation
- Handle symmetry expansion correctly
- Verify gradients with `torch.autograd.gradcheck`

**Step 5**: `optimize.py`
- LBFGS loop with quaternion normalization
- Convergence monitoring

**Step 6**: Integration testing
- Apply known small perturbation to solved structure
- Verify recovery to within expected tolerance

---

### Key Implementation Details

#### Symmetry Expansion for MLTF

The critical point: translation function uses **coherent sum**:

```python
F_calc(h) = Σ_s f_s(h) · exp(2πi h · (R_s · r + t_s + T))
```

Where:
- `R_s, t_s` are the symmetry operation rotation and translation
- `T` is the translation being searched/refined
- The sum is over all symmetry operations

Your existing SF calculation probably does this already - you just need to ensure it's differentiable through PyTorch.

#### Centric vs Acentric Handling

```python
def combined_log_likelihood(F_obs, F_mean, variance, centric_flags):
    ll = torch.zeros_like(F_obs)
    
    acentric = ~centric_flags
    ll[acentric] = rice_log_likelihood(
        F_obs[acentric], F_mean[acentric], variance[acentric]
    )
    ll[centric_flags] = woolfson_log_likelihood(
        F_obs[centric_flags], F_mean[centric_flags], variance[centric_flags]
    )
    
    return ll.sum()
```

#### Gradient Stability

The main numerical issue is `log(I_0(x))` for large x. Test your implementation with:

```python
x = torch.tensor([0.1, 1.0, 10.0, 100.0, 1000.0], requires_grad=True)
y = stable_log_bessel_i0(x)
y.sum().backward()
# Check x.grad is finite and reasonable
```

---

### Questions Before You Start

1. **Your SF calculation**: Does it already handle symmetry expansion, or do you apply symmetry separately?

2. **Coordinate convention**: Fractional or orthogonal? (Plan assumes fractional)

3. **Existing quaternion code**: Do you have quaternion-to-matrix conversion, or should I include that?

4. **Centric flags**: Do you have these precomputed for your reflection set?

5. **Test case**: Do you have a structure where you can apply a known perturbation and test recovery?

For testing please use slurm and sbatch the best partition to use is day 