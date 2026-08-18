# An X-ray data head for OpenFold3, and fine-tuning its ensemble against data

Design note. Two codebases are involved:

- **OpenFold3** — `aqlaboratory/openfold-3` (Apache-2.0), inspected at `72fc3a9`.
  Note the repository name is `openfold-3`, not `openfold3`; the Python package
  is `openfold3`.
- **TorchRef** — this repository.

Everything below about OpenFold3 is read off that checkout, not off the AF3 paper,
so the line references are to code that actually runs.

---

## 0. Summary

OpenFold3 is an AlphaFold3 reproduction: a recycled Pairformer trunk producing
`(s, z)`, followed by a **conditional diffusion model over all-atom Cartesian
coordinates**. The diffusion module is the only part that sees coordinates, and it
is the only sensible attachment point for experimental data.

Three facts dominate the design:

1. **The model has no crystal frame.** `centre_random_augmentation` randomises the
   global pose at every diffusion step, and the training loss superposes onto the
   ground truth with a *detached* Kabsch rotation. Nothing in the network is
   equivariant-aware of a unit cell. A structure-factor target is not
   pose-invariant, so the pose has to be supplied — or eliminated.
2. **Training crops to 384 tokens.** A structure-factor calculation needs the whole
   asymmetric unit, plus bulk solvent over the whole cell.
3. **The 200-step rollout runs under `torch.no_grad()`.** There is no gradient path
   from the sampled ensemble back to the weights today; the trainable path is the
   one-step denoiser `_train_diffusion`.

The recommendation, in one sentence: **bring the data into the model's frame rather
than the model into the data's frame** — compute F_calc in TorchRef under an
externally-maintained (detached) MR pose, turn the residual `mFo − DFc` map into
*frame-invariant, atom-local* probe tokens, and cross-attend to those from the
diffusion transformer through a zero-initialised gate. Then no part of OpenFold3's
SE(3) inductive bias has to be retrained, and the pretrained checkpoint is exactly
recovered at initialisation.

Build it in this order: a **zero-shot guidance baseline first** (no new weights),
then the head. The baseline exercises the entire crystallographic plumbing — pose,
scaling, solvent, gradients, R-factors — and is the thing the head must beat.

---

## 1. OpenFold3 as it stands

### 1.1 Layout

```
openfold3/
  core/
    model/
      feature_embedders/   input_embedders.py, template_embedders.py
      latent/              pairformer.py, msa_module.py, template_module.py, evoformer.py
      layers/              attention_pair_bias.py, diffusion_transformer.py,
                           diffusion_conditioning.py, sequence_local_atom_attention.py,
                           triangular_*.py, transition.py, outer_product_mean.py, msa.py
      structure/           diffusion_module.py, augmentation.py, pocket_constraints.py
      heads/               head_modules.py, prediction_heads.py
      primitives/          attention.py, linear.py, normalization.py, ...
    loss/                  loss_module.py, diffusion.py, confidence.py, distogram.py
    data/                  primitives/, pipelines/{preprocessing,sample_processing,featurization}/,
                           framework/{data_module,single_datasets}/
    metrics/, utils/, kernels/
  projects/of3_all_atom/   model.py, runner.py, config/{model_config,dataset_configs,features}.py
  entry_points/            experiment_runner.py, validator.py
```

`projects/of3_all_atom` is the concrete model; `core` is the reusable library. A new
project (e.g. `of3_xray`) can subclass the model and the config without forking
`core` — that is the intended extension seam and where the data head should live.

### 1.2 The trunk — `OpenFold3.run_trunk`

`projects/of3_all_atom/model.py:166`. AF3 Algorithm 1 lines 1–14:

```
InputEmbedderAllAtom(batch)  ->  s_input [*, N_tok, 449], s_init [*, N_tok, 384], z_init [*, N_tok, N_tok, 128]
for cycle in range(num_recycles + 1):          # num_recycles = 3, sampled 0..3 in training
    z  = z_init + Linear(LayerNorm(z))
    z += TemplateEmbedderAllAtom(...)           # 2 pair-stack blocks, c_t = 64
    m, msa_mask = MSAModuleEmbedder(...)
    z  = MSAModuleStack(m, z, ...)              # 4 blocks, c_m = 64
    s  = s_init + Linear(LayerNorm(s))
    s, z = PairFormerStack(s, z, ...)           # 48 blocks, c_s = 384, c_z = 128
```

Only the last cycle carries gradient (`enable_grad = is_grad_enabled and is_final_iter`).
Dimensions come from `config/model_config.py`: `c_s=384`, `c_z=128`, `c_m=64`, `c_t=64`,
`c_atom=128`, `c_atom_pair=16`, `c_token_embedder=384`, `c_token_diffusion=768`,
`c_s_input=449`, `n_query=32`, `n_key=128`, `sigma_data=16`.

The trunk is a pure function of sequence, MSA and templates. **It never sees
coordinates**, so it cannot consume experimental data in any position-dependent way.
It is the right place for *global* data context (resolution, data quality) and the
wrong place for anything map-like.

### 1.3 The diffusion module — `core/model/structure/diffusion_module.py`

`DiffusionModule.forward` (AF3 Algorithm 20):

```
si, zij  = DiffusionConditioning(batch, t, si_input, si_trunk, zij_trunk)   # + relpos, Fourier(t)
rl_noisy = xl_noisy / sqrt(t^2 + sigma_data^2)
ai, ql, cl, plm = AtomAttentionEncoder(batch, rl_noisy, si_trunk, zij)      # 3 blocks, atom -> token
ai = ai + Linear(LayerNorm(si))
ai = DiffusionTransformer(a=ai, s=si, z=zij, mask=token_mask)               # 24 blocks, c_a = 768, 16 heads
ai = LayerNorm(ai)
rl_update = AtomAttentionDecoder(batch, ai, ql, cl, plm)                    # 3 blocks, token -> atom
xl_out = c_skip(t) * xl_noisy + c_out(t) * rl_update                        # EDM preconditioning
```

Sampling (`SampleDiffusion._sample_rollout`) is the standard EDM churn loop: recentre
and randomly rotate, add churn noise, denoise, take an Euler step. Full rollout is
**200 steps × 5 samples**; the training mini-rollout is **20 steps × 1 sample**.
The one-step training objective (`OpenFold3._train_diffusion`) draws **48** noise
levels `t = sigma_data · exp(-1.2 + 1.5·n)`, `n ~ N(0,1)`, and denoises all 48 in
parallel.

Two things to note for later:

- `DiffusionTransformerBlock` already carries a **cross-attention variant**
  (`CrossAttentionPairBias`, `layers/attention_pair_bias.py:225`) selected by
  `n_query is not None`. It is *not* cross-attention to an external modality — it is
  sequence-local block attention used by the atom transformer. But the underlying
  `Attention` primitive (`primitives/attention.py:231`) takes separate `q_x` and
  `kv_x` with independent `c_q`/`c_k`/`c_v` and a list of additive `biases`
  broadcasting to `[*, H, Q, K]`. Genuine cross-attention to a new token stream
  needs **no new primitive**.
- `pocket_constraints.py` is a working precedent for injecting an external,
  user-supplied conditioning signal end-to-end: a featuriser
  (`pipelines/featurization/pocket_constraints.py:106`) emits new batch keys, and
  `SampleDiffusion.forward` branches on them to alter the rollout. Follow that shape.

### 1.4 Heads and losses

`AuxiliaryHeadsAllAtom` (`heads/head_modules.py:40`) runs the distogram head on `z`,
then **detaches** `si`, `zij` and the predicted coordinates and runs a 4-block
confidence Pairformer for pLDDT / PAE / PDE / experimentally-resolved. `OpenFold3Loss`
(`loss/loss_module.py:31`) sums confidence + diffusion + distogram, each weighted
per-sample by `batch["loss_weights"]`.

`batch["loss_weights"]` is a per-sample dict of scalars (`mse`, `bond`,
`smooth_lddt`, `plddt`, `pae`, ...) written by
`pipelines/featurization/loss_weights.py`. **This is the mechanism for mixing an
X-ray dataset into the existing training mix**: add an `xray` key, set it to 0 for
every non-crystallographic sample, and the loss silently drops out
(`if loss_weights[name].any()`).

### 1.5 Training loop

`projects/of3_all_atom/runner.py` is a Lightning module: EMA, manual gradient
clipping, per-sample parameter freezing for disabled losses
(`_get_sample_disabled_param_names`), Adam via `configure_optimizers`.
`entry_points/experiment_runner.py:446` loads checkpoints with
`strict=self.ckpt_load_settings.strict_loading`, so **new parameters can be added and
a pretrained checkpoint loaded non-strictly**. That is the fine-tuning entry point.

### 1.6 Data pipeline and cropping

`config/dataset_config_components.py:135` — `TokenCropSettings(enabled=True,
token_budget=384)` with crop weights `contiguous 0.2 / spatial 0.4 /
spatial_interface 0.4`. Some dataset configs already set
`TokenCropSettings(enabled=False)`, so uncropped operation is supported.

Features are declared with their shapes in `config/features.py`
(`feature_dict.feat`), built per-sample in `pipelines/featurization/*`, and collated
by `core/data/framework/data_module.py`. A new feature must be registered in all
three places, and must be **crop-aware** if cropping stays on.

### 1.7 Properties that matter here

| Property | Consequence for an X-ray target |
|---|---|
| SE(3)-equivariant by construction; global pose randomised every step | F_calc is *not* pose-invariant — pose must be supplied externally or eliminated |
| No unit cell, no space group, no symmetry anywhere in the model | Symmetry expansion and solvent stay entirely on the TorchRef side |
| Crops to 384 tokens | Partial-structure F_calc needed, or cropping disabled |
| Rollout under `no_grad`; only `_train_diffusion` carries gradient | Train the head on the one-step denoiser, not through the rollout |
| Predicts elements and coordinates but no ADPs and no occupancies | B-factors and occupancies must be refined by TorchRef, not predicted (initially) |
| Confidence heads see detached inputs | pLDDT is a natural, already-plumbed proxy for a per-atom weight in the X-ray target |

---

## 2. TorchRef as it stands — the other half of the seam

Everything needed on the crystallographic side already exists and is differentiable:

| Need | Where |
|---|---|
| F_calc from coordinates by FFT | `model/sf_fft.py` — `SfFFT.compute_structure_factors(hkl, xyz_iso, adp_iso, occ_iso, A_iso, B_iso, ...)`, a functional call taking raw tensors |
| F_calc by direct summation | `model/sf_ds.py` — `SfDS.compute_structure_factors`; O(N_atom · N_refl · N_sym), ideal for reflection minibatching |
| Symmetry | `symmetry/`, late reciprocal-space symmetry via `ReciprocalSymmetryExtractor` |
| Overall / anisotropic scaling, bulk solvent | `scaling/scaler.py`, `scaling/solvent.py` (`SolventModel`, mask-based) |
| Likelihood targets | `refinement/targets/xray/` — `ml`, `ml_full`, `ml_noalpha`, `rice`, `nll`, `nll_beta`, `sigma_a`, `least_squares`; `create_xray_target` factory. Every target exposes `_per_refl` (unreduced), `forward` (summed) and `get_rfactor` |
| Work / free / validation split | `XrayTarget.use_set`, `data.work` / `data.free` / `data.validation` |
| Ensembles | `experimental/ensemble/` — `EnsembleModel` (flat replicated atom list, `xyz_per_member` view, member dropout, birth/death), `EnsembleRefinement`, `RankPenaltyTarget`, `WilsonPriorTarget` |
| Multi-state mixtures | `model/model_collection.py` — `_SharedMixedModel.forward` computes `F = Σ_i w_i F_i` with learnable softmax fractions and a `set_fraction_override` hook for external weights |
| Loss orchestration | `refinement/loss_state.py` — `LossState`, hierarchical weights, lazy evaluation, `run()` closure |
| Molecular replacement | `experimental/alignment/pipeline.py` — ball-harmonic rotation function + FFT translation + rigid body. Needs the `torchref[alignment]` extra for the rotation search. (Its docstrings point at a consolidated `torchref.alignment` FRF engine that is not in this tree.) |
| Superposition | `base/alignment/superposition.py` — `align_torch`, `superpose_vectors_robust_torch` |

The crystallographic ensemble semantics TorchRef already encodes are the right ones:
Bragg intensities come from the **coherent** average `F_Bragg = Σ_m w_m F_m`
(complex), which is exactly what `EnsembleModel` gets by putting all members in one
flat atom list at occupancy `1/N`, and what `_SharedMixedModel.forward` computes
explicitly. Member spread shows up as the fall-off of `|<F>|` — the diffuse part is
not modelled, which is the standard approximation.

---

## 3. The five obstacles, and what to do about each

### 3.1 There is no crystal frame

`centre_random_augmentation` (`structure/augmentation.py:43`) applies a random
rotation and translation to the coordinates at **every** rollout step, and
`weighted_rigid_align` (`loss/diffusion.py:32`) returns `x_align.detach()` — the
alignment is explicitly outside the gradient. The model is trained to be
pose-agnostic and there is no signal anywhere that could anchor it.

Three options:

- **(A) Anchor the model.** Disable the augmentation during the X-ray stage and
  train the network to output crystal-frame coordinates. Cheapest to describe,
  worst in practice: it destroys the pretrained inductive bias, and the model
  would have to learn a global 6-DoF regression from scratch.
- **(B) Predict the pose.** Add an SE(3) head. Adds a hard sub-problem (molecular
  replacement) that classical software already solves better.
- **(C) Keep the pose outside the network — recommended.** Maintain a pose
  `T = (R, t)` as a *detached* bookkeeping variable. Obtain `T₀` by MR on the
  initial prediction; at each subsequent evaluation re-estimate `T` by Kabsch
  superposition of the current `x̂₀` onto the placed reference, optionally
  rigid-body refined every K macro-cycles. Transform `x̂₀` into the crystal frame
  to compute F_calc, and transform the resulting residual signal **back** into the
  model frame before feeding it to the network.

With (C), the network only ever sees quantities expressed in its own frame, so full
SE(3) equivariance is preserved and nothing has to be relearned. The pose is not
differentiated through — which is fine, because at the optimum
`∂L/∂T = 0` for a rigid-body-refined pose, and any residual pose error appears as a
coherent shift that rigid-body refinement removes.

One ensemble-specific detail: superpose the ensemble **as a body** (align the member
mean, apply one `T` to all members). Aligning members individually would destroy the
relative displacements that *are* the disorder model.

### 3.2 Cropping versus the unit cell

A 384-token crop is a fragment. Two fixes, use both:

- **Partial-structure background.** `F_total(h) = F_crop(h; x) + F_bg(h)` where
  `F_bg` is the complex contribution of everything outside the crop (and of the
  bulk solvent), computed once per macro-cycle from the starting model and held
  fixed. This is exactly how crystallographers refine partial models, it is one
  complex tensor of length `N_refl`, and it makes the crop's gradient correct.
- **Uncropped fine-tuning for small entries.** `TokenCropSettings(enabled=False)`
  is already supported; restrict the X-ray corpus to entries whose ASU fits the
  memory budget (roughly ≤ 1500 tokens on 80 GB with checkpointing) and disable
  cropping there.

Bulk solvent must be recomputed on the full cell periodically — the mask depends on
the model — but held fixed within a step. Standard macro-cycle practice.

### 3.3 The rollout carries no gradient

`_rollout` is wrapped in `torch.no_grad()`. So "fine-tune the produced ensemble
against data" cannot mean, at least initially, backpropagating through 200 sampling
steps. Three tractable strategies, in increasing cost:

1. **One-step objective (recommended first).** Add the X-ray term to
   `_train_diffusion`: the module already predicts `x̂₀` from a noised ground truth
   at 48 noise levels. Compute F_calc from `x̂₀` and add the likelihood to the loss.
   The head learns "given a noisy structure and the residual density, produce a
   better `x̂₀`". Cheap, stable, uses the existing loss plumbing. Reduce
   `no_samples` from 48 to 4–8 for this stage.
2. **Truncated rollout.** Enable gradient over the last k (≈ 4–8) sampling steps
   with `checkpoint_blocks`, no_grad the rest. Matches train and test conditions
   better at ~k× the cost.
3. **Guidance only, no training.** Diffusion posterior sampling: at each rollout
   step add `−λ(t) ∇_x L_xray(x̂₀)` to the update. No new weights at all.

(3) is not a fallback — it is the **baseline to build first** (§6, M1).

### 3.4 Cost

Per F_calc evaluation, per diffusion sample:

- **FFT path** — a 100 Å cell at 2.0 Å needs a ~128³ grid: density build plus a
  complex FFT is O(10 ms) on an A100. Times 48 training samples is ~0.5 s/step just
  for structure factors, on top of a step that is already ~1 s. Too much.
- **Direct summation with reflection minibatching** — sample 2000 reflections per
  step: `3000 atoms × 2000 refl × 4 sym ≈ 2.4 × 10⁷` complex MACs. Negligible, and
  the stochastic gradient over reflections is unbiased.

So: **`SfDS` + reflection minibatch for the training gradient, `SfFFT` for
full-data R-factors at validation.** Both share the same ITC92 scattering tables
(`data/itc92_scattering_factors.pt`), so they are consistent by construction — but
verify that agreement numerically once, as an integration test.

Bulk solvent is mask-based and inherently grid-bound, so `F_sol` is precomputed per
macro-cycle and gathered for the minibatch's Miller indices.

### 3.5 Ensemble semantics and overfitting

An N-member ensemble multiplies the coordinate parameter count by N. Against a
typical 2 Å dataset the observation-to-parameter ratio is already near 1 for a
single copy. `R_work` will fall for free; `R_free` is the only signal that matters.
TorchRef already has the counter-measures — `RankPenaltyTarget` (penalises the rank
and magnitude of the displacement matrix), `WilsonPriorTarget` (keeps `<|F_calc|²>`
on the Wilson curve), and `EnsembleModel.configure_dropout` (each SF evaluation uses
a random member subset). All three should be on.

Also: the free set must be **fixed per structure and stored**, not resampled, or the
free R is meaningless across epochs.

---

## 4. The proposed design

### 4.1 Principle

> Compute the crystallographic residual outside the network in the crystal frame;
> project it into per-atom, frame-invariant features; cross-attend to those.

Concretely, three streams of key/value tokens, in increasing sophistication. They
are independent and can be shipped one at a time.

### 4.2 Stream A — atom-local density probes (the informative one)

For each atom `l` with current position `x_l` (model frame), build a local frame
`F_l ∈ SO(3)`:

- protein/nucleic tokens: the standard backbone frame (`core/utils/rigid_utils.py`
  already has `Rigid`/`Rotation` and `quat_to_rot`);
- ligand / atomised tokens: Gram–Schmidt on the two nearest bonded neighbours
  (`batch["token_bonds"]` and the reference conformer give the connectivity);
- atoms with no definable frame: mask the directional probes and keep only the
  isotropic scalars.

Place `P` probe points on a fixed pattern in the local frame — e.g. 2 shells at
0.5 Å and 1.0 Å × 16 directions from a spherical design, plus the atom centre:
`P = 33`. Map each probe to the crystal frame via `T`, trilinearly sample the
residual map `Δρ = mF_o − DF_c` (and optionally `2mF_o − DF_c`), and build a token:

```
kv_A[l, p] = Linear([ Δρ(x_lp), ρ_2fofc(x_lp), shell_id_onehot(p), dir_onehot(p),
                      σ_level(Δρ), pLDDT_l, element_l ])            # -> c_kv
```

Every entry is a scalar or a *local-frame* index, so the token set rotates with the
molecule: SE(3) equivariance is preserved exactly.

Attention is atom-local: queries `[*, N_atom, 1, c_atom]`, keys/values
`[*, N_atom, P, c_kv]`. Cost is `O(N_atom · P)` — trivial next to the atom
transformer's `O(N_atom · n_key)`.

**Cheap variant worth measuring first (Stream A0).** The single most informative
per-atom quantity is the likelihood gradient `g_l = ∂L_xray/∂x_l`, which TorchRef
gives for free by autograd. Expressed in the local frame it is 3 invariant scalars
plus a norm — literally the refinement shift direction. Concatenate
`[F_lᵀ g_l, |g_l|, log|g_l|]` onto the atom conditioning `c_l` in
`AtomAttentionEncoder`. No attention needed. If this alone recovers most of the
benefit, the probe machinery may not be worth its complexity.

### 4.3 Stream B — global data-quality tokens

Pool reflections into `S ≈ 40` resolution shells and emit one token per shell:

```
kv_B[s] = Linear([ d*_s, <|F_o|>_s, <σ_F>_s, completeness_s, <I/σ>_s,
                   σ_A(s), R_work(s), n_refl_s, is_centric_frac_s ])
```

plus a handful of global scalars (space group one-hot, cell parameters normalised,
`V_M`/solvent content, resolution limits, twin fraction, overall B from Wilson).
`σ_A(s)` and `R_work(s)` come from
`refinement/model_error_estimation/sigma_a.py` and `XrayTarget.get_rfactor`, and
they change as the structure improves, which is what makes this stream more than a
static embedding.

Every token cross-attends to these `S + 1` tokens. This is what lets the model learn
"3.4 Å data, do not trust the side-chain density" — the single most common failure
mode of naive real-space fitting.

Stream B is also the only stream that makes sense in the **trunk**: a data-quality
embedding added to `s_init`/`z_init` would let the Pairformer modulate its own
confidence. Optional, second-order, and it costs a full trunk fine-tune — leave it
for later.

### 4.4 Stream C — residual feedback across rollout steps

Streams A and B are static within a step only if the map is static. The interesting
version recomputes `Δρ` from the **current** `x̂₀` every K rollout steps (K ≈ 10 of
200), so the head sees its own effect. That turns the sampler into a learned
refinement loop: predict → compute residual → attend to residual → predict again.

This is where the real gain should be, and also where the cost and the instability
are. Gate it behind a config flag and enable it only after A and B work.

### 4.5 Modules

Two new modules in a new `openfold3/core/model/xray/` package, plus a project
`of3_xray` that subclasses `OpenFold3`:

```python
# core/model/xray/data_embedder.py
class XrayDataEmbedder(nn.Module):
    """Turn precomputed X-ray features into cross-attention key/value tokens.

    Consumes only tensors placed in the batch by the featuriser / the TorchRef
    bridge; performs no crystallography itself.
    """
    def forward(self, batch, x_hat=None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # returns kv_atom [*, N_atom, P, c_kv], kv_atom_mask [*, N_atom, P],
        #         kv_global [*, S+1, c_kv],     kv_global_mask [*, S+1]
```

```python
# core/model/xray/cross_attention.py
class XrayCrossAttention(nn.Module):
    """Gated cross-attention from token/atom representations to X-ray tokens.

    The output projection is zero-initialised and the whole block is wrapped in a
    learned scalar gate, so at initialisation the module is exactly the identity and
    a pretrained OpenFold3 checkpoint is reproduced bit-for-bit.
    """
    def __init__(self, c_a, c_kv, c_hidden, no_heads, c_s=None, use_ada_layer_norm=True):
        super().__init__()
        self.layer_norm_a = AdaLN(c_a=c_a, c_s=c_s) if use_ada_layer_norm else LayerNorm(c_a)
        self.layer_norm_kv = LayerNorm(c_kv)
        self.mha = Attention(c_q=c_a, c_k=c_kv, c_v=c_kv,
                             c_hidden=c_hidden, no_heads=no_heads, gating=True)
        self.linear_out = Linear(c_a, c_a, init="final")   # zero-init
        self.gate = nn.Parameter(torch.zeros(1))           # AdaLN-Zero style

    def forward(self, a, kv, kv_mask, s=None):
        a_n  = self.layer_norm_a(a, s) if self.use_ada_layer_norm else self.layer_norm_a(a)
        bias = (self.inf * (kv_mask - 1))[..., None, None, :]
        out  = self.mha(q_x=a_n, kv_x=self.layer_norm_kv(kv), biases=[bias])
        return self.gate.tanh() * self.linear_out(out)
```

Note this reuses `core/model/primitives/attention.Attention` unchanged — it already
takes independent `c_q`/`c_k`/`c_v` and an additive bias list.

### 4.6 Where to attach

| Attachment | Stream | Why |
|---|---|---|
| `AtomAttentionEncoder.get_atom_reps`, appended to `c_l` | A0 (gradient scalars) | Cheapest, no attention, atom resolution |
| Inside the atom transformer, one `XrayCrossAttention` per block | A (probes) | Atom resolution, atom-local KV, this is where coordinates live |
| `DiffusionTransformerBlock.forward`, after `attention_pair_bias`, before `conditioned_transition` | B (global) | Token resolution, 24 blocks, cheapest place for a global signal |
| `AtomAttentionDecoder` | A | Directly shapes `rl_update`, i.e. the coordinate update itself |

Start with **B in the diffusion transformer** (simplest, no local frames needed) and
**A0 in the atom encoder**. Add **A in the atom transformer** once frames are
validated.

The diffusion-transformer edit is genuinely small:

```python
# layers/diffusion_transformer.py, DiffusionTransformerBlock.forward
a = a + self.attention_pair_bias(a=a, z=z, s=s, mask=mask, ...)
if self.xray_cross_attention is not None:            # None unless configured
    a = a + self.xray_cross_attention(a=a, kv=xray_kv, kv_mask=xray_kv_mask, s=s)
a = a + self.conditioned_transition(a=a, s=s, mask=trans_mask)
```

`xray_kv` threads through `DiffusionTransformer.forward` -> `DiffusionModule.forward`
-> `SampleDiffusion` / `_train_diffusion` the same way `z` already does, via the
`partial(...)` list built for `checkpoint_blocks`.

### 4.7 Feature and config plumbing

1. `projects/of3_xray/config/features.py` — extend `feature_dict.feat` with
   `xray_probe_values [N_ATOMS, P, C_PROBE]`, `xray_probe_mask [N_ATOMS, P]`,
   `xray_local_frames [N_ATOMS, 3, 3]`, `xray_shell_feats [N_SHELLS, C_SHELL]`,
   `xray_global_feats [C_GLOBAL]`, `xray_enabled []`.
2. `pipelines/featurization/xray.py` — the featuriser, mirroring
   `pocket_constraints.py`: returns `{}` when the query has no diffraction data, so
   mixed batches work.
3. `pipelines/featurization/loss_weights.py` — add an `xray` weight, 0 for
   non-crystallographic samples.
4. `config/model_config.py` — an `xray_head` block, `enabled: False` by default.
5. Cropping: the probe tensors are atom-indexed, so they follow the existing atom
   crop mask; the shell/global features are per-sample and unaffected.

Everything is off unless configured, so the stock model is untouched.

### 4.8 Preserving the pretrained checkpoint

- Zero-init `linear_out` and a zero-init scalar gate ⇒ the augmented model is
  **numerically identical** to the pretrained one at step 0. Verify this as a test.
- Load with `strict_loading: false` (`experiment_runner.py:446`).
- Freeze the trunk (MSA module, template embedder, Pairformer) for the first stage;
  train the diffusion module + head. `runner.py:131` already has
  `_freeze_model_params(exempt_submodule=...)` to build on.
- Consider LoRA on the diffusion transformer's attention projections rather than
  full fine-tuning — 24 blocks × `c_a=768` is a lot of parameters to move on what
  will be a comparatively small crystallographic corpus.
- Keep EMA on; the runner already maintains it.

---

## 5. Fine-tuning the ensemble against data

### 5.1 What "the ensemble" means

`SampleDiffusion` returns `xl [*, N_samples, N_atom, 3]` — 5 samples at inference.
Treated as a crystallographic ensemble with weights `w_m`:

```
F_calc(h) = Σ_m w_m · F_m(h)          (coherent, complex — correct for Bragg data)
```

Two ways to get this in TorchRef:

- **`EnsembleModel`** — one flat atom list of `N_samples × N_atom` at occupancy
  `1/N`, single FFT. Fastest, and `xyz_per_member` is a live view so gradients flow
  back to the per-member coordinates.
- **`_SharedMixedModel`** — one `ModelFT` per member with learnable softmax
  fractions, and `set_fraction_override` to drive `w_m` from outside (e.g. from a
  softmax over OpenFold3's own confidence). Useful if you want the *weights* to be
  predicted rather than uniform.

Start with `EnsembleModel` at uniform weights; add learned weights only once uniform
works. If you do learn them, the natural source is a small head on `si_trunk`
pooled per sample — but note that pLDDT is a *geometric* confidence, not a
population estimate, so do not use it as an occupancy without recalibration.

### 5.2 The loss

```
L = L_of3                                 # existing: diffusion MSE, bond, smooth-LDDT, distogram, confidence
  + w_xray   · L_ML(work reflections)     # refinement/targets/xray, 'ml' or 'rice'
  + w_rank   · L_rank                     # RankPenaltyTarget
  + w_wilson · L_wilson                   # WilsonPriorTarget
```

Deliberately: **no geometry restraints added on the OpenFold3 side.** The diffusion
loss already carries bond and smooth-LDDT terms and the model's own prior is far
stronger than a bond-length restraint — `EnsembleRefinement` makes the same choice
and documents it. Adding a second geometry term risks double-counting.

`L_ML` should be normalised per reflection and per asymmetric unit so `w_xray` is
transferable across datasets of wildly different size. `EnsembleRefinement`'s
"per-ASU loss scale" (`_create_loss_state`) is the convention to copy.

Free reflections are **never** in the loss — register a `use_set="free"` target for
monitoring only, at weight 0.

### 5.3 What is actually being optimised

Be explicit about which of three quite different things is meant by "fine-tune the
ensemble against data", because they need different machinery:

1. **Per-structure inference-time optimisation.** Freeze the weights; optimise the
   sampled coordinates directly against the data. This is just ensemble refinement
   with an OpenFold3 starting model, and TorchRef does it today — `EnsembleModel`
   + `EnsembleRefinement`. No neural training at all. *Do this first; it is the
   scientific control.*
2. **Guidance.** Freeze the weights; bias the sampler with `∇_x L_xray`. Changes the
   distribution the model samples, per structure, at inference. No training.
3. **Training the head.** Update weights so that the model, given data, produces
   better ensembles *in general*. This is the actual ask, and it only makes sense if
   (1) and (2) already demonstrate that the data contains signal the model is
   missing.

### 5.4 Corpus and evaluation

- **Corpus.** PDB X-ray entries with deposited structure factors, filtered to:
  resolution ≤ 2.5 Å; ASU matching the OpenFold3 query assembly (or a clean chain
  mapping); no severe twinning; free flags present. PDB-REDO gives a consistently
  re-refined version worth using as the label. Expect tens of thousands of usable
  entries after filtering — an order of magnitude smaller than the OpenFold3
  training corpus, which is the strongest argument for adapters over full
  fine-tuning.
- **Leakage.** OpenFold3 has already seen most of these *structures*. Any claim that
  the data head helps must be evaluated on a **temporal split** — entries released
  after the checkpoint's training cutoff — or the head will merely be re-deriving
  memorised coordinates.
- **Metrics.** `R_free` first and foremost, against three baselines: (i) deposited
  model, (ii) the OpenFold3 prediction rigid-body placed and B-factor refined only,
  (iii) that same prediction put through standard TorchRef refinement. Then
  real-space CC, and ensemble-specific measures — does the spread reproduce known
  alternate conformers (compare against qFit / multi-conformer depositions)?
- **The honest failure mode to watch for:** an ensemble can lower `R_free` purely by
  acting as a smarter B-factor model, without any of the members being individually
  meaningful. Check per-member geometry and per-member real-space fit, not just the
  aggregate.

---

## 6. Staged plan

| Stage | Deliverable | Success criterion |
|---|---|---|
| **M0** | Bridge: OpenFold3 prediction → placed, scaled TorchRef model. MR pose, `Scaler.initialize()`, solvent, free-flag handling. Round-trip test `SfDS` vs `SfFFT`. | `R_free` for a rigid-body-placed OpenFold3 model reproduces what PHENIX reports for the same placement, within a percent |
| **M1** | **Guidance baseline.** `∇_x L_xray` injected into `SampleDiffusion._sample_rollout`, weights frozen. | Guided 5-sample ensemble beats unguided on `R_free` on a 20-structure temporal-split test set |
| **M2** | **Ensemble refinement control.** OpenFold3 samples → `EnsembleModel` → `EnsembleRefinement`. | Quantifies the ceiling: how much `R_free` is available from these coordinates at all |
| **M3** | **Stream A0 + B, one-step training.** Gradient scalars on `c_l`, shell tokens cross-attended in the diffusion transformer, trained through `_train_diffusion` with `no_samples` reduced. Zero-init identity test. | Head-on beats head-off at matched compute on held-out `R_free`; identity test passes exactly |
| **M4** | **Stream A (probe cross-attention).** Local frames, probe sampling, atom-transformer cross-attention. | Beats M3 |
| **M5** | **Stream C (residual feedback) + truncated-rollout gradient.** | Beats M4; sampler is stable over 200 steps |

M0–M2 need no OpenFold3 changes at all — they are TorchRef work plus a thin adapter,
and they are what tells you whether M3+ is worth building.

---

## 7. Open questions

- **ADPs and occupancies.** OpenFold3 predicts neither. Without B-factors an X-ray
  target is badly mis-specified — `R_free` will be dominated by the missing thermal
  model. Either refine B's in TorchRef inside the loop (cheap, well-posed,
  recommended) or add a per-atom B head. An ensemble with individually-refined B's
  double-counts disorder, so pick one: frozen small B + ensemble spread (what
  `EnsembleModel` does today), or refined B + single copy.
- **Hydrogens.** OpenFold3 output is heavy-atom; TorchRef can generate riding
  hydrogens (`Model.hydrogenate`). Matters below ~1.5 Å.
- **Assembly vs. asymmetric unit.** OpenFold3 predicts a biological assembly; the
  crystallographic ASU may be a part of it or contain several copies. Needs an
  explicit mapping step and probably a filter on the corpus.
- **Anomalous / multi-wavelength data**, twinning, translational NCS — all out of
  scope for a first version, all present in the PDB corpus, all need filtering.
- **Diffuse scattering.** The coherent-average model discards it. It is precisely the
  observable most sensitive to the ensemble. Out of scope, but it is the reason a
  Bragg-only ensemble target is a weaker constraint on disorder than it first looks.
- **Latent-space control (ConforNets)** is a complementary mechanism, not an
  alternative one — see §9.

---

## 8. What this would cost

Rough, for the recommended path:

- M0–M2: adapter code, a curated test set, no training. Weeks, one person, no GPU
  fleet.
- M3: fine-tuning the diffusion module + head on a few tens of thousands of
  entries. With the trunk frozen, `no_samples` at 8, and uncropped ASUs under ~1500
  tokens, this is a small-cluster job, not a foundation-model run.
- M4–M5: the residual feedback loop multiplies the per-step cost by the number of
  map refreshes. Budget accordingly, and only after M3 has shown a signal.

---

## 9. ConforNets, and why latent control is the wrong actuator for data fitting

Lee, Kalicki, Jeon, Qabel, Fadini and AlQuraishi, *ConforNets: Latents-Based
Conformational Control in OpenFold3*, arXiv:2604.18559 (2026), CC-BY. Code:
`aqlaboratory/confornets`. The paper's own chassis is OpenFold3-preview, so it is
directly comparable to everything above.

### 9.1 What it is

A ConforNet is a **channel-wise affine transform of the pre-Pairformer pair
latents**:

```python
class ConforNet(nn.Module):                      # confornet/core/confornet.py, verbatim
    def __init__(self, dim):
        self.W = nn.Parameter(torch.eye(dim))    # dim = c_z = 128
        self.b = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        return torch.matmul(x, self.W.t()) + self.b
```

`W ∈ R^{128×128}`, `b ∈ R^{128}`, identity-initialised, applied to `z_pre` at the
**last recycle only** (all earlier recycles run under `no_grad`), then propagated
through the 48-block Pairformer and the diffusion rollout.

Two tasks:

- **Diverse conformation prediction** — `k` ConforNets (k = 21 in the paper) are
  jointly optimised *per protein* to maximise pairwise distance between the `k`
  predicted structures. 20 Adam steps, lr 1e-3 halved every 5, grad clipped at 10.
  Objectives: pairwise coordinate MSE after Kabsch alignment, or pairwise MSE
  between distogram CDFs. The distogram objective needs no rollout at all.
- **Conformational transfer** — supervised: train `ϕ` on one protein against a known
  target state, then apply the *same* `ϕ` to a different protein of the same family.
  This is the paper's genuinely novel claim, and it only works because `ϕ` carries
  no positional index.

The location ablation is the informative part: they compared `z_pre`, `z_post`,
`s_pre`, `s_post`. `z_pre` gives control that survives full diffusion; perturbing
post-Pairformer latents "can fit the mini rollout but degrade under full diffusion,
most notably for `s_post`, suggesting shortcut solutions that do not survive longer
denoising trajectories."

### 9.2 Why it cannot do our job

The reason is not parameter count — 16,512 parameters is *more* than the 3N ≈ 7,000
coordinate degrees of freedom of a 300-residue protein. It is that the actuator is
the wrong shape:

1. **It carries no positional index.** The *same* affine map is applied to every
   `(i, j)` pair of `z_pre`. A ConforNet can say "re-express all contacts in a
   rotated channel basis"; it structurally cannot say "there is unexplained density
   near residue 57." That positional specificity is the entire content of a
   crystallographic residual. It is also exactly what makes transfer across a family
   work — the property that gives the method its reach is the property that
   disqualifies it here.
2. **It selects among priors rather than injecting information.** The paper's own
   reading of its cryptic-pocket result is that "latent-space exploration focuses on
   physically feasible and energetically accessible degrees of freedom" — it
   reweights what the model already believes. A 2.0 Å dataset for that same protein
   is ~25,000 unique reflections, i.e. ~50,000 independent numbers of *new*
   information. There is no route by which a global basis change of the pair
   representation transmits that.
3. **Wrong resolution.** ConforNets act on token-level pair latents, and the
   distogram they optimise against is binned 3.25–50.75 Å in 39 bins. X-ray
   refinement at 1.5–2.5 Å is about atoms — rotamer flips, 0.5–2 Å alternate
   conformations, occupancies, ADPs. The representation being modulated cannot
   express the quantities the data constrains.
4. **The signal has to survive the whole trunk and rollout.** The `s_post` result
   shows how fragile that channel is even for coarse, whole-domain motions. A
   sub-Ångström per-atom correction routed through 48 Pairformer blocks is a very
   long lever on a very small screw.
5. **No population weights.** ConforNets produce `k` distinct structures with no
   occupancies. The paper is explicit that calibrated Boltzmann-weighted ensembles
   are the goal of a *different* line of work. Fitting an ensemble to Bragg data
   requires weights — `F = Σ_m w_m F_m` is meaningless without them.
6. **Per-protein optimisation.** In the diversity setting the ConforNets are
   retrained for each protein, which limits reuse — though the transfer experiments
   are precisely the attempt to escape that.

Note what is *not* on this list: "optimising coordinates directly would be more
direct." Directness is not the merit here — see §11. The objection to ConforNets is
about information capacity and positional resolution, not about working through the
model.

So the instinct is right: **latent modulation is a prior-side actuator.** It is a
poor conditioning channel for an experimental dataset, and it is the wrong place to
attach an X-ray target.

### 9.3 Where it does earn its place

One thing latent control does that a data gradient never will: **it crosses
basins.** Any likelihood gradient — guidance, or a data head, or classical
refinement — is local. It will not carry a kinase from DFG-in to DFG-out or a
transporter from inward- to outward-open, because the barrier is tens of Ångström
of coordinate travel through structures of terrible fit.

That suggests a division of labour rather than a competition:

```
latent / MSA perturbation  ->  k diverse, physically plausible basins   (proposal)
X-ray likelihood            ->  score each basin by LLG / R_free        (selection)
X-ray data head or guidance ->  refine the winner(s) within its basin   (refinement)
```

That is a real workflow, and it is most valuable precisely where crystallography is
hardest: molecular replacement into a low-resolution or conformationally-ambiguous
target, where the search model is in the wrong state and no amount of rigid-body or
restrained refinement will fix it. Generating `k` ConforNet-diversified search
models and ranking them by translation-function score or LLG is a concrete,
cheap experiment that needs none of the machinery in §4.

### 9.4 What to reuse from the codebase

`confornet/core/diffusion.py::diffusion_sample` is a **deterministic DDIM rollout
with a fixed initial-noise argument, fully differentiable end to end** — no
`no_grad` anywhere. That is milestone M5 of §6, already written and demonstrated.
Two things follow:

- The gradient path through the rollout is empirically tractable: they report the
  20 Pairformer backprop steps fitting in 40 GB at ~300 aa and 80 GB at ~600 aa with
  per-block gradient checkpointing, at roughly 2–3× default OF3p sampling cost. Our
  case is cheaper on one axis — we do not need to backprop through the Pairformer at
  all if the head lives in the diffusion module.
- The **fixed-noise reparametrisation** is the key trick for our problem, and it is
  worth stating plainly: holding the initial noise fixed makes the sampled ensemble
  a *deterministic differentiable function* of the conditioning. That is precisely
  what turns "fine-tune the produced ensemble against data" from a
  reinforcement-learning problem into a plain gradient-descent one. Adopt it.

`confornet/core/trunk.py::run_trunk_with_confornet` also shows the practical shape of
"run the recycles under `no_grad`, take the gradient only through the last pass" —
directly applicable if we ever do want a trunk-side data-quality embedding (§4.3).

---

## 10. Nuisance parameters: profile the likelihood, backprop only to coordinates

### 10.1 The formulation

Split the parameters of the crystallographic likelihood into the coordinates we want
a gradient for and everything else:

```
L( x , θ )      θ = ( pose T , ADPs B , scale k/U_aniso , solvent k_sol/B_sol , sigma_A )
```

Define the **profile likelihood** `L̃(x) = L(x, θ*(x))` with `θ*(x) = argmin_θ L(x, θ)`.
Then

```
dL̃/dx  =  ∂L/∂x |_{θ*}  +  (∂L/∂θ)|_{θ*} · dθ*/dx
                              ^^^^^^^^^^^^ = 0 at the inner optimum
```

The second term vanishes by the envelope (Danskin) theorem. **So the nuisance fit can
be run under `no_grad`, `.detach()`-ed, and the coordinate gradient is still exact.**
No unrolling of the inner optimiser, no implicit-function theorem, no backprop through
LBFGS. This is variable projection, and it is the right shape for the problem.

Three things it buys beyond the obvious cost saving:

- **Conditioning.** Overall scale, overall B and coordinate error are near-degenerate
  with one another. Profiling them out removes the flattest directions from the outer
  problem — the classic VarPro result.
- **Transferable loss scale.** §5.2 flagged that `w_xray` has to be normalised per
  reflection and per ASU to transfer across datasets. Profiling the scale out fixes
  that structurally rather than by convention.
- **A clean training signal.** The gradient reaching the diffusion module is then a
  pure coordinate signal. Without profiling, the head would partly learn to
  compensate for a mis-set scale, which is not a transferable skill.

### 10.2 Where the envelope argument actually holds, and where it does not

The theorem needs `∂L/∂θ = 0`. That is a real precondition, not a formality.

- **Scale and solvent — safe.** Small, well-conditioned, converge tightly and cheaply.
  `Refinement.refine_scaler` already warm-starts across macrocycles.
- **Pose — needs care.** Kabsch superposition onto a reference minimises RMSD, *not*
  the likelihood, so `∂L/∂T ≠ 0` and the envelope argument does not hold for it. Use
  Kabsch only as a cheap inter-step tracker to stop the frame drifting; do an actual
  rigid-body refinement against the data at macrocycle boundaries
  (`refinement/rigid_body_refinement.py::RigidBodyRefinementStep`) so the condition
  genuinely holds. For an ensemble, refine one body pose on the member mean (§3.1).
- **sigma_A — must be cross-validated.** Fitted on the work set it absorbs model error
  and flattens exactly the gradient we want. TorchRef already does the right thing:
  `model_error_estimation/sigma_a.py::estimate_beta` takes a `free_mask` and fits on
  the free set.
- **ADPs — the trap.** B-factors and coordinate error are strongly correlated. Freely
  refined per-atom B will absorb the coordinate error we are trying to backpropagate,
  and it will do so *preferentially where the model is worst* — attenuating the
  gradient precisely where it carries the most information. This is the one nuisance
  that can silently invert the whole scheme.

  Mitigations, in order of preference: restrain B
  (`ADPSimilarityTarget`, `ADPLocalityTarget`, `RigidBondTarget`); group B per residue
  or use TLS at lower resolution; or seed B from pLDDT and refine only a global scale
  on top. Keep the number of B parameters well under what the resolution supports.

  With an ensemble there is a second version of the same problem: **per-atom B and
  ensemble spread both model disorder, and refining both double-counts.** Pick one.
  `EnsembleModel` already takes the position that the spread *is* the disorder model
  and holds a small constant B — keep that, and do not turn per-atom B refinement on
  for the ensemble path.

### 10.3 Schedule

```
every N steps, under no_grad:
    rigid-body refine T against the data          # so dL/dT ~ 0
    refine k, U_aniso, k_sol, B_sol               # warm-started scaler
    refine B (restrained / grouped)               # or leave frozen for ensembles
    estimate sigma_A on the FREE set
    detach all of the above

every step, with grad:
    F_calc(x̂0 ; θ*) on a reflection minibatch     # SfDS
    L̃  ->  x̂0  ->  diffusion module / data head
```

The nuisances move far more slowly than the coordinates, so `N` in the tens is fine —
and each refresh is full-data while the coordinate gradient is minibatched, which is
also the only consistent choice (scale and sigma_A are per-shell quantities and cannot
be estimated from a random reflection subset).

One practical caution for the neural setting: keep resolution-shell edges and free
flags **fixed per structure**. If the inner solve changes discretely between steps —
rebinned shells, a rigid-body solution that jumps — `L̃` becomes non-smooth in `x` and
the outer optimisation will show it.

### 10.4 Batching, and what batches over what

Three independent axes, easy to conflate:

| Axis | Where | Layout |
|---|---|---|
| **Atom blocks** | OF3 atom-level attention | `[*, N_blocks, N_query, ...]`, `n_query=32`, `n_key=128` |
| **Diffusion samples** | OF3, inserted by `unsqueeze(1)` in `OpenFold3.forward` | `[*, N_samples, N_atom, 3]` |
| **Reflections** | TorchRef side, our addition | minibatch over `hkl` for the gradient |

The atom axis is the one that matters for where the data head goes. OF3's atom
attention is **sequence-local block attention**, not full attention: atoms are padded
to a multiple of `n_query=32` (`get_query_block_padding`), reshaped to
`[*, N_blocks, N_query, c_atom]`, and each block attends to `n_key=128` keys centred
on it (`partition_atom_indices` in `core/utils/atom_attention_block_utils.py`). The
atom pair representation `plm` is carried in that same block layout
`[*, N_blocks, N_query, N_key, c_atom_pair]` specifically so no `N_atom × N_atom`
tensor is ever formed. The token-level `DiffusionTransformer` is *not* blocked —
`n_query=None` there, full attention over `N_token`.

The per-atom probe cross-attention of §4.2 fits that layout natively: queries
`[*, N_blocks, N_query, 1, c_atom]`, keys/values `[*, N_blocks, N_query, P, c_kv]`.
Each atom attends only to its own `P` probes, so it is embarrassingly batched, adds no
cross-atom coupling, and — like the rest of the atom stack — never materialises
anything quadratic in `N_atom`.

---

## 11. Fine-tuning: the conditioning argument, and what it forces

### 11.1 The premise

Cartesian refinement against a crystallographic likelihood has a small radius of
convergence — roughly an Ångström, less at low resolution. The reason is structural:
the target is a sum over reflections of terms oscillating as `exp(2πi h·x)`, so the
landscape carries spurious minima spaced at ~`d_min`. The whole history of the field
is workarounds for this: simulated annealing (CNS, `phenix.refine`), torsion-angle
refinement (Rice & Brünger — a reparametrisation onto collective coordinates, chosen
purely for its convergence properties), low-resolution-first protocols.

A diffusion model is a better-conditioned optimiser for the same problem, and not
incidentally. The denoiser at noise level `σ` is the score of the data distribution
smoothed at scale `σ`, so annealing the noise schedule *is* a continuation method on
a learned prior — coarse-to-fine, with each step reprojecting onto the manifold of
structures that look like refined PDB entries. That is exactly the prior a
crystallographic target wants, and it is a far better-behaved landscape than 3N
independent Cartesian shifts under geometry restraints.

So the reason to work through the model is convergence, not convenience. Everything
below follows from taking that seriously rather than treating the diffusion module as
a coordinate generator with a loss bolted on.

Two consequences worth stating plainly before the details:

- **The one-step objective of §5.3 is not sufficient.** Training the denoiser to
  produce a better `x̂₀` from a lightly-noised ground truth teaches local correction —
  precisely the Cartesian-refinement regime whose conditioning is the problem. The
  claimed benefit lives in the *trajectory*, so the gradient has to run through
  several denoising steps. The truncated differentiable rollout moves from "later
  optimisation" (M5) to a precondition.
- **The data head has to fire early in the rollout, not just at the end.** A head
  that only contributes below `t ≈ 0.5 Å` cannot influence which basin the sampler
  lands in; it can only polish. Stream C (§4.4, residual feedback across steps) is
  therefore core, not optional.

### 11.2 The noise level is a B-factor — and OF3's default schedule is unusable

This is the quantitative constraint that governs everything else.

Isotropic Gaussian displacement of standard deviation `u` per Cartesian component
multiplies structure factors by `exp(-2π²u²/d²)`, i.e. it is exactly a Debye–Waller
factor with

```
B_eff = 8π² u²           d(e⁻¹ attenuation) = π√2 · u ≈ 4.44 u
```

A diffusion noise level `t` is such a displacement. So:

| `t` (Å) | `B_eff` (Å²) | resolution where signal survives |
|---|---|---|
| 0.3 | 7 | 1.3 Å |
| 0.5 | 20 | 2.2 Å |
| 1.0 | 79 | 4.4 Å |
| 4.8 | 1834 | 21 Å |

OF3 trains with `t = σ_data · exp(-1.2 + 1.5n)`, `n ~ N(0,1)`, `σ_data = 16 Å`. The
**median training noise level is `16·e^{-1.2} ≈ 4.8 Å`** — an effective B of ~1800 Å²,
which annihilates everything past 21 Å. And `P(t < 0.5 Å) ≈ 6.5%`: of the 48 noise
levels drawn per training step, roughly **three** carry any signal at 2 Å resolution.

Two things follow:

1. **Re-weight the noise distribution for the X-ray stage.** Sampling from the stock
   distribution spends >90% of the compute on samples where the X-ray term is
   numerically zero. Use a dedicated low-`t` schedule for the X-ray loss (or apply the
   term only below a `t` threshold and raise `no_samples` in that band), while keeping
   the stock distribution for the existing diffusion MSE so the model's general
   denoising is not degraded.
2. **Use `t` to pre-select reflections.** At noise level `t`, reflections beyond
   `d ≈ 4.4t` cannot be informative. Dropping them from the minibatch is free
   efficiency and costs nothing statistically.

### 11.3 Let sigma_A do the annealing

The table above is the right intuition but the wrong statistical object: `t` is the
blur on the *noisy input* `x_t`, whereas the loss is evaluated on `x̂₀ = E[x₀ | x_t]`,
a posterior mean whose error is smaller than `t` and is not Gaussian.

The correct treatment already exists, and it is the standard crystallographic one:
**`σ_A` is exactly the parameter that measures how much of the model to believe at
each resolution.** With `σ_A` estimated properly, the ML target self-anneals — at high
`t` the model error is large, `σ_A → 0` at high resolution, and those reflections
contribute nothing to the gradient without anyone hand-designing a resolution
schedule. The coarse-to-fine behaviour that makes the diffusion landscape well
conditioned is thereby mirrored in the target rather than bolted onto it.

The one requirement: **estimate `σ_A` per (resolution shell × noise-level bin)**, on
the free set, since it varies strongly with `t`. `estimate_beta` in
`model_error_estimation/sigma_a.py` already fits per shell on a `free_mask`; bin by
`t` as well and cache per macrocycle (§10.3). Use `t` only for the cheap reflection
pre-selection of §11.2; let `σ_A` carry the weighting.

### 11.4 What actually gets trained

```
frozen        MSA module, template embedder, Pairformer trunk
trained       XrayDataEmbedder + XrayCrossAttention (zero-init gate)
              LoRA on the diffusion transformer attention projections
              LoRA or full on AtomAttentionDecoder
gradient      truncated differentiable rollout, last k ≈ 4–8 steps of a
              deterministic DDIM trajectory with fixed initial noise
loss          L_of3 (stock weights)  +  w_xray · L̃_ML  +  w_rank + w_wilson
```

`L̃_ML` is the profiled likelihood of §10 — pose, scale, solvent, ADPs and `σ_A` fitted
under `no_grad` and detached, so the gradient reaching the network is a pure
coordinate signal.

The **fixed-noise reparametrisation** (§9.4) is what makes this a plain
gradient-descent problem: hold the initial noise fixed and the sampled ensemble is a
deterministic differentiable function of the conditioning. `diffusion_sample` in the
ConforNets codebase is a working implementation, with the memory profile measured.

The corpus is small — tens of thousands of entries against OF3's pretraining set — so
adapters rather than full fine-tuning, EMA on, and hold out whole *structures*, not
reflections, on a temporal split past the checkpoint cutoff.

### 11.5 The experiment that decides whether any of this is worth building

The conditioning premise is testable directly, and cheaply, before a single weight is
trained. Take deposited structures at a range of resolutions and construct starting
models with controlled coordinate error — MR solutions, or perturbed depositions — at
0.5, 1, 2, 3 Å RMSD. Then compare, on identical starting points:

1. Cartesian LBFGS refinement (TorchRef, stock);
2. simulated annealing / torsion-angle refinement (PHENIX, as the classical ceiling);
3. diffusion-mediated refinement — partial noising to `t`, then a data-guided rollout.

Measure the fraction recovered below 1 Å, and final `R_free`, as a function of
starting error. If (3) does not have a visibly larger radius of convergence than (1)
and (2), the premise is wrong and none of §11.4 is worth the compute. If it does, the
size of that gap is the budget the fine-tuning has to work with.

### 11.6 The honest counter-argument

Better conditioning is not the same as being able to reach the answer, and the prior
that supplies the conditioning is also a floor:

- The prior is over *deposited, refined* structures — extremely well matched to
  crystallography, but it will resist moving to a data-supported conformation it
  considers unlikely. Strained loops, genuine alternate conformers, and ligand-induced
  distortions are exactly the cases where the data is right and the prior is wrong.
- The model knows nothing about crystal contacts, while deposited conformations are
  frequently shaped by them. That is a systematic disagreement, not noise.
- A sampler that lowers `R_free` by reverting to its own dominant mode has not used
  the data at all. The M1/M2 controls exist to detect precisely that, and the metric
  that catches it is not `R_free` but whether the data-guided ensemble differs from
  the unguided one in the places the difference density says it should.
