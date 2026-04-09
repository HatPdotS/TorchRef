# Extended Figures Design — TorchRef IUCrJ Paper

## Extended Figure 1: Resolution-Binned R-Factor Comparison

**Purpose:** Show that the ~1 pp R-factor gap between TorchRef and Phenix is uniform across resolution, rather than concentrated in a particular regime (e.g. low resolution where weighting matters more, or high resolution where gradient accuracy matters more).

**Data source:** The existing 1,000-structure benchmark. Both TorchRef and Phenix R-work/R-free values are already available from the REFMAC5 zero-cycle validation. Resolution for each structure is in the MTZ headers.

**Panel layout (single figure, two panels side by side):**

- **Panel A:** ΔR-free (TorchRef minus Phenix) vs resolution. Each dot is one structure. Overlay a running median (or bin the data into ~0.25 Å bins and show box plots). Horizontal dashed line at zero. This directly shows where TorchRef wins or loses.

- **Panel B:** Same for ΔR-work.

**Axes and styling:**
- X-axis: resolution (Å), reversed (high resolution on the left, as is convention), range 1.0–3.0 Å.
- Y-axis: ΔR (percentage points), symmetric around zero, roughly −3 to +5.
- Color: single color for scatter, darker for the running median line.
- Add a text annotation with the overall median ΔR-free and ΔR-work values.

**Implementation notes:**
- Data is already computed — this is a plotting task.
- Use the REFMAC5 validation R-values (not the program-reported ones) for fairness.
- Consider also binning by number of atoms or number of reflections as a secondary analysis, though this probably doesn't need its own panel.

---

## Extended Figure 2: Refinement Convergence Traces

**Purpose:** Demonstrate that TorchRef converges smoothly and monotonically over refinement cycles, and that the convergence rate is comparable to Phenix.

**Data source:** Re-run a small number of representative structures (3–5) through both TorchRef and Phenix, logging per-cycle R-work and R-free. Choose structures spanning the resolution range:
- One high-resolution structure (~1.2–1.5 Å)
- One medium-resolution structure (~2.0 Å)
- One low-resolution structure (~2.8–3.0 Å)
- Optionally: one structure where TorchRef outperforms Phenix and one where Phenix outperforms TorchRef (from the tails of the ΔR distribution in Extended Figure 1).

**Panel layout (single figure, one panel per structure, arranged as a row or 2×2 grid):**

Each panel shows:
- X-axis: macro cycle number (1–10)
- Y-axis: R-factor
- Four lines: TorchRef R-work (solid blue), TorchRef R-free (dashed blue), Phenix R-work (solid orange), Phenix R-free (dashed orange)
- Starting point (cycle 0) is the shaken model R-values
- Title: PDB code, resolution, number of atoms

**Implementation notes:**
- TorchRef logs per-cycle statistics by default — extract from refinement logs.
- Phenix: parse the per-macro-cycle R-factors from the Phenix log file.
- Make sure both start from the identical shaken model (same as the benchmark).
- If any structure shows non-monotonic behavior (R-free going up then down), that's actually interesting and worth including — it would show that the weighting scheme self-corrects.

---

## Extended Figure 3: GPU Memory Scaling

**Purpose:** Help users assess whether their structure fits within their GPU's memory and contextualize the OOM errors observed at 8 GB. This is practical information that users need.

**Data source:** Profile peak GPU memory usage across a range of structure sizes. Either:
- (a) Sample ~20–30 structures from the benchmark spanning a range of atom counts (500 to 20,000+ atoms) and reflection counts, run one TorchRef refinement cycle on GPU, record peak memory via `torch.cuda.max_memory_allocated()`.
- (b) Use the 1DAW benchmark structure and artificially vary the grid size / reflection count by changing resolution cutoffs.

Option (a) is more informative for users.

**Panel layout (single figure, two panels):**

- **Panel A:** Peak GPU memory (GB) vs number of atoms. Scatter plot with a fitted trend line (should be roughly linear in the number of atoms × number of reflections, but atom count is the more intuitive axis). Mark the 8 GB and 16 GB lines as horizontal dashed references (common GPU memory sizes). Annotate the approximate structure size limits for each GPU tier.

- **Panel B:** Peak GPU memory vs number of unique reflections. Same style. This captures the FFT grid contribution, which scales with the number of reflections (or equivalently, the unit cell volume × resolution).

**Axes and styling:**
- X-axis: number of atoms (Panel A) or number of reflections (Panel B), linear scale.
- Y-axis: peak memory in GB, linear scale.
- Horizontal dashed lines at 8, 16, 24, 40, 80 GB (common GPU memory tiers: consumer, A5000, A100 40G, A100 80G).
- Color points by resolution to show that resolution is the hidden variable (higher resolution = more reflections = more memory at the same atom count).

**Implementation notes:**
- Use `torch.cuda.reset_peak_memory_stats()` before and `torch.cuda.max_memory_allocated()` after each run.
- Run with `torch.no_grad()` for forward-only memory, and with gradients for the full refinement memory footprint. Report both if they differ substantially.
- Include the overhead from the loss functions, not just F_calc — this is what actually causes the OOM.

---

## Extended Figure 4: Structure Factor Calculation — Splatting Optimization Breakdown

**Purpose:** Quantify the contribution of the decomposed splatting approach vs the trivial approach vs the Triton GPU kernel. The Performance section describes the optimization in detail but provides no quantitative comparison between the approaches.

**Data source:** The 1DAW benchmark structure. Run the structure factor calculation using:
1. Trivial splatting (CPU, 1 core)
2. Decomposed splatting (CPU, 1 core)
3. Decomposed splatting (CPU, 4 cores)
4. Trivial splatting (GPU, no Triton)
5. Triton kernel (GPU)

Also break down the total F_calc time into its three stages:
- Electron density splatting
- FFT
- Symmetry expansion and extraction

**Panel layout (single figure, two panels):**

- **Panel A:** Grouped bar chart comparing the five approaches listed above. Y-axis: wall-clock time per F_calc (ms), log scale. This shows the speedup from each optimization.

- **Panel B:** Stacked bar chart showing the time breakdown into the three stages (splatting, FFT, extraction) for each approach. This reveals which stage is the bottleneck and where the decomposition helps.

**Axes and styling:**
- Y-axis: time in ms, log scale for Panel A, linear for Panel B.
- Color: one color per stage in Panel B (e.g. blue for splatting, orange for FFT, green for extraction).
- Add speedup annotations (e.g. "2.1×" above bars relative to the trivial baseline).

**Implementation notes:**
- TorchRef presumably has a flag or internal switch to select between trivial and decomposed splatting. If not, temporarily modify the code to benchmark both paths.
- For the GPU comparison (trivial vs Triton), the text says "the performance uplift over the trivial approach is negligible" — this figure would quantify that claim.
- Time each stage independently using `torch.cuda.synchronize()` between stages for GPU timing.
- The text states "11×3×4 = 132 evaluations vs 555×4 = 2220 evaluations" — include these theoretical speedup numbers as annotations to connect the algorithmic argument to the measured wall-clock times.