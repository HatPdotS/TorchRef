"""Choice lists the CLI parsers show in ``--help``, held without importing torch.

Each mirrors the table named beside it, which stays the source of truth and validates
the parsed value; importing that table imports torch, which ``--help`` must not await.
"""

#: :data:`torchref.maps.ded_weights.SCHEMES` and ``DEFAULT_SCHEME``.
DED_WEIGHT_SCHEMES, DEFAULT_DED_WEIGHT_SCHEME = ("none", "inverse_variance", "q"), "q"

#: ``difference_power.SIGMA_SCALE_BOUNDS`` (``refinement.model_error_estimation``).
SIGMA_SCALE_BOUNDS = (0.1, 10.0)

#: :data:`torchref.scaling.scaler_base.SCALE_TARGETS` and ``DEFAULT_SCALE_TARGET``.
SCALE_TARGETS, DEFAULT_SCALE_TARGET = ("nll", "ml_noalpha", "ls"), "ls"

#: The rows of :data:`torchref.refinement.targets.xray._specs.XRAY_TARGETS`, summarised.
XRAY_TARGET_DOCS = {
    "ml": "Read MLF, variance epsilon*beta, mean alpha*|F_calc| (default).",
    "ml_noalpha": "As 'ml' with the Luzzati mean coupling fixed at 1.",
    "ml_full": "Full-form MLF, marginalising the error-free amplitude. ~4x the cost.",
    "nll_beta": "Gaussian amplitude NLL on ml's model-error variance (diagnostic).",
    "nll": "Gaussian amplitude NLL on sigma_obs only; does not control overfitting.",
    "nll_i": "Gaussian NLL on the observed intensities, skipping French-Wilson.",
    "ls": "Least squares with unit weights; the scaler owns the overall scale.",
    "ls_wunit_k1": "Phenix-style least squares, one global scale per gradient call.",
}
