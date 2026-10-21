"""Choice lists the CLI parsers show in ``--help``, held without importing torch.

Each name mirrors a table defined beside the code that uses it, and that table stays
the source of truth: importing it imports torch, which a ``--help`` must not wait for.
``tests/unit/cli/test_choices.py`` pins every mirror to its source.
"""

#: :data:`torchref.maps.ded_weights.SCHEMES`.
DED_WEIGHT_SCHEMES = ("none", "inverse_variance", "q")
#: :data:`torchref.maps.ded_weights.DEFAULT_SCHEME`.
DEFAULT_DED_WEIGHT_SCHEME = "q"

#: :data:`torchref.scaling.scaler_base.SCALE_TARGETS`.
SCALE_TARGETS = ("nll", "ml_noalpha", "ls")
#: :data:`torchref.scaling.scaler_base.DEFAULT_SCALE_TARGET`.
DEFAULT_SCALE_TARGET = "ls"

#: ``(name, doc)`` of each row of
#: :data:`torchref.refinement.targets.xray._specs.XRAY_TARGETS`, in table order.
XRAY_TARGET_DOCS = (
    (
        "ml",
        "Read MLF, variance epsilon*beta, conditional mean alpha*|F_calc| (default).",
    ),
    ("ml_noalpha", "As 'ml' with the Luzzati mean coupling fixed at 1."),
    (
        "ml_full",
        "Full-form MLF: marginalises the unknown error-free amplitude, so the "
        "observation error enters as an amplitude-only Gaussian. ~4x the cost.",
    ),
    (
        "nll_beta",
        "Gaussian amplitude NLL on ml's model-error variance -- the large-signal "
        "limit of 'ml'. Diagnostic: isolates the variance model from the shape.",
    ),
    (
        "nll",
        "Gaussian amplitude NLL weighted by the experimental sigma only. No "
        "model-error term, so it does not control overfitting.",
    ),
    (
        "nll_i",
        "Gaussian NLL on the observed INTENSITIES weighted by sigma(I). As 'nll' "
        "but skips the French-Wilson conversion, which reshapes the weak tail.",
    ),
    ("ls", "Least squares with unit weights; the scaler owns the overall scale."),
    (
        "ls_wunit_k1",
        "Phenix-style least squares: unit weights and a single global scale "
        "recomputed every gradient call (bypasses the scaler).",
    ),
)
