"""Shared CLI utilities for torchref command-line scripts.

This module provides reusable argument-group builders, device setup,
file validation, format-aware loaders, and other helpers that are
shared across multiple CLI entry points.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Unbuffered output
# ---------------------------------------------------------------------------

def configure_unbuffered_output():
    """Force line-buffered stdout/stderr so progress prints appear immediately."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    os.environ["PYTHONUNBUFFERED"] = "1"


# ---------------------------------------------------------------------------
# Argparse argument-group builders
# ---------------------------------------------------------------------------

def add_device_arg(parser: argparse.ArgumentParser) -> None:
    """Add ``--device`` argument.

    The default ``auto`` defers to :data:`torchref.config.device`, which
    only selects CUDA when a visible GPU meets this PyTorch build's
    compute-capability requirement *and* has at least ~10 GB VRAM;
    otherwise it falls back to MPS or CPU. Explicit ``cuda`` / ``cpu``
    bypass the auto-selection gates and are pushed back into the global
    config so the rest of TorchRef picks them up.
    """
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help=(
            "Computation device (default: auto; uses CUDA only when a "
            "visible GPU passes the capability + VRAM checks in "
            "torchref.config, else MPS/CPU)"
        ),
    )


def add_verbose_arg(parser: argparse.ArgumentParser) -> None:
    """Add ``-v`` / ``--verbose`` argument."""
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level: 0=quiet, 1=normal, 2=detailed (default: 1)",
    )


def add_outdir_arg(
    parser: argparse.ArgumentParser,
    required: bool = True,
    default: Optional[str] = None,
    help: str = "Output directory for refined structure and results",
) -> None:
    """Add ``-o`` / ``--outdir`` argument."""
    parser.add_argument(
        "-o",
        "--outdir",
        required=required,
        type=str,
        default=default,
        help=help,
    )


def add_output_arg(
    parser: argparse.ArgumentParser,
    required: bool = True,
    help: str = "Output file path",
) -> None:
    """Add ``-o`` / ``--output`` argument (for single-file output scripts)."""
    parser.add_argument(
        "-o",
        "--output",
        required=required,
        type=str,
        help=help,
    )


def add_dmin_arg(
    parser: argparse.ArgumentParser,
    help: str = "High-resolution cutoff in Angstroms (default: from data)",
) -> None:
    """Add ``--dmin`` argument."""
    parser.add_argument(
        "--dmin",
        type=float,
        default=None,
        help=help,
    )


def add_dmax_arg(
    parser: argparse.ArgumentParser,
    help: str = "Low-resolution cutoff in Angstroms (default: from data)",
) -> None:
    """Add ``--dmax`` argument."""
    parser.add_argument(
        "--dmax",
        type=float,
        default=None,
        help=help,
    )


def add_resolution_args(parser: argparse.ArgumentParser) -> None:
    """Add both ``--dmin`` and ``--dmax`` arguments."""
    add_dmin_arg(parser)
    add_dmax_arg(parser)


def add_adp_mode_arg(parser: argparse.ArgumentParser) -> None:
    """Add ``--adp-mode`` and ``--anisotropic-selection`` arguments.

    Controls the atomic displacement parameter (ADP) parametrization: isotropic
    per-atom B (default) vs anisotropic 6-component U for a selected atom set.
    """
    parser.add_argument(
        "--adp-mode",
        type=str,
        default="isotropic",
        choices=["isotropic", "anisotropic"],
        help="ADP parametrization: 'isotropic' (default) refines a per-atom "
        "B-factor; 'anisotropic' refines a 6-component U tensor for the atoms "
        "given by --anisotropic-selection. The model is converted between "
        "representations and the output PDB/mmCIF follows the convention "
        "(ANISOU only for anisotropic atoms).",
    )
    parser.add_argument(
        "--anisotropic-selection",
        type=str,
        default=None,
        metavar="SELECTION",
        help="Phenix-style atom selection refined anisotropically when "
        "--adp-mode anisotropic (e.g. 'chain A', 'not resname HOH'). Default: "
        "'not resname HOH and not element H' (all non-water heavy atoms).",
    )


def add_wavelength_arg(parser: argparse.ArgumentParser) -> None:
    """Add ``--wavelength`` argument (Angstroms; 0 disables anomalous)."""
    parser.add_argument(
        "--wavelength",
        type=float,
        default=1.0,
        help="X-ray wavelength in Angstroms, used for anomalous (f'/f'') "
        "scattering. Set to 0 to disable anomalous refinement entirely, which "
        "also forces a Friedel-merged read of the data (no F(+)/F(-) Bijvoet "
        "pairs). Default 1.0.",
    )


def add_column_args(parser: argparse.ArgumentParser) -> None:
    """Add ``-csf`` and ``-csig`` column-selection arguments."""
    parser.add_argument(
        "-csf",
        "--column-structure-factor",
        type=str,
        default=None,
        metavar="COL",
        help="Column name for observed amplitudes or intensities in the "
             "structure factor file (default: auto-detect)",
    )
    parser.add_argument(
        "-csig",
        "--column-sigma",
        type=str,
        default=None,
        metavar="COL",
        help="Column name for amplitude/intensity sigmas in the structure "
             "factor file (default: auto-detect)",
    )


def add_dual_column_args(parser: argparse.ArgumentParser) -> None:
    """Add dark/light-specific column-selection arguments.

    Adds ``-csf-dark``, ``-csf-light``, ``-csig-dark``, ``-csig-light``,
    ``-cphi-dark``, ``-cphi-light`` for per-dataset column overrides.
    """
    for side in ("dark", "light"):
        parser.add_argument(
            f"-csf-{side}",
            f"--column-structure-factor-{side}",
            type=str,
            default=None,
            metavar="COL",
            help=f"Column name for amplitudes/intensities in the {side} "
                 f"structure factor file (default: auto-detect)",
        )
        parser.add_argument(
            f"-csig-{side}",
            f"--column-sigma-{side}",
            type=str,
            default=None,
            metavar="COL",
            help=f"Column name for sigmas in the {side} structure factor "
                 f"file (default: auto-detect)",
        )
        parser.add_argument(
            f"-cphi-{side}",
            f"--column-phase-{side}",
            type=str,
            default=None,
            metavar="COL",
            help=f"Column name for phases in degrees in the {side} "
                 f"structure factor file (default: auto-detect)",
        )


def add_cif_arg(
    parser: argparse.ArgumentParser,
    nargs: str = "+",
    help: str = "CIF restraint dictionary files for non-standard ligands",
) -> None:
    """Add ``--cif`` argument for restraints."""
    parser.add_argument(
        "--cif",
        type=str,
        nargs=nargs,
        default=None,
        help=help,
    )


def add_single_model_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard single-model input arguments.

    Creates an *Input files* argument group with ``-m``/``--model``,
    ``-sf``/``--structure-factor``, ``--cif`` and a *Column selection*
    group with ``-csf``/``--column-structure-factor``,
    ``-csig``/``--column-sigma``.
    """
    inp = parser.add_argument_group("Input files")
    inp.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        help="Input model file (PDB or CIF format)",
    )
    inp.add_argument(
        "-sf",
        "--structure-factor",
        required=True,
        type=str,
        help="Input structure factor file (MTZ or CIF format)",
    )
    add_cif_arg(inp, nargs=None,
                help="CIF restraints dictionary (auto-detected if not provided)")

    col = parser.add_argument_group("Column selection")
    add_column_args(col)


def add_dual_model_args(
    parser: argparse.ArgumentParser,
    fraction_required: bool = True,
    fraction_default: Optional[float] = None,
) -> None:
    """Add the standard dual-model (dark/light) input arguments.

    Creates an *Input files* argument group with ``-dm``/``--dark-model``,
    ``-lm``/``--light-model``, ``-dsf``/``--dark-structure-factor``,
    ``-lsf``/``--light-structure-factor``, ``--fraction``, ``--cif``
    and a *Column selection* group with per-side column flags.
    """
    inp = parser.add_argument_group("Input files")
    inp.add_argument(
        "-dm",
        "--dark-model",
        required=True,
        type=str,
        help="Dark / reference state model file (PDB or CIF)",
    )
    inp.add_argument(
        "-lm",
        "--light-model",
        required=True,
        type=str,
        help="Light / triggered state model file (PDB or CIF)",
    )
    inp.add_argument(
        "-dsf",
        "--dark-structure-factor",
        required=True,
        type=str,
        help="Dark / reference state structure factor file (MTZ or CIF)",
    )
    inp.add_argument(
        "-lsf",
        "--light-structure-factor",
        required=True,
        type=str,
        help="Light / triggered state structure factor file (MTZ or CIF)",
    )
    frac_kwargs = {
        "type": float,
        "help": "Occupancy fraction of the light/excited state "
                "(e.g. 0.37). Dark fraction is computed as 1 - fraction.",
    }
    if fraction_required:
        frac_kwargs["required"] = True
    else:
        frac_kwargs["default"] = fraction_default
    inp.add_argument("--fraction", **frac_kwargs)
    add_cif_arg(inp)

    col = parser.add_argument_group("Column selection")
    add_dual_column_args(col)


def add_output_format_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--output-format`` argument for coordinate file format."""
    parser.add_argument(
        "--output-format",
        type=str,
        default="both",
        choices=["pdb", "cif", "both"],
        help="Output coordinate file format (default: both PDB and mmCIF)",
    )


def add_metadata_args(parser: argparse.ArgumentParser) -> None:
    """Add metadata-related CLI arguments for deposition headers."""
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title for the output file header",
    )
    parser.add_argument(
        "--authors",
        type=str,
        nargs="+",
        default=None,
        help="Author names for the output file header",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        default=False,
        help="Suppress metadata headers in output files",
    )


def add_general_args(parser: argparse.ArgumentParser) -> None:
    """Add a *General* argument group with ``--device`` and ``-v``/``--verbose``."""
    gen = parser.add_argument_group("General")
    add_device_arg(gen)
    add_verbose_arg(gen)


def add_scaler_mode_arg(parser: argparse.ArgumentParser) -> None:
    """Add ``--scaler-mode`` argument."""
    parser.add_argument(
        "--scaler-mode",
        type=str,
        default="shared",
        choices=["shared", "split"],
        help="Scaler mode: 'shared' uses a single scaler (dark) for all "
             "targets ensuring bulk-solvent cancellation. 'split' uses "
             "separate scalers for dark and light/mixed sides "
             "(default: shared)",
    )


def add_n_cycles_arg(
    parser: argparse.ArgumentParser, default: int = 5
) -> None:
    """Add ``-n`` / ``--n-cycles`` argument."""
    parser.add_argument(
        "-n",
        "--n-cycles",
        type=int,
        default=default,
        help=f"Number of refinement macro cycles (default: {default})",
    )


def add_weights_arg(
    parser: argparse.ArgumentParser,
    default_weights: Optional[dict] = None,
) -> None:
    """Add ``--weights`` argument (JSON string or file path)."""
    help_text = (
        "Target weights as a JSON string or path to a JSON file. "
        "Only the keys you supply override defaults."
    )
    if default_weights is not None:
        help_text += f"  Defaults: {json.dumps(default_weights, indent=None)}"
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help=help_text,
    )


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def parse_device_str(device_str: str) -> "torch.device":
    """Parse the ``--device`` CLI argument into a :class:`torch.device`.

    ``"auto"`` returns the package default resolved at import time by
    :data:`torchref.config.device` (cuda -> mps -> cpu, with capability and
    VRAM checks). Explicit values are pushed back into the global config so
    every TorchRef component picks up the user's CLI choice; if the request
    cannot be satisfied (e.g. ``cuda`` on a CPU-only host) we warn and fall
    back to CPU rather than crashing the run.
    """
    from torchref.config import device as device_config, get_default_device

    if device_str == "auto":
        return get_default_device()

    try:
        device_config.current = device_str
    except (RuntimeError, ValueError) as exc:
        print(
            f"Warning: {exc} Falling back to CPU.",
            file=sys.stderr,
        )
        device_config.current = "cpu"
    return device_config.current


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------

def validate_files(
    path_label_pairs: List[Tuple[str, str]],
    exit_on_error: bool = False,
) -> int:
    """Check that all specified files exist.

    Parameters
    ----------
    path_label_pairs : list of (path, label)
        Each item is ``(file_path, human_label)`` to check.
    exit_on_error : bool
        If *True*, call ``sys.exit(1)`` on the first missing file.
        Otherwise return 1.

    Returns
    -------
    int
        0 if all files exist, 1 on the first missing file.
    """
    for path, label in path_label_pairs:
        if not Path(path).exists():
            print(f"Error: {label} file not found: {path}", file=sys.stderr)
            if exit_on_error:
                sys.exit(1)
            return 1
    return 0


def validate_cif_files(cif_paths: Optional[List[str]]) -> int:
    """Check that all CIF restraint files exist.

    Returns 0 if all exist (or *cif_paths* is ``None``), 1 otherwise.
    """
    if not cif_paths:
        return 0
    for cif_path in cif_paths:
        if not Path(cif_path).exists():
            print(
                f"Error: --cif file not found: {cif_path}",
                file=sys.stderr,
            )
            return 1
    return 0


# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------

def build_column_names(
    column_structure_factor: Optional[str] = None,
    column_sigma: Optional[str] = None,
    column_phase: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Build a ``column_names`` dict from ``-csf``, ``-csig``, ``-cphi`` args.

    Returns ``None`` when none are specified (auto-detect).
    The dict keys (``"F"``, ``"SIGF"``, ``"PHIF"``) match the keys
    expected by :meth:`ReflectionData.load_mtz`.
    """
    if column_structure_factor is None and column_sigma is None and column_phase is None:
        return None
    column_names: Dict[str, str] = {}
    if column_structure_factor is not None:
        column_names["F"] = column_structure_factor
    if column_sigma is not None:
        column_names["SIGF"] = column_sigma
    if column_phase is not None:
        column_names["PHIF"] = column_phase
    return column_names


def build_dual_column_names(
    args,
) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, str]]]:
    """Build column_names dicts for dark and light datasets.

    Reads the per-side flags (``-csf-dark``, ``-csf-light``, etc.)
    from the parsed *args* namespace.

    Returns
    -------
    column_names_dark, column_names_light : dict or None
        Column name overrides for each dataset, or ``None`` for
        auto-detection.
    """
    col_dark = build_column_names(
        args.column_structure_factor_dark,
        args.column_sigma_dark,
        args.column_phase_dark,
    )
    col_light = build_column_names(
        args.column_structure_factor_light,
        args.column_sigma_light,
        args.column_phase_light,
    )
    return col_dark, col_light


# ---------------------------------------------------------------------------
# Weights parsing
# ---------------------------------------------------------------------------

def parse_weights(
    weights_arg: Optional[str],
    defaults: Optional[dict] = None,
) -> Tuple[dict, Optional[str]]:
    """Parse the ``--weights`` argument (JSON string or file path).

    Parameters
    ----------
    weights_arg : str or None
        The raw ``args.weights`` value.
    defaults : dict, optional
        Base weights to merge user overrides into.  A *copy* is made.

    Returns
    -------
    weights : dict
        The resulting weights dictionary.
    error : str or None
        An error message if parsing failed, otherwise ``None``.
    """
    weights = dict(defaults) if defaults is not None else {}
    if weights_arg is None:
        return weights, None

    try:
        if Path(weights_arg).is_file():
            with open(weights_arg) as f:
                user_weights = json.load(f)
        else:
            user_weights = json.loads(weights_arg)
        if not isinstance(user_weights, dict):
            return weights, "--weights must be a JSON dictionary"
        weights.update(user_weights)
        return weights, None
    except (json.JSONDecodeError, ValueError) as e:
        return weights, f"Invalid JSON for --weights: {e}"


# ---------------------------------------------------------------------------
# Format-aware loaders
# ---------------------------------------------------------------------------

def load_model(
    path: str,
    max_res: Optional[float] = None,
    device: Union[str, "torch.device", None] = None,
    verbose: int = 0,
    cif: Optional[Union[str, List[str]]] = None,
) -> "ModelFT":
    """Load a model from PDB or CIF, auto-detected by file extension.

    Parameters
    ----------
    path : str
        Path to a ``.pdb`` or ``.cif`` / ``.mmcif`` file.
    max_res : float, optional
        Resolution limit for FFT grid setup.
    device : str or torch.device
        Target device.
    verbose : int
        Verbosity passed to ModelFT.
    cif : str or list of str, optional
        CIF restraint file(s) to load after the model.

    Returns
    -------
    ModelFT
    """
    from torchref import ModelFT
    from torchref.config import get_default_device

    if device is None:
        device = get_default_device()
    model = ModelFT(max_res=max_res, device=device, verbose=verbose)
    suffix = Path(path).suffix.lower()
    if suffix in (".cif", ".mmcif"):
        model.load_cif(path)
    else:
        model.load_pdb(path)

    if cif is not None:
        model.set_restraints_cif(cif)

    return model


def load_reflection_data(
    path: str,
    device: Union[str, "torch.device", None] = None,
    column_names: Optional[Dict[str, str]] = None,
    verbose: int = 0,
) -> "ReflectionData":
    """Load reflection data from MTZ or CIF, auto-detected by extension.

    Parameters
    ----------
    path : str
        Path to a ``.mtz`` or ``.cif`` file.
    device : str or torch.device
        Target device.
    column_names : dict, optional
        Column name overrides (``"F"``, ``"SIGF"``).  Only used for MTZ.
    verbose : int
        Verbosity passed to ReflectionData.

    Returns
    -------
    ReflectionData
    """
    from torchref import ReflectionData
    from torchref.config import get_default_device

    if device is None:
        device = get_default_device()
    data = ReflectionData(device=str(device), verbose=verbose)
    suffix = Path(path).suffix.lower()
    if suffix in (".cif",):
        data.load_cif(path)
    else:
        data.load_mtz(path, column_names=column_names)
    return data


# ---------------------------------------------------------------------------
# Timing registration
# ---------------------------------------------------------------------------

def register_timing():
    """Register torchref timing hooks (call after parse_args)."""
    from torchref.utils.timing import register_timing as _register
    _register()


# ---------------------------------------------------------------------------
# Metadata + output format helpers
# ---------------------------------------------------------------------------


def write_refinement_outputs(
    refinement,
    outdir: Path,
    args,
    verbose: int = 1,
) -> dict:
    """Write refined structure in the requested format(s) with metadata.

    Parameters
    ----------
    refinement : Refinement
        Completed refinement object.
    outdir : Path
        Output directory.
    args : argparse.Namespace
        Parsed CLI arguments (expects ``output_format``, ``title``,
        ``authors``, ``no_header``).
    verbose : int
        Verbosity level.

    Returns
    -------
    dict
        Dictionary with keys ``"pdb"``, ``"cif"`` mapping to output paths
        (or None if not written).
    """
    from torchref.io.metadata import RefinementMetadata

    output_format = getattr(args, "output_format", "both")
    no_header = getattr(args, "no_header", False)

    # Collect metadata
    metadata = None
    if not no_header:
        metadata = refinement.collect_deposition_metadata()
        # Apply CLI overrides
        if getattr(args, "title", None):
            metadata.title = args.title
        if getattr(args, "authors", None):
            metadata.authors = args.authors

    outputs = {"pdb": None, "cif": None}

    if output_format in ("pdb", "both"):
        output_pdb = outdir / "refined.pdb"
        refinement.write_out_pdb(str(output_pdb), metadata=metadata)
        outputs["pdb"] = output_pdb
        if verbose > 0:
            print(f"  Refined structure (PDB): {output_pdb}")
            sys.stdout.flush()

    if output_format in ("cif", "both"):
        output_cif = outdir / "refined.cif"
        refinement.write_out_cif(str(output_cif), metadata=metadata)
        outputs["cif"] = output_cif
        if verbose > 0:
            print(f"  Refined structure (mmCIF): {output_cif}")
            sys.stdout.flush()

    return outputs
