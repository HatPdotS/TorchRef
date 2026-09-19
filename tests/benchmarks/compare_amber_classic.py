"""Compare classic and AMBER restraints on identical prepared benchmark starts.

Run with the project interpreter and optional OpenMM/PDBFixer dependencies.
The experiment keeps the work/free split, X-ray target, atom set, riding wrapper,
optimizer and ADP settings fixed. AMBER's weight is calibrated once from initial
heavy-coordinate gradient RMS after the riding Jacobian, without consulting R-free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from torchref import Model
from torchref.experimental.targets.amber_target import AmberTarget
from torchref.refinement.lbfgs_refinement import LBFGSRefinement

FILES = Path(__file__).resolve().parents[1] / "files"
IDENTITY = ["chainid", "resseq", "icode", "name"]


def _prepare(code: str, output: Path, seed: int) -> dict:
    """Complete heavy atoms once, preserve existing rows, and add H in TorchRef."""
    import openmm.app as app
    from pdbfixer import PDBFixer

    source = FILES / "pdb" / f"{code}_af.pdb"
    original = (
        Model(device="cpu", verbose=0, strip_H=True, add_hydrogens=False)
        .load_pdb(str(source))
        .strip_altlocs()
    )
    fixer = PDBFixer(filename=str(source))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=seed)
    fixed_path = output / f"{code}_heavy_completed.pdb"
    with fixed_path.open("w") as handle:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    fixed = Model(device="cpu", verbose=0, add_hydrogens=False).load_pdb(
        str(fixed_path)
    )
    original_rows = {
        tuple(row[k] for k in IDENTITY): row for _, row in original.pdb.iterrows()
    }
    frame = fixed.pdb.copy()
    frame.attrs = original.pdb.attrs.copy()
    added = []
    found = set()
    for i, row in frame.iterrows():
        key = tuple(row[k] for k in IDENTITY)
        if key in original_rows:
            found.add(key)
            for column in ["x", "y", "z", "tempfactor", "occupancy", "ATOM"]:
                frame.at[i, column] = original_rows[key][column]
        else:
            added.append(key)
            same_residue = original.pdb[
                (original.pdb.chainid == row.chainid)
                & (original.pdb.resseq == row.resseq)
                & (original.pdb.icode == row.icode)
            ]
            frame.at[i, "tempfactor"] = float(same_residue.tempfactor.mean())
            frame.at[i, "occupancy"] = 1.0
    assert found == set(original_rows), "Heavy-atom preparation dropped original atoms"
    model = original._new_model_from_df(frame, strip_H=False)
    torch.manual_seed(seed)
    model = model.hydrogenate()
    model.set_hydrogen_mode("riding")
    compatibility = AmberTarget(model=model)
    assert compatibility._n_omm_atoms == len(model.pdb)
    path = output / f"{code}_prepared.pdb"
    model.write_pdb(str(path))
    return {
        "code": code,
        "source": str(source),
        "prepared": str(path),
        "added_heavy_atoms": added,
        "original_heavy_atoms": len(original.pdb),
        "prepared_atoms": len(model.pdb),
        "hydrogens": int(model.pdb.element.str.strip().isin(["H", "D"]).sum()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _magnitude(values: torch.Tensor) -> dict:
    """Summarize atom-vector norms, or absolute scalar parameter gradients."""
    values = values.detach().cpu()
    norms = values.norm(dim=-1) if values.ndim == 2 else values.abs().flatten()
    if not norms.numel():
        return {"n": 0}
    return {
        "n": norms.numel(),
        "rms": float(norms.square().mean().sqrt()),
        "median": float(norms.median()),
        "p95": float(torch.quantile(norms, 0.95)),
        "max": float(norms.max()),
        "l2": float(values.norm()),
    }


def _classic_loss(ref: LBFGSRefinement) -> torch.Tensor:
    """Sum the active classic components, excluding the disabled Rama prior."""
    return sum(
        target()
        for name, target in ref.geometry_target.items()
        if name != "ramachandran"
    )


def _gradient_snapshot(ref: LBFGSRefinement, amber: AmberTarget) -> tuple[dict, dict]:
    """Record raw atomic and model-parameter gradients for both priors and X-ray."""
    # The classic prior is inactive during AMBER refinement, so its contact list
    # needs maintenance before evaluating it on the final AMBER coordinates.
    for target in ref.geometry_target.values():
        target.maintenance()
    model = ref.model
    heavy = torch.as_tensor(
        ~model.pdb.element.str.strip().isin(["H", "D"]).to_numpy(), device=model.device
    )
    tensors = {}
    result = {}
    for name, function in [
        ("classic_raw", lambda: _classic_loss(ref)),
        ("amber_raw", amber.forward),
        ("xray", ref.xray_target_work.forward),
    ]:
        model.reset_cache()
        xyz = model.xyz()
        leaves = model.xyz.optimization_parameters()
        value = function()
        gradients = torch.autograd.grad(value, [xyz] + leaves, allow_unused=True)
        atomic = gradients[0]
        assert atomic is not None and torch.isfinite(atomic).all(), name
        tensors[name] = atomic.detach()
        result[name] = {
            "loss": float(value.detach()),
            "heavy": _magnitude(atomic[heavy]),
            "hydrogen": _magnitude(atomic[~heavy]),
            "all": _magnitude(atomic),
            "parameter_gradients": [
                _magnitude(g) if g is not None else {"n": 0} for g in gradients[1:]
            ],
        }
        if name == "amber_raw":
            import openmm.unit as unit

            forces = np.asarray(
                amber._context.getState(getForces=True)
                .getForces(asNumpy=True)
                .value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
            )
            result[name]["clipped_atom_fraction"] = float(
                (np.linalg.norm(forces, axis=1) > 10000).mean()
            )
            unclipped = torch.as_tensor(
                -forces[amber._model_to_omm] * 0.1 / len(xyz),
                dtype=xyz.dtype,
                device=xyz.device,
            )
            result["amber_unclipped"] = {
                "heavy": _magnitude(unclipped[heavy]),
                "hydrogen": _magnitude(unclipped[~heavy]),
                "all": _magnitude(unclipped),
            }
    for name in ["classic_raw", "amber_raw"]:
        g = tensors[name][heavy].flatten()
        x = tensors["xray"][heavy].flatten()
        result[name]["heavy_cosine_to_xray"] = float(
            torch.nn.functional.cosine_similarity(g, x, dim=0)
        )
    c, a = (
        tensors["classic_raw"][heavy].flatten(),
        tensors["amber_raw"][heavy].flatten(),
    )
    result["classic_amber_heavy_cosine"] = float(
        torch.nn.functional.cosine_similarity(c, a, dim=0)
    )
    return result, tensors


def _metrics(ref: LBFGSRefinement) -> dict:
    """Report R factors and dictionary geometry using heavy atoms alone."""
    with torch.no_grad():
        rwork, rfree = ref.get_rfactor()
        model = ref.model
        restraints = model.restraints
        restraints.cat_dict()
        heavy = torch.as_tensor(
            ~model.pdb.element.str.strip().isin(["H", "D"]).to_numpy(),
            device=model.device,
        )
        result = {"rwork": float(rwork), "rfree": float(rfree)}
        for name, method in [
            ("bond", restraints.bond_deviations),
            ("angle", restraints.angle_deviations),
        ]:
            deviations, sigmas = method()
            indices = restraints.restraints[name]["all"]["indices"]
            keep = heavy[indices].all(dim=-1)
            d, z = deviations[keep], deviations[keep] / sigmas[keep]
            if name == "angle":
                d = torch.rad2deg(d)
            result[f"heavy_{name}_rms_delta"] = float(d.square().mean().sqrt())
            result[f"heavy_{name}_rms_z"] = float(z.square().mean().sqrt())
        return result


def _run(
    code: str,
    prepared: Path,
    arm: str,
    output: Path,
    cycles: int,
    seed: int,
    weight: float | None,
) -> dict:
    """Run classic alternating scaler/XYZ/ADP refinement with one active prior."""
    torch.manual_seed(seed)
    started = time.perf_counter()
    ref = LBFGSRefinement(
        pdb=str(prepared),
        data_file=str(FILES / "mtz" / f"{code}.mtz"),
        device=torch.device("cpu"),
        verbose=0,
        add_hydrogens=False,
        hydrogens_in_xray=True,
        target_mode="ml",
    )
    ref.set_hydrogen_mode("riding")
    amber = AmberTarget(model=ref.model, normalize_by_atoms=True, verbose=0)
    state = ref.loss_state
    initial_gradients, raw = _gradient_snapshot(ref, amber)
    calibration = (
        0.2
        * initial_gradients["classic_raw"]["parameter_gradients"][0]["rms"]
        / initial_gradients["amber_raw"]["parameter_gradients"][0]["rms"]
    )
    if weight is None:
        weight = calibration
    if arm == "amber":
        state.targets = {
            key: target
            for key, target in state.targets.items()
            if not key.startswith("geometry/")
        }
        state.clear()
        state.register_target("amber", amber)
        state.set_weight("amber", weight)
        state.refresh_loss_leaves()
    active_names = [
        key for key in state.targets if state.get_effective_weight(key) != 0
    ]
    if arm == "amber":
        assert "amber" in active_names and not any(
            key.startswith("geometry/") for key in active_names
        )
    else:
        assert "amber" not in active_names
    initial = _metrics(ref)
    xyz_hash = hashlib.sha256(
        ref.model.xyz().detach().cpu().numpy().tobytes()
    ).hexdigest()
    result = {
        "code": code,
        "arm": arm,
        "cycles": cycles,
        "seed": seed,
        "amber_weight": weight,
        "calibration_weight": calibration,
        "calibration_basis": "heavy_xyz_parameter_gradient_rms_after_riding",
        "cartesian_calibration_weight": (
            0.2
            * initial_gradients["classic_raw"]["heavy"]["rms"]
            / initial_gradients["amber_raw"]["heavy"]["rms"]
        ),
        "classic_group_weight": 0.2,
        "initial": initial,
        "initial_gradients": initial_gradients,
        "initial_xyz_sha256": xyz_hash,
        "active_targets": active_names,
        "n_atoms": len(ref.model.pdb),
        "n_reflections": len(ref.reflection_data.hkl),
        "reflection_split_sha256": hashlib.sha256(
            ref.reflection_data.hkl.detach().cpu().numpy().tobytes()
            + ref.reflection_data.rfree_flags.detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "parameter_gradient_order": [
            "heavy_xyz_per_angstrom",
            "torsion_per_radian",
            "water_rotation_per_radian",
        ],
        "geometry_units": {"bond_rms_delta": "angstrom", "angle_rms_delta": "degree"},
        "setup_seconds": time.perf_counter() - started,
        "trajectory": [],
    }
    target_path = output / f"{code}_{arm}.json"
    target_path.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "event": "initial",
                "code": code,
                "arm": arm,
                "metrics": initial,
                "weight": weight,
                "grad_rms_classic": initial_gradients["classic_raw"]["heavy"]["rms"],
                "grad_rms_amber": initial_gradients["amber_raw"]["heavy"]["rms"],
            }
        ),
        flush=True,
    )
    for cycle in range(1, cycles + 1):
        start_cycle = time.perf_counter()
        ref.refine(macro_cycles=1)
        entry = {
            "cycle": cycle,
            **_metrics(ref),
            "seconds": time.perf_counter() - start_cycle,
        }
        result["trajectory"].append(entry)
        target_path.write_text(json.dumps(result, indent=2))
        print(
            json.dumps({"event": "cycle", "code": code, "arm": arm, **entry}),
            flush=True,
        )
    result["final"] = _metrics(ref)
    result["final_gradients"], _ = _gradient_snapshot(ref, amber)
    result["refinement_seconds"] = sum(x["seconds"] for x in result["trajectory"])
    result["total_seconds"] = time.perf_counter() - started
    ref.model.write_pdb(str(output / f"{code}_{arm}_refined.pdb"))
    target_path.write_text(json.dumps(result, indent=2))
    return result


def _main() -> None:
    """Prepare one benchmark or run one arm in its own process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "classic", "amber"])
    parser.add_argument("--code", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--amber-weight", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        details = _prepare(args.code, args.output, args.seed)
        (args.output / f"{args.code}_preparation.json").write_text(
            json.dumps(details, indent=2)
        )
        print(json.dumps(details), flush=True)
    else:
        weight = args.amber_weight
        if args.action == "amber" and weight is None:
            baseline = json.loads(
                (args.output / f"{args.code}_classic.json").read_text()
            )
            gradients = baseline["initial_gradients"]
            weight = (
                0.2
                * gradients["classic_raw"]["parameter_gradients"][0]["rms"]
                / gradients["amber_raw"]["parameter_gradients"][0]["rms"]
            )
        _run(
            args.code,
            args.output / f"{args.code}_prepared.pdb",
            args.action,
            args.output,
            args.cycles,
            args.seed,
            weight,
        )


if __name__ == "__main__":
    _main()
