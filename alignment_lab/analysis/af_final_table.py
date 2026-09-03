"""The AlphaFold placement benchmark, both resolution windows in one table.

Our placement is refined with the Figure-2 recipe plus the rigid-body step;
the Phaser-placed arm is refined identically and does not depend on our window.
"""
import json, re, glob, os, sys

RUNS = "alignment_lab/runs"
WORKLIST = "alignment_lab/analysis/af_pilot_worklist.txt"


def pilot_rows(pattern):
    out = {}
    for f in glob.glob(pattern):
        for line in open(f, errors="replace"):
            if not line.startswith("ROW "):
                continue
            body = line[4:]
            sg = re.search(r"sg='([^']*)'", body)
            body = re.sub(r"sg='[^']*'", "", body)
            d = dict(kv.split("=", 1) for kv in body.split() if "=" in kv)
            d["sg"] = sg.group(1) if sg else ""
            out[d["code"]] = d
    return out


def r3_rows(pattern):
    out = {}
    for f in glob.glob(pattern):
        for line in open(f, errors="replace"):
            if line.startswith("R3 "):
                d = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
                out[d["code"]] = d
    return out


def placement(code):
    """Per-component pose at 15-3 A, and how much of the model it covers."""
    f = f"{RUNS}/af_mr_dmin3.0/{code}/summary.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    poses, chains = d["pose_vs_phaser"], d["chains"]
    sizes = [c["n_res"] for c in chains]
    total = sum(sizes)
    ok = [p["rot_deg"] <= 5.0 and p["trans_A"] <= 2.0 for p in poses]
    placed_res = sum(s for s, k in zip(sizes, ok) if k)
    return {"all_ok": all(ok), "frac": placed_res / max(total, 1),
            "worst": max((p["rot_deg"] for p, k in zip(poses, ok) if not k), default=0.0)}


def main():
    codes = [l.split()[0] for l in open(WORKLIST) if l.strip()]
    p4 = pilot_rows(f"alignment_lab/slurm/afmr2_{sys.argv[1]}_*.out")
    r3 = r3_rows(f"alignment_lab/slurm/afr3_{sys.argv[2]}_*.out")
    print(f"{'code':6s} {'sg':13s} {'placed':>7s} {'R-free 4A':>10s} {'R-free 3A':>10s} {'Phaser':>8s} {'3A-Phaser':>10s}")
    tally = {"close": 0, "worse": 0, "bad": 0}
    for c in codes:
        a, b, pl = p4.get(c), r3.get(c), placement(c)
        if not a:
            continue
        ph = float(a["rfree_phaser"])
        o4 = float(a["rfree_ours"])
        o3 = float(b["rfree"]) if b and "rfree" in b else float("nan")
        cover = "all" if pl and pl["all_ok"] else (f"{pl['frac']*100:.0f}%" if pl else "?")
        diff = o3 - ph
        key = "close" if diff <= 0.010 else ("worse" if diff <= 0.030 else "bad")
        tally[key] += 1
        print(f"{c:6s} {a['sg']:13s} {cover:>7s} {o4:10.4f} {o3:10.4f} {ph:8.4f} {diff:+10.4f}")
    n = sum(tally.values())
    print(f"\nAt 15-3 A, {n} structures: {tally['close']} within 0.010 of the Phaser-placed "
          f"refinement, {tally['worse']} between 0.010 and 0.030, {tally['bad']} beyond 0.030")


if __name__ == "__main__":
    main()
