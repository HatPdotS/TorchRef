"""Collect the AlphaFold placement benchmark into one table.

Reads the ROW lines of a pilot array's SLURM logs: pose against the Phaser
placement, and R-free of our placement and Phaser's after the same refinement.
"""
import re, sys, glob

ROW = re.compile(r"^ROW (.*)$")


def rows(pattern):
    out = {}
    for f in glob.glob(pattern):
        for line in open(f, errors="replace"):
            m = ROW.match(line.strip())
            if not m:
                continue
            body = m.group(1)
            # sg is quoted and contains spaces; lift it out before splitting
            sg = re.search(r"sg='([^']*)'", body)
            body = re.sub(r"sg='[^']*'", "", body)
            d = dict(kv.split("=", 1) for kv in body.split() if "=" in kv)
            d["sg"] = sg.group(1) if sg else ""
            out[d["code"]] = d
    return out


def main():
    r = rows(sys.argv[1])
    print(f"{'code':6s} {'sg':13s} {'rot':>6s} {'trans':>6s} {'ours':>7s} {'phaser':>7s} {'diff':>8s}  verdict")
    near = worse = bad = 0
    for c in sorted(r):
        d = r[c]
        rot, tr = float(d["pose_rot"]), float(d["pose_trans"])
        o, p = float(d["rfree_ours"]), float(d["rfree_phaser"])
        placed_ok = rot <= 5.0 and tr <= 2.0
        diff = o - p
        if diff <= 0.010:
            v, near = "close", near + 1
        elif diff <= 0.030:
            v, worse = "worse", worse + 1
        else:
            v, bad = "MUCH WORSE", bad + 1
        if not placed_ok:
            v += " (misplaced)"
        print(f"{c:6s} {d.get('sg',''):13s} {rot:6.1f} {tr:6.2f} {o:7.4f} {p:7.4f} {diff:+8.4f}  {v}")
    n = len(r)
    print(f"\n{n} structures: {near} within 0.010 of Phaser, {worse} between 0.010 and 0.030, {bad} worse than 0.030")


if __name__ == "__main__":
    main()
