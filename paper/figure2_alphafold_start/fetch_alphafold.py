#!/usr/bin/env python3
"""Phase B — download AlphaFold DB models for the UniProt accessions found in
``manifest.json`` (produced by ``build_af_manifest.py``).

For each unique UniProt accession it asks the AlphaFold prediction API

    https://alphafold.ebi.ac.uk/api/prediction/{accession}

for the model record (which carries the correct ``pdbUrl`` for the latest model
version) and downloads the PDB into ``afdb/AF-{accession}.pdb``. Accessions with
no AlphaFold model (empty API response) are recorded explicitly — many benchmark
structures (engineered constructs, complexes, nucleic acids, designed proteins)
simply have no entry.

After downloading it rewrites ``manifest.json`` with two added flags:

* per-chain ``af_available`` (bool) — its UniProt model downloaded.
* per-structure ``af_complete`` (bool) — *every* protein chain has an AF model.
  These ``af_complete`` codes define the eventual AlphaFold-start benchmark arm.

A coverage report is written to ``af_coverage.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST_JSON = HERE / "manifest.json"
AFDB_DIR = HERE / "afdb"
COVERAGE_CSV = HERE / "af_coverage.csv"

API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"


def _get(url: str, timeout: int):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def fetch_one(acc: str, retries: int = 3, timeout: int = 60):
    """Download the latest AF model for ``acc``.

    Returns one of ``ok``/``missing``/``error``. Skips the download if the file
    already exists. Resolves the model URL via the prediction API so the current
    model version is used automatically; an empty API response means no model.
    """
    dest = AFDB_DIR / f"AF-{acc}.pdb"
    if dest.exists() and dest.stat().st_size > 0:
        return acc, "ok"

    last_err = None
    for attempt in range(retries):
        try:
            records = json.loads(_get(API_URL.format(acc=acc), timeout))
            if not records:
                return acc, "missing"
            pdb_url = records[0].get("pdbUrl")
            if not pdb_url:
                return acc, "missing"
            data = _get(pdb_url, timeout)
            tmp = dest.with_suffix(".pdb.tmp")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return acc, "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return acc, "missing"
            last_err = e
        except Exception as e:  # noqa: BLE001 - network flakiness, retry
            last_err = e
        time.sleep(1.0 + attempt)
    print(f"  error fetching {acc}: {last_err}", file=sys.stderr)
    return acc, "error"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=16, help="Concurrent downloads.")
    ap.add_argument(
        "--accessions",
        nargs="+",
        help="Fetch only these accessions (for testing); does not rewrite manifest.",
    )
    args = ap.parse_args()

    AFDB_DIR.mkdir(parents=True, exist_ok=True)

    if args.accessions:
        accessions = list(dict.fromkeys(args.accessions))
        records = None
    else:
        records = json.loads(MANIFEST_JSON.read_text())
        accessions = sorted(
            {a for r in records for a in r.get("uniprot_accessions", [])}
        )

    print(f"Fetching {len(accessions)} unique AlphaFold models...")
    status = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_one, a): a for a in accessions}
        for fut in as_completed(futures):
            acc, st = fut.result()
            status[acc] = st
            done += 1
            if done % 50 == 0 or done == len(accessions):
                n_ok = sum(1 for s in status.values() if s == "ok")
                print(f"  [{done:4d}/{len(accessions)}] {n_ok} downloaded")

    n_ok = sum(1 for s in status.values() if s == "ok")
    n_missing = sum(1 for s in status.values() if s == "missing")
    n_err = sum(1 for s in status.values() if s == "error")
    print("\n" + "=" * 60)
    print(f"Accessions requested:   {len(accessions)}")
    print(f"  downloaded (ok):      {n_ok}")
    print(f"  no AF model (404):    {n_missing}")
    print(f"  errors:               {n_err}")

    if records is None:
        return 0

    # Annotate the manifest and compute the af_complete arm.
    n_complete = 0
    with open(COVERAGE_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "n_protein_chains", "n_chains_with_af", "af_complete"])
        for r in records:
            chains = r["chains"]
            n_with_af = 0
            for ch in chains:
                avail = bool(ch.get("uniprot")) and status.get(ch["uniprot"]) == "ok"
                ch["af_available"] = avail
                if avail:
                    n_with_af += 1
            complete = bool(chains) and n_with_af == len(chains)
            r["af_complete"] = complete
            if complete:
                n_complete += 1
            w.writerow([r["code"], len(chains), n_with_af, complete])

    MANIFEST_JSON.write_text(json.dumps(records, indent=2))

    print(f"\nStructures with a complete AF model set (AF arm): {n_complete}/{len(records)}")
    print(f"Manifest updated: {MANIFEST_JSON}")
    print(f"Coverage report:  {COVERAGE_CSV}")
    print(f"AF models dir:    {AFDB_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
