#!/usr/bin/env python3
"""Phase A — build a per-structure sequence + UniProt manifest for the
AlphaFold-start arm of the Figure 2 validation benchmark.

For every PDB code in ``figure2_validation/structures.json`` this script:

1. Extracts per-chain protein sequences *locally* from the deposited model
   (``data/{code}/{code}.pdb``, falling back to ``.cif``) via
   :pyattr:`torchref.model.model.Model.chain_sequences`.
2. Queries the RCSB Data API to map each polymer entity to a UniProt accession
   (the key that locates the AlphaFold DB model) and to its polymer type.
3. Joins the local chains to the RCSB entities via ``auth_asym_ids``.

Outputs (written next to this script):

* ``manifest.json``         — full per-structure records.
* ``manifest_summary.csv``  — one row per code for a quick coverage view.
* ``fasta/{code}.fasta``    — one record per protein chain.

This step is read-only against the benchmark; nothing under
``figure2_validation/`` or ``data/`` is modified.
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

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent                       # figure2_alphafold_start/
PAPER_ROOT = HERE.parent                                     # paper/
STRUCTURES_FILE = PAPER_ROOT / "figure2_validation" / "structures.json"
DATA = PAPER_ROOT / "data"                                   # symlink → scientific_testing/data

FASTA_DIR = HERE / "fasta"
MANIFEST_JSON = HERE / "manifest.json"
SUMMARY_CSV = HERE / "manifest_summary.csv"

# RCSB Data API — GraphQL batch endpoint (many entries per request).
GRAPHQL_URL = "https://data.rcsb.org/graphql"
GRAPHQL_CHUNK = 50  # entry IDs per GraphQL request

# A protein chain must have at least this many resolved standard residues to be
# considered (filters ligand-only / peptide-tag chains). Chains that are all
# unknown ('?'/'X') are dropped regardless.
MIN_PROTEIN_LEN = 20


# ──────────────────────────────────────────────────────────────────────────────
# Local sequence extraction
# ──────────────────────────────────────────────────────────────────────────────

def _local_sequences(code: str):
    """Return ``{chain_id: sequence}`` for the deposited model, or ``{}``.

    Uses the deposited ``.pdb`` (preferred) or ``.cif``. Chains that are empty,
    all-unknown, or shorter than ``MIN_PROTEIN_LEN`` resolved residues are
    dropped here so the manifest only carries plausible protein chains.
    """
    from torchref.model.model import Model

    pdb_path = DATA / code / f"{code}.pdb"
    cif_path = DATA / code / f"{code}.cif"

    model = Model()
    if pdb_path.exists():
        model.load_pdb(str(pdb_path))
    elif cif_path.exists():
        model.load_cif(str(cif_path))
    else:
        return {}

    out = {}
    for chain_id, seq in model.chain_sequences:
        resolved = len(seq.replace("?", "").replace("X", ""))
        if resolved >= MIN_PROTEIN_LEN:
            out[str(chain_id)] = seq
    return out


# ──────────────────────────────────────────────────────────────────────────────
# RCSB UniProt mapping
# ──────────────────────────────────────────────────────────────────────────────

_GRAPHQL_TEMPLATE = """{{
  entries(entry_ids: [{ids}]) {{
    rcsb_id
    polymer_entities {{
      rcsb_polymer_entity_container_identifiers {{
        auth_asym_ids
        reference_sequence_identifiers {{ database_name database_accession }}
      }}
      entity_poly {{ rcsb_entity_polymer_type pdbx_seq_one_letter_code_can }}
    }}
  }}
}}"""


def _graphql_entries(codes, retries: int = 3, timeout: int = 60):
    """POST one GraphQL query for ``codes`` and return its ``entries`` list."""
    ids = ",".join(f'"{c}"' for c in codes)
    query = _GRAPHQL_TEMPLATE.format(ids=ids)
    body = json.dumps({"query": query}).encode()
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                GRAPHQL_URL, data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.load(resp)
            return payload.get("data", {}).get("entries") or []
        except Exception as e:  # noqa: BLE001 - network flakiness, retry
            last_err = e
        time.sleep(1.0 + attempt)
    raise RuntimeError(f"GraphQL query failed for {codes[:3]}...: {last_err}")


def _chain_maps_for_chunk(codes):
    """Return ``{code: {auth_asym_id: {uniprot, polymer_type}}}`` for ``codes``.

    Entries absent from the RCSB response (obsolete/invalid) are omitted.
    """
    out = {}
    for entry in _graphql_entries(codes):
        chain_map = {}
        for pe in entry.get("polymer_entities") or []:
            ci = pe.get("rcsb_polymer_entity_container_identifiers", {}) or {}
            ep = pe.get("entity_poly") or {}
            polymer_type = ep.get("rcsb_entity_polymer_type")
            rcsb_seq = (ep.get("pdbx_seq_one_letter_code_can") or "").replace("\n", "")
            uniprot = None
            for ref in ci.get("reference_sequence_identifiers") or []:
                if ref.get("database_name") == "UniProt":
                    uniprot = ref.get("database_accession")
                    break
            for ch in ci.get("auth_asym_ids") or []:
                chain_map[str(ch)] = {
                    "uniprot": uniprot,
                    "polymer_type": polymer_type,
                    "rcsb_seq": rcsb_seq,
                }
        out[entry["rcsb_id"]] = chain_map
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Record assembly (join local sequences with the RCSB chain map)
# ──────────────────────────────────────────────────────────────────────────────

def build_record(code: str, seqs: dict, chain_map, mapped: bool):
    """Assemble the manifest record for one code from already-fetched data.

    ``seqs`` is the local ``{chain_id: sequence}`` map; ``chain_map`` is the RCSB
    ``{auth_asym_id: {uniprot, polymer_type, rcsb_seq}}`` map; ``mapped`` is
    whether the code was present in the RCSB GraphQL response.

    The protein chain set is taken from RCSB (authoritative, and robust to
    deposited-PDB parse failures). For each chain the *local* sequence is used
    when available, falling back to the RCSB canonical sequence otherwise — so a
    structure is never dropped from the AF arm just because the local reader
    choked, since the AlphaFold model is keyed by the UniProt accession.
    """
    record = {
        "code": code,
        "chains": [],
        "uniprot_accessions": [],
        "has_uniprot": False,
        "error": None,
    }

    protein_chains = {
        ch: m for ch, m in chain_map.items() if m.get("polymer_type") == "Protein"
    }

    # Choose the authoritative chain list: RCSB protein chains if mapped,
    # otherwise whatever the local reader produced.
    if protein_chains:
        chain_ids = sorted(protein_chains)
    elif seqs:
        chain_ids = sorted(seqs)
    else:
        if not mapped:
            record["error"] = "entry not found in RCSB"
        else:
            record["error"] = "no protein chains"
        return record

    accessions = set()
    for chain_id in chain_ids:
        meta = chain_map.get(chain_id, {})
        local_seq = seqs.get(chain_id)
        seq = local_seq or meta.get("rcsb_seq") or ""
        if not seq:
            continue
        uni = meta.get("uniprot")
        record["chains"].append(
            {
                "chain_id": chain_id,
                "seq": seq,
                "seq_source": "local" if local_seq else "rcsb",
                "uniprot": uni,
                "polymer_type": meta.get("polymer_type"),
            }
        )
        if uni:
            accessions.add(uni)

    if not record["chains"]:
        record["error"] = "no protein chains"
        return record

    if not seqs:
        record["error"] = "local read failed; sequences from RCSB"

    record["uniprot_accessions"] = sorted(accessions)
    record["has_uniprot"] = bool(accessions)
    return record


# ──────────────────────────────────────────────────────────────────────────────
# Outputs
# ──────────────────────────────────────────────────────────────────────────────

def write_fasta(record):
    """Write one FASTA file per code with a record per protein chain."""
    code = record["code"]
    lines = []
    for ch in record["chains"]:
        uni = ch["uniprot"] or "NA"
        lines.append(f">{code}_{ch['chain_id']}|{uni}")
        lines.append(ch["seq"])
    if lines:
        (FASTA_DIR / f"{code}.fasta").write_text("\n".join(lines) + "\n")


def write_summary(records):
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "n_protein_chains", "n_uniprot", "seq_source", "status"])
        for r in sorted(records, key=lambda x: x["code"]):
            sources = {c.get("seq_source") for c in r["chains"]}
            if not r["chains"]:
                seq_source = "-"
            elif sources == {"local"}:
                seq_source = "local"
            elif sources == {"rcsb"}:
                seq_source = "rcsb"
            else:
                seq_source = "mixed"

            if not r["chains"]:
                status = f"error: {r['error']}"
            elif r["has_uniprot"]:
                status = "ok"
            else:
                status = "no_uniprot"
            w.writerow(
                [
                    r["code"],
                    len(r["chains"]),
                    len(r["uniprot_accessions"]),
                    seq_source,
                    status,
                ]
            )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--codes",
        nargs="+",
        help="Explicit PDB codes to process (default: all in structures.json).",
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="Process only the first N codes."
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent GraphQL chunk requests.",
    )
    args = ap.parse_args()

    if args.codes:
        codes = list(args.codes)
    else:
        codes = json.loads(STRUCTURES_FILE.read_text())
    if args.limit:
        codes = codes[: args.limit]

    FASTA_DIR.mkdir(parents=True, exist_ok=True)

    # Local sequence extraction runs serially: the torchref model reader uses
    # lazy initialisation that is not thread-safe.
    print(f"Extracting local sequences for {len(codes)} structures...")
    local_seqs = {}
    for i, code in enumerate(codes, 1):
        try:
            local_seqs[code] = _local_sequences(code)
        except Exception as e:  # noqa: BLE001
            local_seqs[code] = {}
            print(f"  local read failed for {code}: {e}", file=sys.stderr)
        if i % 100 == 0 or i == len(codes):
            print(f"  [{i:4d}/{len(codes)}]")

    # RCSB UniProt mapping via batched GraphQL: ~50 entries per request.
    chunks = [codes[i : i + GRAPHQL_CHUNK] for i in range(0, len(codes), GRAPHQL_CHUNK)]
    print(
        f"\nQuerying RCSB GraphQL: {len(chunks)} chunks of <= {GRAPHQL_CHUNK} "
        f"({args.workers} concurrent)..."
    )
    chain_maps = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_chain_maps_for_chunk, ch): ch for ch in chunks}
        for fut in as_completed(futures):
            try:
                chain_maps.update(fut.result())
            except Exception as e:  # noqa: BLE001
                print(f"  chunk failed: {e}", file=sys.stderr)
            done += 1
            if done % 5 == 0 or done == len(chunks):
                print(f"  [{done:3d}/{len(chunks)}] chunks")

    records = []
    for code in codes:
        rec = build_record(
            code,
            local_seqs.get(code, {}),
            chain_maps.get(code, {}),
            mapped=code in chain_maps,
        )
        records.append(rec)
        write_fasta(rec)

    records.sort(key=lambda r: r["code"])
    MANIFEST_JSON.write_text(json.dumps(records, indent=2))
    write_summary(records)

    n_ok = sum(1 for r in records if r["has_uniprot"])
    n_no_uni = sum(1 for r in records if not r["has_uniprot"] and r["chains"])
    n_err = sum(1 for r in records if r["error"] and not r["chains"])
    print("\n" + "=" * 60)
    print(f"Total structures:       {len(records)}")
    print(f"With UniProt mapping:   {n_ok}")
    print(f"No UniProt mapping:     {n_no_uni}")
    print(f"Errors / no chains:     {n_err}")
    print(f"\nManifest:  {MANIFEST_JSON}")
    print(f"Summary:   {SUMMARY_CSV}")
    print(f"FASTA dir: {FASTA_DIR}")


if __name__ == "__main__":
    sys.exit(main())
