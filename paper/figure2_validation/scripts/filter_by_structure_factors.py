#!/usr/bin/env python3
"""
Script to filter PDB IDs by checking if structure factors are available.
This creates a new list containing only entries with structure factor data.
"""

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def check_structure_factors(pdb_id: str, session: requests.Session) -> bool:
    """
    Check if structure factors are available for a PDB entry.
    
    Args:
        pdb_id: PDB identifier
        session: requests Session object
    
    Returns:
        True if structure factors are available, False otherwise
    """
    # Check structure factor CIF from RCSB
    sf_url = f"https://files.rcsb.org/download/{pdb_id.lower()}-sf.cif"
    
    try:
        response = session.head(sf_url, timeout=5)
        return response.status_code == 200
    except:
        return False


def filter_pdb_ids(input_file: str, output_file: str, max_workers: int = 20):
    """
    Filter PDB IDs to keep only those with structure factors available.
    
    Args:
        input_file: Input file with PDB IDs (one per line)
        output_file: Output file for filtered PDB IDs
        max_workers: Number of parallel workers
    """
    # Read input PDB IDs
    with open(input_file, 'r') as f:
        pdb_ids = [line.strip() for line in f if line.strip()]
    
    total = len(pdb_ids)
    print(f"Checking structure factor availability for {total} PDB entries...")
    print(f"Using {max_workers} parallel workers")
    print("="*80)
    
    session = requests.Session()
    available_ids = []
    checked = 0
    
    # Check availability in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdb = {
            executor.submit(check_structure_factors, pdb_id, session): pdb_id 
            for pdb_id in pdb_ids
        }
        
        for future in as_completed(future_to_pdb):
            pdb_id = future_to_pdb[future]
            checked += 1
            
            try:
                if future.result():
                    available_ids.append(pdb_id)
                    status = "✓ HAS SF"
                else:
                    status = "✗ NO SF"
            except Exception as e:
                status = f"✗ ERROR: {e}"
            
            # Print progress
            if checked % 50 == 0 or checked == total:
                percent = (checked / total) * 100
                print(f"[{checked:4d}/{total}] {percent:5.1f}% complete | Found {len(available_ids)} with structure factors")
    
    # Save filtered list
    with open(output_file, 'w') as f:
        for pdb_id in available_ids:
            f.write(f"{pdb_id}\n")
    
    # Print summary
    print("\n" + "="*80)
    print("FILTERING SUMMARY")
    print("="*80)
    print(f"Total entries checked:     {total}")
    print(f"With structure factors:    {len(available_ids)} ({len(available_ids)/total*100:.1f}%)")
    print(f"Without structure factors: {total - len(available_ids)} ({(total - len(available_ids))/total*100:.1f}%)")
    print(f"\n✓ Filtered list saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Filter PDB IDs by structure factor availability'
    )
    parser.add_argument(
        '-i', '--input',
        default='pdb_ids_filtered.txt',
        help='Input file with PDB IDs'
    )
    parser.add_argument(
        '-o', '--output',
        default='pdb_ids_with_sf.txt',
        help='Output file for filtered PDB IDs'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=20,
        help='Number of parallel workers (default: 20)'
    )
    
    args = parser.parse_args()
    filter_pdb_ids(args.input, args.output, args.workers)
