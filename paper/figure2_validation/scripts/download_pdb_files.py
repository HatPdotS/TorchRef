#!/usr/bin/env python3
"""
Script to download PDB, CIF, and MTZ (structure factor) files for a list of PDB entries.
Each structure gets its own folder in the data directory.
"""

import os
import requests
import time
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


class PDBDownloader:
    """Class to handle downloading PDB files and experimental data."""
    
    # PDB file URLs
    PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
    CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
    
    # Structure factor URLs (MTZ or CIF format)
    SF_CIF_URL = "https://files.rcsb.org/download/{pdb_id}-sf.cif"
    
    # Restraint/ligand files
    LIGAND_CIF_URL = "https://files.rcsb.org/ligands/download/{ligand_code}.cif"
    
    # Alternative structure factor repository
    PDB_REDO_MTZ_URL = "https://pdb-redo.eu/db/{pdb_id}/{pdb_id}_final.mtz"
    
    def __init__(self, base_dir: str = "data", max_workers: int = 5):
        """
        Initialize the downloader.
        
        Args:
            base_dir: Base directory for storing downloaded files
            max_workers: Number of parallel download threads
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        self.session = requests.Session()
        
    def get_structure_dir(self, pdb_id: str) -> Path:
        """Get the directory path for a specific PDB structure."""
        structure_dir = self.base_dir / pdb_id.upper()
        structure_dir.mkdir(exist_ok=True)
        return structure_dir
    
    def download_file(self, url: str, output_path: Path, description: str = "") -> bool:
        """
        Download a file from a URL.
        
        Args:
            url: URL to download from
            output_path: Path to save the file
            description: Description for logging
        
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            file_size = output_path.stat().st_size
            if file_size > 0:
                return True
            else:
                # Empty file, remove it
                output_path.unlink()
                return False
                
        except requests.exceptions.RequestException as e:
            if output_path.exists():
                output_path.unlink()
            return False
    
    def download_pdb(self, pdb_id: str, structure_dir: Path) -> bool:
        """Download PDB format file."""
        url = self.PDB_URL.format(pdb_id=pdb_id.lower())
        output_path = structure_dir / f"{pdb_id.upper()}.pdb"
        
        if output_path.exists():
            return True
        
        return self.download_file(url, output_path, "PDB")
    
    def download_cif(self, pdb_id: str, structure_dir: Path) -> bool:
        """Download mmCIF format file."""
        url = self.CIF_URL.format(pdb_id=pdb_id.lower())
        output_path = structure_dir / f"{pdb_id.upper()}.cif"
        
        if output_path.exists():
            return True
        
        return self.download_file(url, output_path, "CIF")
    
    def download_structure_factors(self, pdb_id: str, structure_dir: Path) -> Tuple[bool, str]:
        """
        Download structure factors (experimental data).
        Tries structure factor CIF first, then PDB-REDO MTZ.
        
        Returns:
            Tuple of (success, format) where format is 'sf-cif', 'mtz', or 'none'
        """
        # Try structure factor CIF from RCSB
        sf_cif_url = self.SF_CIF_URL.format(pdb_id=pdb_id.lower())
        sf_cif_path = structure_dir / f"{pdb_id.upper()}-sf.cif"
        
        if sf_cif_path.exists():
            return True, 'sf-cif'
        
        if self.download_file(sf_cif_url, sf_cif_path, "SF-CIF"):
            return True, 'sf-cif'
        
        # Try MTZ from PDB-REDO
        mtz_url = self.PDB_REDO_MTZ_URL.format(pdb_id=pdb_id.lower())
        mtz_path = structure_dir / f"{pdb_id.upper()}.mtz"
        
        if mtz_path.exists():
            return True, 'mtz'
        
        if self.download_file(mtz_url, mtz_path, "MTZ"):
            return True, 'mtz'
        
        return False, 'none'
    
    def download_restraints(self, pdb_id: str, structure_dir: Path) -> Tuple[bool, int]:
        """
        Download restraint/ligand CIF files by extracting ligand codes from the PDB file.
        
        Returns:
            Tuple of (success, count) where count is number of ligands downloaded
        """
        # First, try to read the PDB file to extract ligand codes
        pdb_file = structure_dir / f"{pdb_id.upper()}.pdb"
        cif_file = structure_dir / f"{pdb_id.upper()}.cif"
        
        ligand_codes = set()
        
        # Try PDB format first (HETNAM records)
        if pdb_file.exists():
            try:
                with open(pdb_file, 'r') as f:
                    for line in f:
                        if line.startswith('HETNAM') or line.startswith('HET   '):
                            # Extract 3-letter code
                            parts = line.split()
                            if len(parts) >= 2:
                                code = parts[1]
                                # Skip water and common ions
                                if code not in ['HOH', 'WAT', 'NA', 'CL', 'K', 'CA', 'MG', 'ZN', 'FE']:
                                    ligand_codes.add(code)
            except:
                pass
        
        # Download restraints for each unique ligand
        downloaded_count = 0
        restraints_dir = structure_dir / "restraints"
        
        if ligand_codes:
            restraints_dir.mkdir(exist_ok=True)
            
            for ligand_code in ligand_codes:
                ligand_url = self.LIGAND_CIF_URL.format(ligand_code=ligand_code)
                ligand_path = restraints_dir / f"{ligand_code}.cif"
                
                if ligand_path.exists() or self.download_file(ligand_url, ligand_path, "LIGAND"):
                    downloaded_count += 1
        
        return (downloaded_count > 0, downloaded_count)
    
    def download_structure(self, pdb_id: str) -> dict:
        """
        Download all available files for a single PDB structure.
        
        Args:
            pdb_id: PDB identifier
        
        Returns:
            Dictionary with download status for each file type
        """
        pdb_id = pdb_id.strip().upper()
        structure_dir = self.get_structure_dir(pdb_id)
        
        results = {
            'pdb_id': pdb_id,
            'pdb': False,
            'cif': False,
            'sf_format': 'none',
            'sf_success': False,
            'restraints': False,
            'restraints_count': 0
        }
        
        # Download PDB file
        results['pdb'] = self.download_pdb(pdb_id, structure_dir)
        
        # Download CIF file
        results['cif'] = self.download_cif(pdb_id, structure_dir)
        
        # Download structure factors
        sf_success, sf_format = self.download_structure_factors(pdb_id, structure_dir)
        results['sf_success'] = sf_success
        results['sf_format'] = sf_format
        
        # Download restraints (requires PDB file to be downloaded first)
        restraints_success, restraints_count = self.download_restraints(pdb_id, structure_dir)
        results['restraints'] = restraints_success
        results['restraints_count'] = restraints_count
        
        return results
    
    def download_from_file(self, input_file: str, progress_file: str = "download_progress.txt"):
        """
        Download files for all PDB IDs listed in a text file.
        
        Args:
            input_file: Path to text file containing PDB IDs (one per line)
            progress_file: Path to save progress information
        """
        # Read PDB IDs
        with open(input_file, 'r') as f:
            pdb_ids = [line.strip() for line in f if line.strip()]
        
        total = len(pdb_ids)
        print(f"Starting download for {total} PDB structures...")
        print(f"Using {self.max_workers} parallel workers")
        print(f"Base directory: {self.base_dir.absolute()}")
        print("="*80)
        
        # Statistics
        stats = {
            'total': total,
            'completed': 0,
            'pdb_success': 0,
            'cif_success': 0,
            'sf_success': 0,
            'sf_cif': 0,
            'sf_mtz': 0,
            'restraints_success': 0,
            'failed': []
        }
        
        # Download with progress tracking
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_pdb = {
                executor.submit(self.download_structure, pdb_id): pdb_id 
                for pdb_id in pdb_ids
            }
            
            # Process completed downloads
            for future in as_completed(future_to_pdb):
                pdb_id = future_to_pdb[future]
                try:
                    result = future.result()
                    stats['completed'] += 1
                    
                    # Update statistics
                    if result['pdb']:
                        stats['pdb_success'] += 1
                    if result['cif']:
                        stats['cif_success'] += 1
                    if result['sf_success']:
                        stats['sf_success'] += 1
                        if result['sf_format'] == 'sf-cif':
                            stats['sf_cif'] += 1
                        elif result['sf_format'] == 'mtz':
                            stats['sf_mtz'] += 1
                    if result['restraints']:
                        stats['restraints_success'] += 1
                    
                    # Track failures
                    if not (result['pdb'] or result['cif']):
                        stats['failed'].append(pdb_id)
                    
                    # Progress output
                    status_symbols = []
                    status_symbols.append('✓' if result['pdb'] else '✗')
                    status_symbols.append('✓' if result['cif'] else '✗')
                    
                    if result['sf_success']:
                        if result['sf_format'] == 'sf-cif':
                            status_symbols.append('✓[SF-CIF]')
                        elif result['sf_format'] == 'mtz':
                            status_symbols.append('✓[MTZ]')
                    else:
                        status_symbols.append('✗')
                    
                    # Add restraints status with count
                    if result['restraints']:
                        status_symbols.append(f"✓[{result['restraints_count']}]")
                    else:
                        status_symbols.append('✗')
                    
                    status = f"PDB:{status_symbols[0]} CIF:{status_symbols[1]} SF:{status_symbols[2]} RESTR:{status_symbols[3]}"
                    
                    print(f"[{stats['completed']:4d}/{total}] {pdb_id}: {status}")
                    
                except Exception as e:
                    print(f"[{stats['completed']:4d}/{total}] {pdb_id}: ERROR - {e}")
                    stats['failed'].append(pdb_id)
                    stats['completed'] += 1
        
        # Print summary
        print("\n" + "="*80)
        print("DOWNLOAD SUMMARY")
        print("="*80)
        print(f"Total structures processed: {stats['completed']}/{stats['total']}")
        print(f"PDB files downloaded:      {stats['pdb_success']} ({stats['pdb_success']/total*100:.1f}%)")
        print(f"CIF files downloaded:      {stats['cif_success']} ({stats['cif_success']/total*100:.1f}%)")
        print(f"Structure factors (total): {stats['sf_success']} ({stats['sf_success']/total*100:.1f}%)")
        print(f"  - SF-CIF format:         {stats['sf_cif']}")
        print(f"  - MTZ format:            {stats['sf_mtz']}")
        print(f"Restraint files:           {stats['restraints_success']} ({stats['restraints_success']/total*100:.1f}%)")
        print(f"Completely failed:         {len(stats['failed'])}")
        
        # Save progress and failed list
        with open(progress_file, 'w') as f:
            f.write("DOWNLOAD SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Total structures processed: {stats['completed']}/{stats['total']}\n")
            f.write(f"PDB files downloaded:      {stats['pdb_success']}\n")
            f.write(f"CIF files downloaded:      {stats['cif_success']}\n")
            f.write(f"Structure factors:         {stats['sf_success']}\n")
            f.write(f"  - SF-CIF format:         {stats['sf_cif']}\n")
            f.write(f"  - MTZ format:            {stats['sf_mtz']}\n")
            f.write(f"Restraint files:           {stats['restraints_success']}\n")
            f.write(f"\nFailed downloads ({len(stats['failed'])}):\n")
            for pdb_id in stats['failed']:
                f.write(f"{pdb_id}\n")
        
        print(f"\nProgress saved to: {progress_file}")
        print("="*80)
        
        return stats


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download PDB, CIF, and structure factor files for PDB entries.'
    )
    parser.add_argument(
        '-i', '--input',
        default='pdb_ids_filtered.txt',
        help='Input file containing PDB IDs (one per line)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='data',
        help='Output directory for downloaded files'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=5,
        help='Number of parallel download workers (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        return
    
    # Create downloader and start downloading
    downloader = PDBDownloader(base_dir=args.output_dir, max_workers=args.workers)
    downloader.download_from_file(args.input)


if __name__ == "__main__":
    main()
