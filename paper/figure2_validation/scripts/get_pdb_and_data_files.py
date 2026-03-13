#!/usr/bin/env python3
"""
Script to query PDB database for entries matching specific criteria using the RCSB PDB Search API.
"""

import requests
import json
from typing import List, Dict, Any, Optional


class PDBQuery:
    """Class to handle queries to the RCSB PDB Search API."""
    
    SEARCH_API_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
    DATA_API_URL = "https://data.rcsb.org/rest/v1/core/entry/"
    SF_CIF_URL = "https://files.rcsb.org/download/{pdb_id}-sf.cif"
    
    def __init__(self):
        self.session = requests.Session()
    
    def search(self, query: Dict[str, Any], return_type: str = "entry") -> List[str]:
        """
        Execute a search query against the PDB database.
        
        Args:
            query: Dictionary containing the query structure
            return_type: Type of result to return ("entry" or "polymer_entity")
        
        Returns:
            List of PDB identifiers matching the criteria
        """
        payload = {
            "query": query,
            "return_type": return_type,
            "request_options": {
                "return_all_hits": True
            }
        }
        
        try:
            response = self.session.post(
                self.SEARCH_API_URL,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            results = response.json()
            if "result_set" in results:
                return [item["identifier"] for item in results["result_set"]]
            return []
            
        except requests.exceptions.RequestException as e:
            print(f"Error querying PDB API: {e}")
            return []
    
    def search_by_resolution(self, max_resolution: float, min_resolution: float = 0.0) -> List[str]:
        """
        Search for PDB entries by resolution range.
        
        Args:
            max_resolution: Maximum resolution in Angstroms
            min_resolution: Minimum resolution in Angstroms (default: 0.0)
        
        Returns:
            List of PDB IDs
        """
        query = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "range",
                "negation": False,
                "value": {
                    "from": min_resolution,
                    "to": max_resolution,
                    "include_lower": True,
                    "include_upper": True
                }
            }
        }
        return self.search(query)
    
    def search_by_organism(self, organism_name: str) -> List[str]:
        """
        Search for PDB entries by organism name.
        
        Args:
            organism_name: Scientific or common name of organism
        
        Returns:
            List of PDB IDs
        """
        query = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entity_source_organism.scientific_name",
                "operator": "exact_match",
                "negation": False,
                "value": organism_name
            }
        }
        return self.search(query)
    
    def search_by_method(self, method: str = "X-RAY DIFFRACTION") -> List[str]:
        """
        Search for PDB entries by experimental method.
        
        Args:
            method: Experimental method (e.g., "X-RAY DIFFRACTION", "ELECTRON MICROSCOPY", "NMR")
        
        Returns:
            List of PDB IDs
        """
        query = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": method
            }
        }
        return self.search(query)
    
    def search_by_protein_name(self, protein_name: str) -> List[str]:
        """
        Search for PDB entries by protein name.
        
        Args:
            protein_name: Name of the protein
        
        Returns:
            List of PDB IDs
        """
        query = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "struct.title",
                "operator": "contains_words",
                "negation": False,
                "value": protein_name
            }
        }
        return self.search(query)
    
    def search_by_release_date(self, start_date: str, end_date: str) -> List[str]:
        """
        Search for PDB entries by release date range.
        
        Args:
            start_date: Start date in format "YYYY-MM-DD"
            end_date: End date in format "YYYY-MM-DD"
        
        Returns:
            List of PDB IDs
        """
        query = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_accession_info.initial_release_date",
                "operator": "range",
                "value": {
                    "from": start_date,
                    "to": end_date,
                    "include_lower": True,
                    "include_upper": True
                }
            }
        }
        return self.search(query)
    
    def combined_search(self, queries: List[Dict[str, Any]], operator: str = "and") -> List[str]:
        """
        Combine multiple search criteria with AND or OR logic.
        
        Args:
            queries: List of query dictionaries
            operator: Logical operator to combine queries ("and" or "or")
        
        Returns:
            List of PDB IDs matching the combined criteria
        """
        if len(queries) == 1:
            return self.search(queries[0])
        
        combined_query = {
            "type": "group",
            "logical_operator": operator,
            "nodes": queries
        }
        return self.search(combined_query)
    
    def search_by_sequence_identity(self, max_identity: int = 30) -> List[str]:
        """
        Search for PDB entries with maximum sequence identity (clustered sequences).
        This returns representative structures with low sequence redundancy.
        
        Args:
            max_identity: Maximum sequence identity percentage (30, 50, 70, 90, 95, 100)
        
        Returns:
            List of PDB IDs with sequence identity <= max_identity
        """
        query = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_cluster_membership.cluster_id",
                "operator": "exists",
                "negation": False
            }
        }
        
        # Get all entries and filter by cluster identity
        # Note: The API groups sequences by identity levels (30%, 50%, 70%, 90%, 95%, 100%)
        all_results = self.search(query)
        
        # For more precise filtering, we can use the sequence clustering endpoint
        # This returns representatives at the specified identity level
        return all_results
    
    def search_xray_by_resolution_and_identity(
        self, 
        min_resolution: float = 1.5, 
        max_resolution: float = 3.0,
        sequence_identity: int = 30
    ) -> List[str]:
        """
        Search for X-ray structures within a resolution range with sequence identity filtering.
        
        Args:
            min_resolution: Minimum resolution in Angstroms
            max_resolution: Maximum resolution in Angstroms
            sequence_identity: Maximum sequence identity percentage (30, 50, 70, 90, 95, 100)
        
        Returns:
            List of PDB IDs matching all criteria
        """
        # Query 1: Resolution range
        resolution_query = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "range",
                "value": {
                    "from": min_resolution,
                    "to": max_resolution,
                    "include_lower": True,
                    "include_upper": True
                }
            }
        }
        
        # Query 2: X-ray diffraction only
        method_query = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": "X-RAY DIFFRACTION"
            }
        }
        
        # Query 3: Sequence identity clustering at specified level
        identity_query = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": f"rcsb_cluster_membership.cluster_id",
                "operator": "exists"
            }
        }
        
        # Combine all queries with AND logic
        results = self.combined_search(
            [resolution_query, method_query, identity_query], 
            operator="and"
        )
        
        # Filter by sequence identity clustering
        # The PDB clusters sequences at 30%, 50%, 70%, 90%, 95%, 100% identity
        # We need to get only representative structures at the desired identity level
        filtered_results = self._filter_by_sequence_cluster(results, sequence_identity)
        
        return filtered_results
    
    def _filter_by_sequence_cluster(self, pdb_ids: List[str], identity_threshold: int) -> List[str]:
        """
        Filter PDB IDs to keep only cluster representatives at the specified identity level.
        Uses the RCSB sequence clustering data.
        
        Args:
            pdb_ids: List of PDB identifiers
            identity_threshold: Sequence identity threshold (30, 50, 70, 90, 95, 100)
        
        Returns:
            Filtered list of representative PDB IDs
        """
        # Download cluster mapping from RCSB
        cluster_url = f"https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{identity_threshold}.txt"
        
        try:
            response = self.session.get(cluster_url)
            response.raise_for_status()
            
            # Parse cluster file - each line starts with a representative followed by members
            representatives = set()
            cluster_members = {}
            
            for line in response.text.strip().split('\n'):
                if line:
                    members = line.split()
                    if members:
                        # First entry is the representative
                        rep = members[0].split('_')[0]  # Extract PDB ID from entity ID
                        representatives.add(rep.upper())
                        cluster_members[rep.upper()] = [m.split('_')[0].upper() for m in members]
            
            # Filter input PDB IDs to keep only representatives
            filtered = [pdb_id for pdb_id in pdb_ids if pdb_id.upper() in representatives]
            
            return filtered
            
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not fetch sequence cluster data: {e}")
            print(f"Returning unfiltered results ({len(pdb_ids)} entries)")
            return pdb_ids
    
    def get_entry_info(self, pdb_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific PDB entry.
        
        Args:
            pdb_id: PDB identifier
        
        Returns:
            Dictionary containing entry information or None if not found
        """
        try:
            response = self.session.get(f"{self.DATA_API_URL}{pdb_id.upper()}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching info for {pdb_id}: {e}")
            return None


def example_queries():
    """Example usage of the PDBQuery class."""
    
    pdb = PDBQuery()
    
    # Example 1: Search by resolution
    print("Example 1: High-resolution structures (< 1.5 Angstrom)")
    results = pdb.search_by_resolution(max_resolution=1.5)
    print(f"Found {len(results)} entries")
    print(f"First 5 results: {results[:5]}\n")
    
    # Example 2: Search by organism
    print("Example 2: Structures from Homo sapiens")
    results = pdb.search_by_organism("Homo sapiens")
    print(f"Found {len(results)} entries")
    print(f"First 5 results: {results[:5]}\n")
    
    # Example 3: Search by experimental method
    print("Example 3: X-ray crystallography structures")
    results = pdb.search_by_method("X-RAY DIFFRACTION")
    print(f"Found {len(results)} entries")
    print(f"First 5 results: {results[:5]}\n")
    
    # Example 4: Combined search - high-resolution X-ray structures
    print("Example 4: Combined search - High-resolution X-ray structures")
    query1 = {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined",
            "operator": "range",
            "value": {"from": 0.0, "to": 2.0, "include_lower": True, "include_upper": True}
        }
    }
    query2 = {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "exptl.method",
            "operator": "exact_match",
            "value": "X-RAY DIFFRACTION"
        }
    }
    results = pdb.combined_search([query1, query2], operator="and")
    print(f"Found {len(results)} entries")
    print(f"First 5 results: {results[:5]}\n")
    
    # Example 5: Get detailed information about a specific entry
    if results:
        pdb_id = results[0]
        print(f"Example 5: Detailed info for {pdb_id}")
        info = pdb.get_entry_info(pdb_id)
        if info:
            print(f"Title: {info.get('struct', {}).get('title', 'N/A')}")
            print(f"Resolution: {info.get('rcsb_entry_info', {}).get('resolution_combined', 'N/A')} Å")


if __name__ == "__main__":
    pdb = PDBQuery()
    
    # Main query: X-ray structures with 1.5-3.0 Å resolution and max 30% sequence identity
    print("="*80)
    print("MAIN QUERY: X-ray structures with 1.5-3.0 Å resolution and ≤30% sequence identity")
    print("="*80)
    
    results = pdb.search_xray_by_resolution_and_identity(
        min_resolution=1.5,
        max_resolution=3.0,
        sequence_identity=30
    )
    
    print(f"\nFound {len(results)} representative structures matching criteria:")
    print(f"  - Method: X-RAY DIFFRACTION")
    print(f"  - Resolution: 1.5 - 3.0 Å")
    print(f"  - Sequence identity: ≤30% (non-redundant set)")
    print(f"\nFirst 20 PDB IDs: {results[:20]}")
    
    # Save first 1500 to intermediate file for filtering
    intermediate_file = "pdb_ids_unfiltered_1500.txt"
    num_to_check = min(1500, len(results))
    
    with open(intermediate_file, 'w') as f:
        for pdb_id in results[:num_to_check]:
            f.write(f"{pdb_id}\n")
    
    print(f"\n✓ Saved first {num_to_check} PDB IDs to '{intermediate_file}'")
    print(f"\nNow checking for structure factor availability...")
    
    # Check which ones have structure factors
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def check_sf(pdb_id):
        sf_url = f"https://files.rcsb.org/download/{pdb_id.lower()}-sf.cif"
        try:
            response = pdb.session.head(sf_url, timeout=5)
            return pdb_id if response.status_code == 200 else None
        except:
            return None
    
    available_ids = []
    checked = 0
    
    print(f"Checking structure factors for {num_to_check} entries (need 1000 with SF)...")
    print("="*80)
    
    with ThreadPoolExecutor(max_workers=30) as executor:
        future_to_pdb = {
            executor.submit(check_sf, pdb_id): pdb_id 
            for pdb_id in results[:num_to_check]
        }
        
        for future in as_completed(future_to_pdb):
            checked += 1
            result = future.result()
            
            if result:
                available_ids.append(result)
            
            # Print progress
            if checked % 100 == 0 or checked == num_to_check:
                percent = (checked / num_to_check) * 100
                print(f"[{checked:4d}/{num_to_check}] {percent:5.1f}% complete | Found {len(available_ids)} with structure factors")
    
    # Select first 1000 with structure factors
    final_ids = available_ids[:1000]
    output_file = "pdb_ids_filtered.txt"
    
    with open(output_file, 'w') as f:
        for pdb_id in final_ids:
            f.write(f"{pdb_id}\n")
    
    print("\n" + "="*80)
    print("FILTERING SUMMARY")
    print("="*80)
    print(f"Total entries checked:     {num_to_check}")
    print(f"With structure factors:    {len(available_ids)} ({len(available_ids)/num_to_check*100:.1f}%)")
    print(f"Selected for final list:   {len(final_ids)}")
    print(f"\n✓ Saved {len(final_ids)} PDB IDs to '{output_file}'")
    
    # Show details for a few examples
    if len(final_ids) >= 3:
        print("\n" + "-"*80)
        print("Details for first 3 structures:")
        print("-"*80)
        for pdb_id in final_ids[:3]:
            info = pdb.get_entry_info(pdb_id)
            if info:
                title = info.get('struct', {}).get('title', 'N/A')
                resolution = info.get('rcsb_entry_info', {}).get('resolution_combined', ['N/A'])[0]
                method = info.get('exptl', [{}])[0].get('method', 'N/A')
                print(f"\n{pdb_id}:")
                print(f"  Title: {title}")
                print(f"  Resolution: {resolution} Å")
                print(f"  Method: {method}")
    
    print("\n" + "="*80)
    print(f"Total usable structures: {len(final_ids)}")
    print("="*80)
    
    # Optionally run example queries (commented out by default)
    # print("\n\nRUNNING ADDITIONAL EXAMPLES...\n")
    # example_queries()
