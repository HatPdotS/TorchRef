#!/usr/bin/env python3
"""Test if link plane data is being loaded from CIF files."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torchref.restraints_helper import read_link_definitions

def test_link_planes():
    """Check if TRANS link has plane restraints defined."""
    link_dict, link_list = read_link_definitions()
    
    print(f"Total links loaded: {len(link_dict)}")
    print(f"Link types: {list(link_dict.keys())[:10]}")
    
    if 'TRANS' not in link_dict:
        print("\nERROR: TRANS link not found!")
        return
    
    trans_link = link_dict['TRANS']
    print(f"\nTRANS link keys: {list(trans_link.keys())}")
    
    if 'planes' in trans_link:
        print("\n✓ TRANS link HAS plane restraints!")
        print(f"\nNumber of plane definitions: {len(trans_link['planes'])}")
        print("\nPlane restraints:")
        print(trans_link['planes'])
    else:
        print("\n✗ TRANS link does NOT have plane restraints")
        print("Available sections:", list(trans_link.keys()))

if __name__ == '__main__':
    test_link_planes()
