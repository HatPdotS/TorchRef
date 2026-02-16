"""
Maps module for torchref.

Provides classes for computing and writing crystallographic electron density maps.

Classes
-------
Map
    Base class for 2mFo-DFc and Fcalc maps.
DifferenceMap
    Isomorphous difference map from two datasets.
"""

from torchref.maps.map import Map
from torchref.maps.difference_map import DifferenceMap

__all__ = ["Map", "DifferenceMap"]
