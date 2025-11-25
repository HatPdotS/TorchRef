#!/usr/bin/env python
"""
Test script for the new ReflectionData class.
"""

from torchref.Data import ReflectionData
import torch

print("Test: Load MTZ and filter by resolution")
data = ReflectionData(verbose=0)
data = data.load_from_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.mtz')
filtered = data.filter_by_resolution(d_min=1.5, d_max=10.0)
print(60*"-")

print("Test: Load MTZ and filter by resolution verbose 1")
data = ReflectionData(verbose=1)
data = data.load_from_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.mtz')
filtered = data.filter_by_resolution(d_min=1.5, d_max=10.0)

print(60*"-")

print("Test: Load MTZ and filter by resolution verbose 2")
data = ReflectionData(verbose=2)
data = data.load_from_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.mtz')
filtered = data.filter_by_resolution(d_min=1.7, d_max=10.0)
data_cut_low_res = filtered.filter_by_resolution(d_min=1.5,d_max=3)


filtered_hkl = filtered.get_hkl()

present_data, valid_hkl = data_cut_low_res.validate_hkl(filtered_hkl)


print('valid_hkl sum:', torch.sum(valid_hkl), valid_hkl.shape)
print('present_data shape:', present_data.F.shape)