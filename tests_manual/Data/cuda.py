from torchref.Data import ReflectionData




m = ReflectionData(verbose=2).load_from_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br/BR_LCLS_refine_8.mtz')



m.cuda()

hkl, Fobs, sigma_Fobs, rfree = m()

print(hkl.device)
print(Fobs.device)
print(sigma_Fobs.device)
print(rfree.device)
