#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u
#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/Dark_LBFGS/screen_multiple.out

from torchref.model_ft import ModelFT
from torchref.scaler import Scaler
from torchref.Data import ReflectionData
import torch
import matplotlib.pyplot as plt
import os
data = ReflectionData(verbose=2).load_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.mtz')

M = ModelFT(verbose=0,max_res=1.7).load_pdb('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.pdb')
data.filter_by_resolution(d_min=1.7)

S = Scaler(M, data, nbins=10,verbose=0)
S.initialize()

outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/Dark_LBFGS/all_corrections'

os.makedirs(outdir, exist_ok=True)

S.screen_solvent_params()

data.find_outliers(M, S, z_threshold=4.0)

rwork, rfree = S.rfactor()
print(f"Initial R-work: {rwork:.4f}, R-free: {rfree:.4f}")

fcalc = S(M(data.get_hkl()))
_, fobs, _, _ = data()

plt.scatter(torch.abs(fobs).cpu().numpy(), torch.abs(fcalc).detach().cpu().numpy(), alpha=0.5)
plt.plot([0,500],[0,500], color='red')
plt.xlim(0,500)
plt.ylim(0,500)
plt.xlabel('Observed |F|')
plt.ylabel('Calculated |F| before Scaling Fit')
plt.title('Observed vs Calculated Structure Factors before Scaling Fit')
plt.savefig(f'{outdir}/observed_vs_calculated_before_fit.png')
plt.close()


S.verbose=10
S.refine_lbfgs()
S.verbose=0
rwork, rfree = S.rfactor()
print(f"Post-fitting R-work: {rwork:.4f}, R-free: {rfree:.4f}")

data.find_outliers(M, S,z_threshold=4.0)

rwork, rfree = S.rfactor()
print(f"Post-outlier R-work: {rwork:.4f}, R-free: {rfree:.4f}")
max_res, rwork, rfree = S.bin_wise_rfactor()
print(max_res, rwork, rfree)


plt.plot(max_res.cpu().numpy(), rwork.cpu().numpy(), label='R-work')
plt.plot(max_res.cpu().numpy(), rfree.cpu().numpy(), label='R-free')
plt.xlabel('Resolution (Å)')
plt.ylabel('R-factor')
plt.title('Bin-wise R-factors after Scaling Fit')
plt.legend()
plt.savefig(f'{outdir}/bin_wise_rfactors.png')

plt.close()


fcalc = S(M(data.get_hkl()))
_, fobs, _, _ = data()

plt.scatter(torch.abs(fobs).cpu().numpy(), torch.abs(fcalc).detach().cpu().numpy(), alpha=0.5)
plt.plot([0,500],[0,500], color='red')
plt.xlim(0,500)
plt.ylim(0,500)
plt.xlabel('Observed |F|')
plt.ylabel('Calculated |F| after Scaling')
plt.title('Observed vs Calculated Structure Factors after Scaling Fit')
plt.savefig(f'{outdir}/observed_vs_calculated.png')
plt.close()

fobsmean, fcalcmean, res = S.get_binwise_mean_intensity()
plt.plot(res.detach().cpu().numpy(), fobsmean.detach().cpu().numpy(), label='Observed Mean |F|')
plt.plot(res.detach().cpu().numpy(), fcalcmean.detach().cpu().numpy(), label='Calculated Mean |F|')
plt.xlabel('Resolution (Å)')
plt.ylabel('Mean |F|')
plt.title('Bin-wise Mean Structure Factor Amplitudes after Scaling Fit')
plt.legend()
plt.savefig(f'{outdir}/mean_structure_factors.png')
plt.close()

res, work, rfree = S.bin_wise_rfactor()



plt.plot(res.detach().cpu().numpy(), work.detach().cpu().numpy(), label='R-work')
plt.plot(res.detach().cpu().numpy(), rfree.detach().cpu().numpy(), label='R-free')
plt.xlabel('Resolution (Å)')
plt.ylabel('R-factor')
plt.title('Bin-wise R-factors after Scaling Fit')
plt.legend()
plt.savefig(f'{outdir}/bin_wise_rfactors_post_fit.png')
plt.close()

print("\n\n" + "="*80)
print("Testing anisotropy-only fitting now...")
print("\n\n" + "="*80)

data = ReflectionData(verbose=2).load_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.mtz')

M = ModelFT(verbose=0,max_res=1.7).load_pdb('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.pdb')
data.filter_by_resolution(d_min=1.7)

S = Scaler(M, data, nbins=10,verbose=0)
S.calc_initial_scale()
S.setup_anisotropy_correction()
outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/Dark_LBFGS/anisotropic_only'

os.makedirs(outdir, exist_ok=True)

data.find_outliers(M, S, z_threshold=4.0)

rwork, rfree = S.rfactor()
print(f"Initial R-work: {rwork:.4f}, R-free: {rfree:.4f}")

fcalc = S(M(data.get_hkl()))
_, fobs, _, _ = data()

plt.scatter(torch.abs(fobs).cpu().numpy(), torch.abs(fcalc).detach().cpu().numpy(), alpha=0.5)
plt.plot([0,500],[0,500], color='red')
plt.xlim(0,500)
plt.ylim(0,500)
plt.xlabel('Observed |F|')
plt.ylabel('Calculated |F| before Scaling Fit')
plt.title('Observed vs Calculated Structure Factors before Scaling Fit')
plt.savefig(f'{outdir}/observed_vs_calculated_before_fit.png')
plt.close()
S.verbose=10
S.refine_lbfgs()
S.verbose=0
rwork, rfree = S.rfactor()
print(f"Post-fitting R-work: {rwork:.4f}, R-free: {rfree:.4f}")

data.find_outliers(M, S,z_threshold=4.0)

rwork, rfree = S.rfactor()
print(f"Post-outlier R-work: {rwork:.4f}, R-free: {rfree:.4f}")
max_res, rwork, rfree = S.bin_wise_rfactor()


plt.plot(max_res.cpu().numpy(), rwork.cpu().numpy(), label='R-work')
plt.plot(max_res.cpu().numpy(), rfree.cpu().numpy(), label='R-free')
plt.xlabel('Resolution (Å)')
plt.ylabel('R-factor')
plt.title('Bin-wise R-factors after Scaling Fit')
plt.legend()
plt.savefig(f'{outdir}/bin_wise_rfactors.png')

plt.close()


fcalc = S(M(data.get_hkl()))
_, fobs, _, _ = data()

plt.scatter(torch.abs(fobs).cpu().numpy(), torch.abs(fcalc).detach().cpu().numpy(), alpha=0.5)
plt.plot([0,500],[0,500], color='red')
plt.xlim(0,500)
plt.ylim(0,500)
plt.xlabel('Observed |F|')
plt.ylabel('Calculated |F| after Scaling')
plt.title('Observed vs Calculated Structure Factors after Scaling Fit')
plt.savefig(f'{outdir}/observed_vs_calculated.png')
plt.close()

fobsmean, fcalcmean, res = S.get_binwise_mean_intensity()
plt.plot(res.detach().cpu().numpy(), fobsmean.detach().cpu().numpy(), label='Observed Mean |F|')
plt.plot(res.detach().cpu().numpy(), fcalcmean.detach().cpu().numpy(), label='Calculated Mean |F|')
plt.xlabel('Resolution (Å)')
plt.ylabel('Mean |F|')
plt.title('Bin-wise Mean Structure Factor Amplitudes after Scaling Fit')
plt.legend()
plt.savefig(f'{outdir}/mean_structure_factors.png')
plt.close()

res, work, rfree = S.bin_wise_rfactor()

plt.plot(res.detach().cpu().numpy(), work.detach().cpu().numpy(), label='R-work')
plt.plot(res.detach().cpu().numpy(), rfree.detach().cpu().numpy(), label='R-free')
plt.xlabel('Resolution (Å)')
plt.ylabel('R-factor')
plt.title('Bin-wise R-factors after Scaling Fit')
plt.legend()
plt.savefig(f'{outdir}/bin_wise_rfactors_post_fit.png')
plt.close()

print("\n\n" + "="*80)
print("Testing solvent-only fitting now...")
print("\n\n" + "="*80)

data = ReflectionData(verbose=2).load_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.mtz')

M = ModelFT(verbose=0,max_res=1.7).load_pdb('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.pdb')
data.filter_by_resolution(d_min=1.7)

S = Scaler(M, data, nbins=10,verbose=0)
S.calc_initial_scale()
S.setup_solvent()
outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/Dark_LBFGS/only_solvent'

os.makedirs(outdir, exist_ok=True)

S.screen_solvent_params()

data.find_outliers(M, S, z_threshold=4.0)

rwork, rfree = S.rfactor()
print(f"Initial R-work: {rwork:.4f}, R-free: {rfree:.4f}")

fcalc = S(M(data.get_hkl()))
_, fobs, _, _ = data()

plt.scatter(torch.abs(fobs).cpu().numpy(), torch.abs(fcalc).detach().cpu().numpy(), alpha=0.5)
plt.plot([0,500],[0,500], color='red')
plt.xlim(0,500)
plt.ylim(0,500)
plt.xlabel('Observed |F|')
plt.ylabel('Calculated |F| before Scaling Fit')
plt.title('Observed vs Calculated Structure Factors before Scaling Fit')
plt.savefig(f'{outdir}/observed_vs_calculated_before_fit.png')
plt.close()
S.verbose=10
S.refine_lbfgs()
S.verbose=0
rwork, rfree = S.rfactor()
print(f"Post-fitting R-work: {rwork:.4f}, R-free: {rfree:.4f}")

data.find_outliers(M, S,z_threshold=4.0)

rwork, rfree = S.rfactor()
print(f"Post-outlier R-work: {rwork:.4f}, R-free: {rfree:.4f}")
max_res, rwork, rfree = S.bin_wise_rfactor()


plt.plot(max_res.cpu().numpy(), rwork.cpu().numpy(), label='R-work')
plt.plot(max_res.cpu().numpy(), rfree.cpu().numpy(), label='R-free')
plt.xlabel('Resolution (Å)')
plt.ylabel('R-factor')
plt.title('Bin-wise R-factors after Scaling Fit')
plt.legend()
plt.savefig(f'{outdir}/bin_wise_rfactors.png')

plt.close()


fcalc = S(M(data.get_hkl()))
_, fobs, _, _ = data()

plt.scatter(torch.abs(fobs).cpu().numpy(), torch.abs(fcalc).detach().cpu().numpy(), alpha=0.5)
plt.plot([0,500],[0,500], color='red')
plt.xlim(0,500)
plt.ylim(0,500)
plt.xlabel('Observed |F|')
plt.ylabel('Calculated |F| after Scaling')
plt.title('Observed vs Calculated Structure Factors after Scaling Fit')
plt.savefig(f'{outdir}/observed_vs_calculated.png')
plt.close()

fobsmean, fcalcmean, res = S.get_binwise_mean_intensity()
plt.plot(res.detach().cpu().numpy(), fobsmean.detach().cpu().numpy(), label='Observed Mean |F|')
plt.plot(res.detach().cpu().numpy(), fcalcmean.detach().cpu().numpy(), label='Calculated Mean |F|')
plt.xlabel('Resolution (Å)')
plt.ylabel('Mean |F|')
plt.title('Bin-wise Mean Structure Factor Amplitudes after Scaling Fit')
plt.legend()
plt.savefig(f'{outdir}/mean_structure_factors.png')
plt.close()

res, work, rfree = S.bin_wise_rfactor()

plt.plot(res.detach().cpu().numpy(), work.detach().cpu().numpy(), label='R-work')
plt.plot(res.detach().cpu().numpy(), rfree.detach().cpu().numpy(), label='R-free')
plt.xlabel('Resolution (Å)')
plt.ylabel('R-factor')
plt.title('Bin-wise R-factors after Scaling Fit')
plt.legend()
plt.savefig(f'{outdir}/bin_wise_rfactors_post_fit.png')
plt.close()


data = ReflectionData(verbose=2).load_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.mtz')

M = ModelFT(verbose=0,max_res=1.7).load_pdb('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.pdb')
data.filter_by_resolution(d_min=1.7)

S = Scaler(M, data, nbins=10,verbose=0)
S.initialize()
S.setup_bin_wise_bfactor()

outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/Dark_LBFGS/all_corrections_bin_wise_B'

os.makedirs(outdir, exist_ok=True)

S.screen_solvent_params()

data.find_outliers(M, S, z_threshold=4.0)

rwork, rfree = S.rfactor()
print(f"Initial R-work: {rwork:.4f}, R-free: {rfree:.4f}")

fcalc = S(M(data.get_hkl()))
_, fobs, _, _ = data()

plt.scatter(torch.abs(fobs).cpu().numpy(), torch.abs(fcalc).detach().cpu().numpy(), alpha=0.5)
plt.plot([0,500],[0,500], color='red')
plt.xlim(0,500)
plt.ylim(0,500)
plt.xlabel('Observed |F|')
plt.ylabel('Calculated |F| before Scaling Fit')
plt.title('Observed vs Calculated Structure Factors before Scaling Fit')
plt.savefig(f'{outdir}/observed_vs_calculated_before_fit.png')
plt.close()


S.verbose=10
S.refine_lbfgs()
S.verbose=0
rwork, rfree = S.rfactor()
print(f"Post-fitting R-work: {rwork:.4f}, R-free: {rfree:.4f}")

data.find_outliers(M, S,z_threshold=4.0)

rwork, rfree = S.rfactor()
print(f"Post-outlier R-work: {rwork:.4f}, R-free: {rfree:.4f}")
max_res, rwork, rfree = S.bin_wise_rfactor()
print(max_res, rwork, rfree)


plt.plot(max_res.cpu().numpy(), rwork.cpu().numpy(), label='R-work')
plt.plot(max_res.cpu().numpy(), rfree.cpu().numpy(), label='R-free')
plt.xlabel('Resolution (Å)')
plt.ylabel('R-factor')
plt.title('Bin-wise R-factors after Scaling Fit')
plt.legend()
plt.savefig(f'{outdir}/bin_wise_rfactors.png')

plt.close()


fcalc = S(M(data.get_hkl()))
_, fobs, _, _ = data()

plt.scatter(torch.abs(fobs).cpu().numpy(), torch.abs(fcalc).detach().cpu().numpy(), alpha=0.5)
plt.plot([0,500],[0,500], color='red')
plt.xlim(0,500)
plt.ylim(0,500)
plt.xlabel('Observed |F|')
plt.ylabel('Calculated |F| after Scaling')
plt.title('Observed vs Calculated Structure Factors after Scaling Fit')
plt.savefig(f'{outdir}/observed_vs_calculated.png')
plt.close()

fobsmean, fcalcmean, res = S.get_binwise_mean_intensity()
plt.plot(res.detach().cpu().numpy(), fobsmean.detach().cpu().numpy(), label='Observed Mean |F|')
plt.plot(res.detach().cpu().numpy(), fcalcmean.detach().cpu().numpy(), label='Calculated Mean |F|')
plt.xlabel('Resolution (Å)')
plt.ylabel('Mean |F|')
plt.title('Bin-wise Mean Structure Factor Amplitudes after Scaling Fit')
plt.legend()
plt.savefig(f'{outdir}/mean_structure_factors.png')
plt.close()

res, work, rfree = S.bin_wise_rfactor()



plt.plot(res.detach().cpu().numpy(), work.detach().cpu().numpy(), label='R-work')
plt.plot(res.detach().cpu().numpy(), rfree.detach().cpu().numpy(), label='R-free')
plt.xlabel('Resolution (Å)')
plt.ylabel('R-factor')
plt.title('Bin-wise R-factors after Scaling Fit')
plt.legend()
plt.savefig(f'{outdir}/bin_wise_rfactors_post_fit.png')
plt.close()

print("\n\n" + "="*80)
print("Testing not stripping hydrogen fitting now...")
print("\n\n" + "="*80)



data = ReflectionData(verbose=2).load_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.mtz')

M = ModelFT(verbose=0,max_res=1.7,strip_H=False).load_pdb('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.pdb')
data.filter_by_resolution(d_min=1.7)

S = Scaler(M, data, nbins=10,verbose=0)
S.initialize()

outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/Dark_LBFGS/all_corrections_with_hydrogen'

os.makedirs(outdir, exist_ok=True)

S.screen_solvent_params()

data.find_outliers(M, S, z_threshold=4.0)

rwork, rfree = S.rfactor()
print(f"Initial R-work: {rwork:.4f}, R-free: {rfree:.4f}")

fcalc = S(M(data.get_hkl()))
_, fobs, _, _ = data()

plt.scatter(torch.abs(fobs).cpu().numpy(), torch.abs(fcalc).detach().cpu().numpy(), alpha=0.5)
plt.plot([0,500],[0,500], color='red')
plt.xlim(0,500)
plt.ylim(0,500)
plt.xlabel('Observed |F|')
plt.ylabel('Calculated |F| before Scaling Fit')
plt.title('Observed vs Calculated Structure Factors before Scaling Fit')
plt.savefig(f'{outdir}/observed_vs_calculated_before_fit.png')
plt.close()


S.verbose=10
S.refine_lbfgs()
S.verbose=0
rwork, rfree = S.rfactor()
print(f"Post-fitting R-work: {rwork:.4f}, R-free: {rfree:.4f}")

data.find_outliers(M, S,z_threshold=4.0)

rwork, rfree = S.rfactor()
print(f"Post-outlier R-work: {rwork:.4f}, R-free: {rfree:.4f}")
max_res, rwork, rfree = S.bin_wise_rfactor()
print(max_res, rwork, rfree)


plt.plot(max_res.cpu().numpy(), rwork.cpu().numpy(), label='R-work')
plt.plot(max_res.cpu().numpy(), rfree.cpu().numpy(), label='R-free')
plt.xlabel('Resolution (Å)')
plt.ylabel('R-factor')
plt.title('Bin-wise R-factors after Scaling Fit')
plt.legend()
plt.savefig(f'{outdir}/bin_wise_rfactors.png')

plt.close()


fcalc = S(M(data.get_hkl()))
_, fobs, _, _ = data()

plt.scatter(torch.abs(fobs).cpu().numpy(), torch.abs(fcalc).detach().cpu().numpy(), alpha=0.5)
plt.plot([0,500],[0,500], color='red')
plt.xlim(0,500)
plt.ylim(0,500)
plt.xlabel('Observed |F|')
plt.ylabel('Calculated |F| after Scaling')
plt.title('Observed vs Calculated Structure Factors after Scaling Fit')
plt.savefig(f'{outdir}/observed_vs_calculated.png')
plt.close()

fobsmean, fcalcmean, res = S.get_binwise_mean_intensity()
plt.plot(res.detach().cpu().numpy(), fobsmean.detach().cpu().numpy(), label='Observed Mean |F|')
plt.plot(res.detach().cpu().numpy(), fcalcmean.detach().cpu().numpy(), label='Calculated Mean |F|')
plt.xlabel('Resolution (Å)')
plt.ylabel('Mean |F|')
plt.title('Bin-wise Mean Structure Factor Amplitudes after Scaling Fit')
plt.legend()
plt.savefig(f'{outdir}/mean_structure_factors.png')
plt.close()

res, work, rfree = S.bin_wise_rfactor()



plt.plot(res.detach().cpu().numpy(), work.detach().cpu().numpy(), label='R-work')
plt.plot(res.detach().cpu().numpy(), rfree.detach().cpu().numpy(), label='R-free')
plt.xlabel('Resolution (Å)')
plt.ylabel('R-factor')
plt.title('Bin-wise R-factors after Scaling Fit')
plt.legend()
plt.savefig(f'{outdir}/bin_wise_rfactors_post_fit.png')
plt.close()

print("\n\n" + "="*80)
print("Testing anisotropy-only fitting now...")
print("\n\n" + "="*80)