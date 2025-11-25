#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u
#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/Dark_LBFGS/new_french_wilson.out

from torchref.model_ft import ModelFT
from torchref.scaler import Scaler
from torchref.Data import ReflectionData
import torch
import matplotlib.pyplot as plt
import os
data = ReflectionData(verbose=2).load_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.mtz')

M = ModelFT(verbose=0,max_res=1.7).load_pdb('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.pdb')
data.filter_by_resolution(d_min=1.7)


S = Scaler(M, data, nbins=10,verbose=2)

S.initialize()

S.screen_solvent_params(steps=20)

outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/Dark_LBFGS/new_french_wilson'

os.makedirs(outdir, exist_ok=True)



rwork, rfree = S.rfactor()
print(f"Initial R-work: {rwork:.4f}, R-free: {rfree:.4f}")

fcalc = S(M(data.get_hkl()))
_, fobs, _, _ = data()

data.find_outliers(M, S,z_threshold=5.0)


plt.scatter(torch.abs(fobs).cpu().numpy(), torch.abs(fcalc).detach().cpu().numpy(), alpha=0.5)
plt.plot([0,500],[0,500], color='red')
plt.xlim(0,500)
plt.ylim(0,500)
plt.xlabel('Observed |F|')
plt.ylabel('Calculated |F| before Scaling Fit')
plt.title('Observed vs Calculated Structure Factors before Scaling Fit')
plt.savefig(f'{outdir}/observed_vs_calculated_before_fit.png')
plt.close()

max_res, rwork, rfree = S.bin_wise_rfactor()

plt.plot(max_res.cpu().numpy(), rwork.cpu().numpy(), label='R-work')
plt.plot(max_res.cpu().numpy(), rfree.cpu().numpy(), label='R-free')
plt.xlabel('Resolution (Å)')
plt.ylabel('R-factor')
plt.title('Bin-wise R-factors before Scaling Fit')
plt.legend()
plt.savefig(f'{outdir}/bin_wise_rfactors.png')

plt.close()

S.verbose=10
S.refine_lbfgs()
S.verbose=0

rwork, rfree = S.rfactor()
print(f"Post-fitting R-work: {rwork:.4f}, R-free: {rfree:.4f}")

data.find_outliers(M, S,z_threshold=5.0)

rwork, rfree = S.rfactor()
print(f"Post-outlier R-work: {rwork:.4f}, R-free: {rfree:.4f}")


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


# plot ratios of mean F and mean Fcalc


fcalc = S(M(data.get_hkl()))
_, fobs, _, _ = data()

ratios = torch.log(torch.clamp(fobs.abs(), min=1e-6)) - torch.log(torch.clamp(fcalc.abs(), min=1e-6))

ratios = (ratios - ratios.mean() ) / ratios.std()

print('Outlier fraction for zscore 4:',(torch.sum(torch.abs(ratios)>4.0) / len(ratios)).item())

plt.hist(ratios.detach().cpu().numpy(), bins=100, alpha=0.7, edgecolor='black')
plt.xlabel('log(Observed |F|) - log(Calculated |F|)')
plt.ylabel('Count')
plt.title('Histogram of log |F| Ratios after Scaling Fit')
plt.vlines(x=[-4,4], ymin=0,ymax=2000, color='red', linestyle='--')
plt.savefig(f'{outdir}/log_f_ratio_histogram.png')

fcalc = M(data.get_hkl())

_, fobs, _, _ = data()

ratios = torch.log(torch.clamp(fobs.abs(), min=1e-6)) - torch.log(torch.clamp(fcalc.abs(), min=1e-6))

ratios = (ratios - ratios.mean() ) / ratios.std()

print('Outlier fraction for zscore 4 no fit:',(torch.sum(torch.abs(ratios)>4.0) / len(ratios)).item())

plt.hist(ratios.detach().cpu().numpy(), bins=100, alpha=0.7, edgecolor='black')
plt.xlabel('log(Observed |F|) - log(Calculated |F|)')
plt.ylabel('Count')
plt.title('Histogram of log |F| Ratios no Scaling Fit')
plt.vlines(x=[-4,4], ymin=0,ymax=2000, color='red', linestyle='--')
plt.savefig(f'{outdir}/log_f_ratio_histogram_no_corrections.png')
plt.close()


print(fobs.shape)

ratios = data.get_log_ratio(M, S)

ratios = ratios[~torch.isnan(ratios)]

ratios = (ratios - ratios.mean() ) / ratios.std()

print('Outlier fraction for zscore 4 no fit:',(torch.sum(torch.abs(ratios)>4.0) / len(ratios)).item())

plt.hist(ratios.detach().cpu().numpy(), bins=100, alpha=0.7, edgecolor='black')
plt.xlabel('log(Observed |F|) - log(Calculated |F|)')
plt.ylabel('Count')
plt.title('Histogram of log |F| Ratios')
plt.vlines(x=[-4,4], ymin=0,ymax=2000, color='red', linestyle='--')
plt.savefig(f'{outdir}/log_f_ratio_histogram_from_data_method.png')