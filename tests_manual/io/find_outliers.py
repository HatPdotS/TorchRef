#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u 

#SBATCH -c 32
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/io/test.log

import torch
from torchref.model_ft import ModelFT
from torchref.Data import ReflectionData
from torchref.scaler import Scaler
from matplotlib import pyplot as plt
import numpy as np

Model = ModelFT().load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.pdb')
Data = ReflectionData(verbose=2).load_from_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz')

scaler = Scaler(Model(Data.get_hkl()),Data)

log_ratios_before = Data.get_log_ratio(Model, scaler)




log_ratios_np = log_ratios_before.detach().cpu().numpy()

plt.hist(log_ratios_np, bins=100, density=True, alpha=0.6, color='g')
plt.xlabel('Log Ratio (log(F_obs) - log(F_calc))')
plt.ylabel('Density')
plt.title('Histogram of Log Ratios')
plt.grid(True)
plt.savefig('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/io/log_ratio_histogram.png')
plt.close()

Data_cleanish = Data

for z in [3.5]:
    scaler_inbetween = Scaler(Model(Data_cleanish.get_hkl()),Data_cleanish,verbose=0,nbins=50)
    Data_cleanish = Data_cleanish.remove_outliers(Model, scaler_inbetween, z_threshold=z)

Data_clean = Data_cleanish



scaler_new = Scaler(Model(Data_clean.get_hkl()),Data_clean)

log_ratios_after = Data_clean.get_log_ratio(Model, scaler_new)

log_ratios_np_after = log_ratios_after.detach().cpu().numpy()
plt.hist(log_ratios_np_after, bins=100, density=True, alpha=0.6, color='b')
plt.xlabel('Log Ratio (log(F_obs) - log(F_calc))')
plt.ylabel('Density')
plt.title('Histogram of Log Ratios After Outlier Removal')
plt.grid(True)
plt.savefig('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/io/log_ratio_histogram_no_outliers.png')
plt.close()

def rfactor(F_obs, F_calc):
    """Calculate R-factor between observed and calculated structure factors."""
    return torch.sum(torch.abs(F_obs - F_calc)) / torch.sum(F_obs)

# With outliers 

Fcalc = torch.abs(scaler(Model(Data.get_hkl())))
_, Fobs, _,_ = Data()
rf_before = rfactor(Fobs, Fcalc)


Fcalc = torch.abs(scaler_new(Model(Data_clean.get_hkl())))
_, Fobs, _,_ = Data_clean()
rf_after = rfactor(Fobs, Fcalc)
print("R-factors before and after outlier removal: \n" + '='*50)

print(f"R-factor before outlier removal: {rf_before.item():.4f}")
print(f"R-factor after outlier removal: {rf_after.item():.4f}")
print('='*50)



