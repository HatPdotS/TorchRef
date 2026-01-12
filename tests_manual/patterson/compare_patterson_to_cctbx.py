from torchref.io import ReflectionData
import numpy as np
import gemmi


mtz_path = '/das/work/p17/p17490/Peter/Library/torchref/tests_manual/patterson/dark.mtz'

data = ReflectionData().load_mtz(mtz_path)




from iotbx import mtz
from cctbx import miller
from cctbx.array_family import flex

# Load data
mtz_obj = mtz.object(mtz_path)
miller_arrays = mtz_obj.as_miller_arrays()

# Get the F array - use F-obs-filtered (index 2) for amplitude data
# Note: miller_arrays[0] is I-obs (intensities), which is different from F values
# For fair comparison, use F-obs since torchref uses F values (from French-Wilson)
f_obs = miller_arrays[2]  # F-obs-filtered

# Calculate Patterson map directly
patterson_map = f_obs.patterson_map(
    symmetry_flags=None
)

# Access the map
patterson_data = patterson_map.real_map_unpadded()

# Get map as numpy array
import numpy as np
patterson_np = patterson_data.as_numpy_array()

print(f"Patterson shape: {patterson_np.shape}")
print(f"Origin value (should be max): {patterson_np[0, 0, 0]}")
print(f"Max value: {patterson_np.max()}")

patterson_map = data.calc_patterson(patterson_np.shape).detach().cpu().numpy()

print(f"Patterson shape: {patterson_map.shape}")

print(np.corrcoef(patterson_np.flatten(), patterson_map.flatten())[0,1])