#!/usr/bin/env python
"""
Test script to demonstrate intensity priority and French-Wilson conversion.
"""

from torchref.Data import ReflectionData
import reciprocalspaceship as rs
import numpy as np
import torch

print("="*80)
print("Testing Intensity Priority System and French-Wilson Conversion")
print("="*80)

# Create a test MTZ with both intensity and amplitude data
print("\n1. Creating test MTZ with both I and F data...")

# Generate some fake data
n_refl = 1000
h = np.random.randint(-10, 10, n_refl)
k = np.random.randint(-10, 10, n_refl)
l = np.random.randint(-10, 10, n_refl)

# Simulate intensities (can be negative due to noise)
I_true = np.random.exponential(100, n_refl)
I_noise = np.random.normal(0, 10, n_refl)
I_obs = I_true + I_noise
SIGI = np.ones(n_refl) * 10

# Simulate amplitudes from intensities
F_obs = np.sqrt(np.abs(I_obs))
SIGF = SIGI / (2 * np.sqrt(np.abs(I_obs) + 1e-6))

# Create dataset
dataset = rs.DataSet({
    'H': h,
    'K': k,
    'L': l,
    'I-obs': I_obs,
    'SIGI-obs': SIGI,
    'F-obs': F_obs,
    'SIGF-obs': SIGF,
}, cell=[50, 50, 50, 90, 90, 90], spacegroup='P1')

dataset.set_index(['H', 'K', 'L'], inplace=True)
dataset.infer_mtz_dtypes(inplace=True)

# Write to file
test_mtz = '/tmp/test_intensity_priority.mtz'
dataset.write_mtz(test_mtz)

print(f"  Created MTZ with {len(dataset)} reflections")
print(f"  Columns: {list(dataset.columns)}")
print(f"  Dtypes: {dict(dataset.dtypes)}")

# Test 1: Load with both I and F present (should prefer I)
print("\n" + "="*80)
print("Test 1: Both I-obs and F-obs present (should prefer I-obs)")
print("="*80)

data1 = ReflectionData()
data1.load_from_mtz(test_mtz)

assert data1.intensity_source == 'I-obs', f"Expected I-obs, got {data1.intensity_source}"
assert data1.amplitude_source == 'I-obs (French-Wilson)', f"Expected French-Wilson, got {data1.amplitude_source}"
assert data1.F is not None, "F should be populated"
assert data1.F_sigma is not None, "F_sigma should be populated"
assert data1.I is not None, "I should be stored"
assert data1.I_sigma is not None, "I_sigma should be stored"

print(f"✓ Correctly chose intensity: {data1.intensity_source}")
print(f"✓ Converted to amplitude: {data1.amplitude_source}")
print(f"✓ F range: {data1.F.min():.2f} - {data1.F.max():.2f}")
print(f"✓ F_sigma range: {data1.F_sigma.min():.2f} - {data1.F_sigma.max():.2f}")

# Test 2: Create MTZ with only F data
print("\n" + "="*80)
print("Test 2: Only F-obs present (should use F-obs directly)")
print("="*80)

dataset2 = rs.DataSet({
    'H': h,
    'K': k,
    'L': l,
    'F-obs': F_obs,
    'SIGF-obs': SIGF,
}, cell=[50, 50, 50, 90, 90, 90], spacegroup='P1')

dataset2.set_index(['H', 'K', 'L'], inplace=True)
dataset2.infer_mtz_dtypes(inplace=True)

test_mtz2 = '/tmp/test_amplitude_only.mtz'
dataset2.write_mtz(test_mtz2)

data2 = ReflectionData()
data2.load_from_mtz(test_mtz2)

assert data2.amplitude_source == 'F-obs', f"Expected F-obs, got {data2.amplitude_source}"
assert data2.intensity_source is None, "Should not have intensity source"
assert data2.F is not None, "F should be populated"
assert data2.F_sigma is not None, "F_sigma should be populated"

print(f"✓ Correctly used amplitude: {data2.amplitude_source}")
print(f"✓ No French-Wilson conversion needed")
print(f"✓ F range: {data2.F.min():.2f} - {data2.F.max():.2f}")

# Test 3: Verify French-Wilson handles negative intensities correctly
print("\n" + "="*80)
print("Test 3: French-Wilson with negative intensities")
print("="*80)

# Create data with some negative intensities
I_with_neg = I_obs.copy()
neg_mask = np.random.random(n_refl) < 0.1  # 10% negative
I_with_neg[neg_mask] = -np.abs(I_with_neg[neg_mask])

dataset3 = rs.DataSet({
    'H': h,
    'K': k,
    'L': l,
    'IOBS': I_with_neg,
    'SIGIOBS': SIGI,
}, cell=[50, 50, 50, 90, 90, 90], spacegroup='P1')

dataset3.set_index(['H', 'K', 'L'], inplace=True)
dataset3.infer_mtz_dtypes(inplace=True)

test_mtz3 = '/tmp/test_negative_intensities.mtz'
dataset3.write_mtz(test_mtz3)

data3 = ReflectionData()
data3.load_from_mtz(test_mtz3)

n_negative = (data3.I < 0).sum().item()
print(f"  Input: {n_negative} negative intensities")
print(f"  Output: All F values non-negative: {(data3.F >= 0).all().item()}")
print(f"  F range: {data3.F.min():.2f} - {data3.F.max():.2f}")

assert (data3.F >= 0).all(), "French-Wilson should produce non-negative amplitudes"
print(f"✓ French-Wilson correctly handled negative intensities")

# Test 4: Priority system with F-model vs I-obs
print("\n" + "="*80)
print("Test 4: Priority - I-obs (high priority) vs F-model (low priority)")
print("="*80)

dataset4 = rs.DataSet({
    'H': h,
    'K': k,
    'L': l,
    'I-obs': I_obs,
    'SIGI-obs': SIGI,
    'F-model': F_obs * 1.5,  # Different values
    'SIGF': SIGF * 1.5,
}, cell=[50, 50, 50, 90, 90, 90], spacegroup='P1')

dataset4.set_index(['H', 'K', 'L'], inplace=True)
dataset4.infer_mtz_dtypes(inplace=True)

test_mtz4 = '/tmp/test_priority.mtz'
dataset4.write_mtz(test_mtz4)

data4 = ReflectionData()
data4.load_from_mtz(test_mtz4)

assert data4.intensity_source == 'I-obs', "Should prefer I-obs over F-model"
assert data4.amplitude_source == 'I-obs (French-Wilson)', "Should convert from I-obs"
print(f"✓ Correctly prioritized I-obs over F-model")

print("\n" + "="*80)
print("All tests passed! ✓")
print("="*80)

print("\nSummary:")
print("  1. ✓ Intensities are preferred over amplitudes when both present")
print("  2. ✓ Sigma columns are automatically found and paired")
print("  3. ✓ French-Wilson conversion is applied automatically to intensities")
print("  4. ✓ Negative intensities are handled correctly")
print("  5. ✓ Priority system works as expected (observations > model)")
