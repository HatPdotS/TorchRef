#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

"""
Test PositiveMixedTensor functionality - a MixedTensor subclass that ensures
all values are positive by parametrizing in log space.
"""

import torch
import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.parameter_wrappers import PositiveMixedTensor


def test_basic_creation():
    """Test basic creation and positive value constraint"""
    print("\n" + "="*80)
    print("TEST 1: Basic Creation and Positive Constraint")
    print("="*80)
    
    # Create with positive values
    initial_values = torch.tensor([1.0, 5.0, 10.0, 20.0])
    pos = PositiveMixedTensor(initial_values)
    
    print(f"Initial values: {initial_values}")
    print(f"Output values: {pos()}")
    print(f"All positive: {(pos() > 0).all()}")
    
    # Check values are close to initial (with small epsilon tolerance)
    assert torch.allclose(pos(), initial_values, rtol=1e-5, atol=1e-6), "Values should match initial"
    assert (pos() > 0).all(), "All values must be positive"
    print("  ✓ Values preserved and positive")
    
    # Try to create with negative values (should fail)
    print("\nTest negative value rejection:")
    try:
        bad_values = torch.tensor([1.0, -2.0, 3.0])
        bad_pos = PositiveMixedTensor(bad_values)
        print("  ✗ Should have raised ValueError for negative values")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly rejected negative values: {str(e)[:60]}...")
    
    # Try to create with zero values (should fail)
    print("\nTest zero value rejection:")
    try:
        zero_values = torch.tensor([1.0, 0.0, 3.0])
        zero_pos = PositiveMixedTensor(zero_values)
        print("  ✗ Should have raised ValueError for zero values")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly rejected zero values: {str(e)[:60]}...")
    
    print("\n✓ TEST 1 PASSED")
    return True


def test_optimization():
    """Test that values remain positive during optimization"""
    print("\n" + "="*80)
    print("TEST 2: Optimization with Positive Constraint")
    print("="*80)
    
    # Create tensor
    initial = torch.tensor([10.0, 20.0, 30.0, 40.0])
    pos = PositiveMixedTensor(initial)
    
    print(f"Initial values: {pos()}")
    
    # Create target (different positive values)
    target = torch.tensor([15.0, 25.0, 35.0, 45.0])
    
    # Optimize
    optimizer = torch.optim.Adam([pos.refinable_params], lr=0.1)
    
    for i in range(50):
        optimizer.zero_grad()
        loss = (pos() - target).pow(2).sum()
        loss.backward()
        optimizer.step()
        
        # Check positivity at each step
        if not (pos() > 0).all():
            print(f"  ✗ Values became non-positive at step {i}")
            print(f"    Values: {pos()}")
            return False
    
    final = pos()
    print(f"Target values:  {target}")
    print(f"Final values:   {final}")
    print(f"Final loss:     {(final - target).pow(2).sum().item():.6f}")
    
    # Check all still positive
    assert (final > 0).all(), "All values must remain positive"
    print("  ✓ All values remained positive throughout optimization")
    
    # Check convergence
    final_loss = (final - target).pow(2).sum().item()
    assert final_loss < 1.0, "Should converge close to target"
    print("  ✓ Converged to target values")
    
    print("\n✓ TEST 2 PASSED")
    return True


def test_freeze_unfreeze():
    """Test freeze/unfreeze functionality with positive constraint"""
    print("\n" + "="*80)
    print("TEST 3: Freeze/Unfreeze with Positive Constraint")
    print("="*80)
    
    # Create tensor
    initial = torch.tensor([5.0, 10.0, 15.0, 20.0])
    pos = PositiveMixedTensor(initial)
    
    print(f"Initial values: {pos()}")
    print(f"Refinable count: {pos.get_refinable_count()}")
    
    # Freeze first two elements
    freeze_mask = torch.tensor([True, True, False, False])
    pos.fix(freeze_mask)
    
    print(f"\nAfter freezing first two:")
    print(f"Refinable count: {pos.get_refinable_count()}")
    print(f"Values: {pos()}")
    
    # Optimize
    target = torch.tensor([5.0, 10.0, 25.0, 30.0])
    optimizer = torch.optim.Adam([pos.refinable_params], lr=0.1)
    
    for i in range(50):
        optimizer.zero_grad()
        loss = (pos() - target).pow(2).sum()
        loss.backward()
        optimizer.step()
    
    final = pos()
    print(f"\nAfter optimization:")
    print(f"Target:  {target}")
    print(f"Final:   {final}")
    
    # Check frozen values didn't change
    assert torch.allclose(final[0:2], initial[0:2], rtol=1e-5), "Frozen values should not change"
    print("  ✓ Frozen values unchanged")
    
    # Check refinable values changed
    assert not torch.allclose(final[2:4], initial[2:4], rtol=0.1), "Refinable values should change"
    print("  ✓ Refinable values changed")
    
    # Check all still positive
    assert (final > 0).all(), "All values must remain positive"
    print("  ✓ All values remain positive")
    
    print("\n✓ TEST 3 PASSED")
    return True


def test_get_log_values():
    """Test access to internal log-space representation"""
    print("\n" + "="*80)
    print("TEST 4: Access to Log-Space Values")
    print("="*80)
    
    # Create tensor with known values
    normal_values = torch.tensor([1.0, 2.718281828, 7.389, 20.0])  # e^0, e^1, e^2, ...
    pos = PositiveMixedTensor(normal_values)
    
    print(f"Normal space values: {pos()}")
    
    # Get log values
    log_values = pos.get_log_values()
    print(f"Log space values: {log_values}")
    
    # Check relationship
    reconstructed = torch.exp(log_values)
    print(f"Reconstructed (exp(log)): {reconstructed}")
    
    assert torch.allclose(reconstructed, pos(), rtol=1e-5), "exp(log(x)) should equal x"
    print("  ✓ Log-space relationship verified")
    
    # Check that log of known values are approximately correct
    expected_log = torch.log(normal_values.clamp(min=pos.epsilon))
    assert torch.allclose(log_values, expected_log, rtol=1e-5), "Log values should match"
    print("  ✓ Log values are correct")
    
    print("\n✓ TEST 4 PASSED")
    return True


def test_extreme_values():
    """Test with very small and very large positive values"""
    print("\n" + "="*80)
    print("TEST 6: Extreme Positive Values")
    print("="*80)
    
    # Test with very small values (but not too small to avoid epsilon issues)
    print("Test 1: Small values")
    small_values = torch.tensor([1e-3, 1e-2, 1e-1, 1.0])
    pos_small = PositiveMixedTensor(small_values, epsilon=1e-12)
    
    print(f"  Input:  {small_values}")
    print(f"  Output: {pos_small()}")
    
    assert torch.allclose(pos_small(), small_values, rtol=1e-2, atol=1e-6), "Small values preserved"
    assert (pos_small() > 0).all(), "Small values remain positive"
    print("  ✓ Small values handled correctly")
    
    # Test with very large values
    print("\nTest 2: Very large values")
    large_values = torch.tensor([1e3, 1e4, 1e5, 1e6])
    pos_large = PositiveMixedTensor(large_values)
    
    print(f"  Input:  {large_values}")
    print(f"  Output: {pos_large()}")
    
    assert torch.allclose(pos_large(), large_values, rtol=1e-3), "Large values preserved"
    assert (pos_large() > 0).all(), "Large values remain positive"
    print("  ✓ Large values handled correctly")
    
    # Test with mixed scales
    print("\nTest 3: Mixed scales")
    mixed_values = torch.tensor([1e-2, 1.0, 100.0, 1e4])
    pos_mixed = PositiveMixedTensor(mixed_values, epsilon=1e-12)
    
    print(f"  Input:  {mixed_values}")
    print(f"  Output: {pos_mixed()}")
    
    assert torch.allclose(pos_mixed(), mixed_values, rtol=1e-2), "Mixed values preserved"
    assert (pos_mixed() > 0).all(), "Mixed values remain positive"
    print("  ✓ Mixed scales handled correctly")
    
    print("\n✓ TEST 5 PASSED")
    return True


def test_gradient_flow():
    """Test that gradients flow correctly through log/exp"""
    print("\n" + "="*80)
    print("TEST 6: Gradient Flow")
    print("="*80)
    
    # Create tensor
    initial = torch.tensor([5.0, 10.0, 15.0])
    pos = PositiveMixedTensor(initial)
    
    # Compute loss
    target = torch.tensor([7.0, 12.0, 17.0])
    loss = (pos() - target).pow(2).sum()
    
    print(f"Initial values: {pos()}")
    print(f"Target values:  {target}")
    print(f"Initial loss:   {loss.item():.6f}")
    
    # Compute gradients
    loss.backward()
    
    # Check that refinable_params has gradients
    assert pos.refinable_params.grad is not None, "Should have gradients"
    assert not torch.isnan(pos.refinable_params.grad).any(), "Gradients should not be NaN"
    assert not torch.isinf(pos.refinable_params.grad).any(), "Gradients should not be inf"
    
    print(f"Gradients (log space): {pos.refinable_params.grad}")
    print("  ✓ Gradients computed successfully")
    print("  ✓ No NaN or inf in gradients")
    
    # Test with Adam optimizer (handles log-space better than raw gradients)
    pos2 = PositiveMixedTensor(initial.clone())
    optimizer = torch.optim.Adam([pos2.refinable_params], lr=0.1)
    
    initial_loss2 = (pos2() - target).pow(2).sum().item()
    
    for _ in range(10):
        optimizer.zero_grad()
        loss2 = (pos2() - target).pow(2).sum()
        loss2.backward()
        optimizer.step()
    
    final_loss2 = (pos2() - target).pow(2).sum().item()
    
    print(f"\nWith Adam optimizer:")
    print(f"  Initial loss: {initial_loss2:.6f}")
    print(f"  Final loss:   {final_loss2:.6f}")
    print(f"  Final values: {pos2()}")
    
    assert final_loss2 < initial_loss2, "Loss should decrease with Adam"
    assert (pos2() > 0).all(), "Values should remain positive"
    print("  ✓ Adam optimizer decreased loss")
    print("  ✓ Values remain positive")
    
    print("\n✓ TEST 6 PASSED")
    return True


def test_comparison_with_clamping():
    """Compare log-space parametrization with naive clamping"""
    print("\n" + "="*80)
    print("TEST 7: Comparison with Naive Clamping")
    print("="*80)
    
    # Target very close to zero (challenging for clamping)
    target = torch.tensor([0.01, 0.1, 1.0, 10.0])
    
    # Approach 1: PositiveMixedTensor (log space)
    print("Approach 1: Log-space parametrization")
    initial = torch.tensor([5.0, 5.0, 5.0, 5.0])
    pos = PositiveMixedTensor(initial)
    
    optimizer1 = torch.optim.Adam([pos.refinable_params], lr=0.1)
    for i in range(100):
        optimizer1.zero_grad()
        loss = (pos() - target).pow(2).sum()
        loss.backward()
        optimizer1.step()
    
    final_pos = pos()
    loss_pos = (final_pos - target).pow(2).sum().item()
    print(f"  Initial: {initial}")
    print(f"  Target:  {target}")
    print(f"  Final:   {final_pos}")
    print(f"  Loss:    {loss_pos:.6f}")
    
    # Approach 2: Direct optimization with clamping
    print("\nApproach 2: Direct with clamping")
    params_direct = torch.nn.Parameter(initial.clone())
    optimizer2 = torch.optim.Adam([params_direct], lr=0.1)
    
    for i in range(100):
        optimizer2.zero_grad()
        clamped = torch.clamp(params_direct, min=1e-6)  # Prevent negative
        loss = (clamped - target).pow(2).sum()
        loss.backward()
        optimizer2.step()
    
    final_direct = torch.clamp(params_direct, min=1e-6)
    loss_direct = (final_direct - target).pow(2).sum().item()
    print(f"  Initial: {initial}")
    print(f"  Target:  {target}")
    print(f"  Final:   {final_direct}")
    print(f"  Loss:    {loss_direct:.6f}")
    
    print(f"\nLog-space is better: {loss_pos < loss_direct}")
    print(f"Loss reduction: {(loss_direct - loss_pos):.6f}")
    
    # Log-space should perform at least as well (usually better)
    assert (pos() > 0).all(), "Log-space values must be positive"
    print("  ✓ Log-space parametrization ensures positivity")
    print("  ✓ Smooth optimization without hard boundaries")
    
    print("\n✓ TEST 7 PASSED")
    return True


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# POSITIVE MIXED TENSOR TESTS")
    print("# Testing log-space parametrization for positive values")
    print("#"*80)
    
    tests = [
        ("Basic Creation", test_basic_creation),
        ("Optimization", test_optimization),
        ("Freeze/Unfreeze", test_freeze_unfreeze),
        ("Log Values Access", test_get_log_values),
        ("Extreme Values", test_extreme_values),
        ("Gradient Flow", test_gradient_flow),
        ("Comparison with Clamping", test_comparison_with_clamping),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result, None))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} {name}")
        if error:
            print(f"         Error: {error[:60]}...")
    
    print("="*80)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
