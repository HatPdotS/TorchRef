#!/usr/bin/env python
"""
Quick test script to verify the KineticModel implementation.
Run this to ensure everything is working correctly.
"""

import torch
import numpy as np
import sys

def test_basic_functionality():
    """Test basic model creation and forward pass."""
    print("="*60)
    print("TEST 1: Basic Functionality")
    print("="*60)
    
    try:
        from torchref.kinetics import KineticModel
        
        # Create simple model with new comma syntax
        t = torch.linspace(0, 5, 50)
        model = KineticModel(
            flow_chart="A->B,B->C",
            timepoints=t,
            instrument_function='none',
            verbose=0
        )
        
        # Forward pass
        populations = model()
        
        # Check shape
        assert populations.shape == (50, 3), f"Expected shape (50, 3), got {populations.shape}"
        
        # Check conservation
        total_pop = torch.sum(populations, dim=1)
        assert torch.allclose(total_pop, torch.ones(50), atol=1e-4), "Population not conserved!"
        
        print("✓ Model creation successful")
        print("✓ Forward pass successful")
        print("✓ Population conservation verified")
        print("✓ TEST 1 PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ TEST 1 FAILED: {e}\n")
        return False


def test_flow_chart_parsing():
    """Test flow chart parsing for different schemes."""
    print("="*60)
    print("TEST 2: Flow Chart Parsing (New Comma Syntax)")
    print("="*60)
    
    try:
        from torchref.kinetics import KineticModel
        
        test_cases = [
            ("A->B,B->C", 3, 2),
            ("A->B,B->A", 2, 2),
            ("A->B,A->C", 3, 2),
            ("A->B,B->C,B->A", 3, 3),
            ("A->B,B->C,C->D,C->A", 4, 4),
        ]
        
        for flow_chart, expected_states, expected_transitions in test_cases:
            model = KineticModel(
                flow_chart=flow_chart,
                timepoints=[0, 1],
                verbose=0
            )
            
            assert model.n_states == expected_states, \
                f"Flow chart '{flow_chart}': expected {expected_states} states, got {model.n_states}"
            assert len(model.transitions) == expected_transitions, \
                f"Flow chart '{flow_chart}': expected {expected_transitions} transitions, got {len(model.transitions)}"
            
            print(f"✓ '{flow_chart}' -> {model.n_states} states, {len(model.transitions)} transitions")
        
        print("✓ TEST 2 PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ TEST 2 FAILED: {e}\n")
        return False


def test_gradients():
    """Test gradient computation."""
    print("="*60)
    print("TEST 3: Gradient Computation")
    print("="*60)
    
    try:
        from torchref.kinetics import KineticModel
        
        model = KineticModel(
            flow_chart="A->B",
            timepoints=torch.linspace(0, 5, 20),
            instrument_function='none',
            verbose=0
        )
        
        # Forward pass
        populations = model()
        loss = torch.sum(populations ** 2)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist for all parameters
        assert model.log_rate_constants.grad is not None, "No gradients for rate constants!"
        assert model.logit_efficiencies.grad is not None, "No gradients for efficiencies!"
        assert not torch.isnan(model.log_rate_constants.grad).any(), "NaN in rate gradients!"
        assert not torch.isnan(model.logit_efficiencies.grad).any(), "NaN in efficiency gradients!"
        
        print("✓ Gradients computed for rate constants")
        print("✓ Gradients computed for efficiencies")
        print("✓ No NaN values in gradients")
        print("✓ TEST 3 PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ TEST 3 FAILED: {e}\n")
        return False


def test_parameter_access():
    """Test parameter access methods."""
    print("="*60)
    print("TEST 4: Parameter Access & Efficiencies")
    print("="*60)
    
    try:
        from torchref.kinetics import KineticModel
        
        model = KineticModel(
            flow_chart="A->B,B->C",
            timepoints=[0, 1],
            verbose=0
        )
        
        # Set known rates and efficiencies
        with torch.no_grad():
            model.log_rate_constants[0] = np.log(2.0)
            model.log_rate_constants[1] = np.log(0.5)
            # Set efficiencies to 0.8 (logit(0.8) ≈ 1.386)
            model.logit_efficiencies[0] = 1.386
            model.logit_efficiencies[1] = 1.386
        
        # Get rate constants
        rate_dict = model.get_rate_constants()
        assert 'A->B' in rate_dict, "Missing A->B in rate dictionary"
        assert 'B->C' in rate_dict, "Missing B->C in rate dictionary"
        assert abs(rate_dict['A->B'] - 2.0) < 1e-5, f"Expected rate 2.0, got {rate_dict['A->B']}"
        assert abs(rate_dict['B->C'] - 0.5) < 1e-5, f"Expected rate 0.5, got {rate_dict['B->C']}"
        
        # Get efficiencies
        eff_dict = model.get_efficiencies()
        assert 'A->B' in eff_dict, "Missing A->B in efficiency dictionary"
        assert abs(eff_dict['A->B'] - 0.8) < 0.01, f"Expected efficiency ~0.8, got {eff_dict['A->B']}"
        
        # Get effective rates
        eff_rate_dict = model.get_effective_rates()
        expected_eff_rate = 2.0 * 0.8
        assert abs(eff_rate_dict['A->B'] - expected_eff_rate) < 0.05, \
            f"Expected effective rate ~{expected_eff_rate}, got {eff_rate_dict['A->B']}"
        
        # Get all tensors
        tensors = model.get_all_tensors()
        assert len(tensors) == 3, f"Expected 3 tensors, got {len(tensors)}"
        
        print("✓ Rate constants accessible")
        print("✓ Efficiencies accessible")
        print("✓ Effective rates computed correctly")
        print("✓ All tensors retrievable")
        print("✓ TEST 4 PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ TEST 4 FAILED: {e}\n")
        return False


def test_instrument_function():
    """Test instrument response function."""
    print("="*60)
    print("TEST 5: Instrument Function & Visualization")
    print("="*60)
    
    try:
        from torchref.kinetics import KineticModel
        import os
        
        t = torch.linspace(-2, 5, 100)
        
        # Test with Gaussian IRF
        model = KineticModel(
            flow_chart="A->B",
            timepoints=t,
            instrument_function='gaussian',
            instrument_width=0.5,
            verbose=0
        )
        
        populations = model()
        
        # Check that it runs without error
        assert populations.shape == (100, 2), f"Unexpected shape: {populations.shape}"
        
        # Check approximate conservation (IRF can slightly affect this)
        total_pop = torch.sum(populations, dim=1)
        assert torch.allclose(total_pop, torch.ones(100), atol=0.15), \
            "Population not approximately conserved with IRF"
        
        # Test visualization function
        test_plot = '/tmp/test_kinetics_plot.png'
        model.plot_occupancies(test_plot, log=False)
        assert os.path.exists(test_plot), "Plot file not created"
        os.remove(test_plot)  # Clean up
        
        print("✓ Gaussian instrument function works")
        print("✓ Population approximately conserved")
        print("✓ Visualization function works")
        print("✓ TEST 5 PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ TEST 5 FAILED: {e}\n")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("KINETIC MODEL VERIFICATION TESTS")
    print("="*60 + "\n")
    
    results = []
    
    results.append(("Basic Functionality", test_basic_functionality()))
    results.append(("Flow Chart Parsing", test_flow_chart_parsing()))
    results.append(("Gradient Computation", test_gradients()))
    results.append(("Parameter Access", test_parameter_access()))
    results.append(("Instrument Function", test_instrument_function()))
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("="*60)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        print("The KineticModel implementation is working correctly.")
    else:
        print("SOME TESTS FAILED! ✗")
        print("Please check the error messages above.")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
