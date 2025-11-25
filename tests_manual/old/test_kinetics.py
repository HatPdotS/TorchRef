"""
Unit tests for the KineticModel module.
"""

import torch
import numpy as np
import pytest
from torchref.kinetics import KineticModel


class TestFlowChartParsing:
    """Test flow chart parsing functionality."""
    
    def test_simple_sequential(self):
        """Test parsing simple sequential scheme A->B->C"""
        model = KineticModel(
            flow_chart="A->B->C",
            timepoints=[0, 1, 2],
            verbose=0
        )
        assert model.states == ['A', 'B', 'C']
        assert model.transitions == [('A', 'B'), ('B', 'C')]
        assert model.n_states == 3
        assert model.n_rates == 2
    
    def test_with_back_reaction(self):
        """Test parsing scheme with back reaction"""
        model = KineticModel(
            flow_chart="A->B->A",
            timepoints=[0, 1, 2],
            verbose=0
        )
        assert set(model.states) == {'A', 'B'}
        assert len(model.transitions) == 2
        assert ('A', 'B') in model.transitions
        assert ('B', 'A') in model.transitions
    
    def test_complex_scheme(self):
        """Test parsing complex scheme with multiple transitions"""
        model = KineticModel(
            flow_chart="A->B->C&B->A&C->A",
            timepoints=[0, 1, 2],
            verbose=0
        )
        assert set(model.states) == {'A', 'B', 'C'}
        assert len(model.transitions) == 4
    
    def test_parallel_pathways(self):
        """Test parsing parallel pathways"""
        model = KineticModel(
            flow_chart="A->B&A->C",
            timepoints=[0, 1, 2],
            verbose=0
        )
        assert set(model.states) == {'A', 'B', 'C'}
        assert ('A', 'B') in model.transitions
        assert ('A', 'C') in model.transitions


class TestRateMatrix:
    """Test rate matrix construction."""
    
    def test_rate_matrix_shape(self):
        """Test that rate matrix has correct shape"""
        model = KineticModel(
            flow_chart="A->B->C",
            timepoints=[0, 1],
            verbose=0
        )
        rates = torch.tensor([1.0, 2.0])
        K = model._build_rate_matrix(rates)
        assert K.shape == (3, 3)
    
    def test_rate_matrix_conservation(self):
        """Test that columns of rate matrix sum to zero"""
        model = KineticModel(
            flow_chart="A->B->C",
            timepoints=[0, 1],
            verbose=0
        )
        rates = torch.tensor([1.0, 2.0])
        K = model._build_rate_matrix(rates)
        
        # Each column should sum to zero (conservation)
        col_sums = torch.sum(K, dim=0)
        assert torch.allclose(col_sums, torch.zeros(3), atol=1e-6)
    
    def test_rate_matrix_simple_case(self):
        """Test rate matrix for simple A->B case"""
        model = KineticModel(
            flow_chart="A->B",
            timepoints=[0, 1],
            verbose=0
        )
        rate = torch.tensor([2.0])
        K = model._build_rate_matrix(rate)
        
        # Expected: K = [[-2, 0], [2, 0]]
        expected = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
        assert torch.allclose(K, expected)


class TestKineticSolution:
    """Test kinetic equation solving."""
    
    def test_initial_population(self):
        """Test that initial population is correct"""
        model = KineticModel(
            flow_chart="A->B",
            timepoints=[0.0],
            instrument_function='none',
            verbose=0
        )
        populations = model()
        
        # At t=0, should have all population in state A
        assert populations[0, 0] > 0.99  # State A
        assert populations[0, 1] < 0.01  # State B
    
    def test_conservation_of_population(self):
        """Test that total population is conserved"""
        model = KineticModel(
            flow_chart="A->B->C",
            timepoints=torch.linspace(0, 10, 100),
            instrument_function='none',
            verbose=0
        )
        populations = model()
        
        # Total population should always be 1
        total_pop = torch.sum(populations, dim=1)
        assert torch.allclose(total_pop, torch.ones(100), atol=1e-4)
    
    def test_final_state_sequential(self):
        """Test that final state is reached in sequential kinetics"""
        model = KineticModel(
            flow_chart="A->B->C",
            timepoints=torch.tensor([0.0, 100.0]),
            instrument_function='none',
            verbose=0
        )
        
        # Set fast rates
        with torch.no_grad():
            model.log_rate_constants[:] = torch.log(torch.tensor([10.0, 10.0]))
        
        populations = model()
        
        # At t=100, should have all population in state C
        assert populations[-1, 2] > 0.95  # State C


class TestInstrumentFunction:
    """Test instrument response function."""
    
    def test_no_instrument_function(self):
        """Test that 'none' instrument function doesn't change populations"""
        model = KineticModel(
            flow_chart="A->B",
            timepoints=[0, 1, 2],
            instrument_function='none',
            verbose=0
        )
        
        # Should work without errors
        populations = model()
        assert populations.shape == (3, 2)
    
    def test_gaussian_instrument_function(self):
        """Test Gaussian instrument function"""
        model = KineticModel(
            flow_chart="A->B",
            timepoints=torch.linspace(-2, 5, 100),
            instrument_function='gaussian',
            instrument_width=0.5,
            verbose=0
        )
        
        # Should work without errors
        populations = model()
        assert populations.shape == (100, 2)
        
        # Check that total population is still conserved (approximately)
        total_pop = torch.sum(populations, dim=1)
        assert torch.allclose(total_pop, torch.ones(100), atol=0.1)


class TestParameterAccess:
    """Test parameter access methods."""
    
    def test_get_rate_constants(self):
        """Test getting rate constants"""
        model = KineticModel(
            flow_chart="A->B->C",
            timepoints=[0, 1],
            verbose=0
        )
        
        rate_dict = model.get_rate_constants()
        assert 'A->B' in rate_dict
        assert 'B->C' in rate_dict
        assert len(rate_dict) == 2
    
    def test_get_time_constants(self):
        """Test getting time constants"""
        model = KineticModel(
            flow_chart="A->B",
            timepoints=[0, 1],
            verbose=0
        )
        
        # Set known rate
        with torch.no_grad():
            model.log_rate_constants[0] = np.log(2.0)
        
        time_dict = model.get_time_constants()
        assert 'A->B' in time_dict
        assert abs(time_dict['A->B'] - 0.5) < 1e-5  # tau = 1/k = 1/2 = 0.5


class TestGradients:
    """Test gradient computation for optimization."""
    
    def test_gradients_exist(self):
        """Test that gradients can be computed"""
        model = KineticModel(
            flow_chart="A->B",
            timepoints=[0, 1, 2],
            verbose=0
        )
        
        populations = model()
        loss = torch.sum(populations ** 2)
        loss.backward()
        
        # Check that gradients exist
        assert model.log_rate_constants.grad is not None
        assert not torch.isnan(model.log_rate_constants.grad).any()
    
    def test_optimization_step(self):
        """Test that parameters can be optimized"""
        model = KineticModel(
            flow_chart="A->B",
            timepoints=torch.linspace(0, 5, 50),
            verbose=0
        )
        
        # Create synthetic target
        target = torch.rand(50, 2)
        target = target / target.sum(dim=1, keepdim=True)  # Normalize
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        initial_loss = None
        for i in range(10):
            optimizer.zero_grad()
            populations = model()
            loss = torch.mean((populations - target) ** 2)
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        final_loss = loss.item()
        
        # Loss should decrease
        assert final_loss < initial_loss


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_state(self):
        """Test with single state (no transitions)"""
        # This should work but be trivial
        model = KineticModel(
            flow_chart="A",
            timepoints=[0, 1],
            verbose=0
        )
        assert model.n_states == 1
        assert model.n_rates == 0
    
    def test_custom_initial_state(self):
        """Test setting custom initial state"""
        model = KineticModel(
            flow_chart="A->B->C",
            timepoints=[0.0],
            initial_state='B',
            instrument_function='none',
            verbose=0
        )
        populations = model()
        
        # Should start in state B
        assert populations[0, 1] > 0.99  # State B


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
