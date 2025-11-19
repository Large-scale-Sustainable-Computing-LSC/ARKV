"""
Unit tests for AKCB calculator module
"""

import pytest
import torch
import numpy as np
from akcb.calculator import calculate_heavy_hitter, calculate_entropy


class TestCalculateHeavyHitter:
    """Test heavy hitter score calculation"""
    
    def test_basic_calculation(self):
        """Test basic heavy hitter calculation"""
        batch_size = 2
        num_heads = 4
        query_len = 8
        key_len = 64
        window_size = 8
        gamma = 1.0
        
        # Create random attention scores (should be after softmax, sum to 1)
        attention_scores = torch.rand(batch_size, num_heads, query_len, key_len)
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        # Calculate heavy hitter scores
        hh_scores = calculate_heavy_hitter(
            attention_scores=attention_scores,
            gamma=gamma,
            window_size=window_size,
            bsz=batch_size,
            num_key_value_heads=num_heads,
            num_key_value_groups=1
        )
        
        # Check output shape
        expected_len = key_len - window_size
        assert hh_scores.shape == (batch_size, num_heads, expected_len)
        
        # Check values are non-negative
        assert torch.all(hh_scores >= 0)
    
    def test_empty_window(self):
        """Test when window size >= key length"""
        batch_size = 2
        num_heads = 4
        query_len = 8
        key_len = 8
        window_size = 8
        gamma = 1.0
        
        attention_scores = torch.rand(batch_size, num_heads, query_len, key_len)
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        hh_scores = calculate_heavy_hitter(
            attention_scores=attention_scores,
            gamma=gamma,
            window_size=window_size,
            bsz=batch_size,
            num_key_value_heads=num_heads,
            num_key_value_groups=1
        )
        
        # Should return empty tensor
        assert hh_scores.shape[-1] == 0
    
    def test_single_query(self):
        """Test with single query token"""
        batch_size = 1
        num_heads = 2
        query_len = 1
        key_len = 32
        window_size = 4
        gamma = 1.0
        
        attention_scores = torch.rand(batch_size, num_heads, query_len, key_len)
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        hh_scores = calculate_heavy_hitter(
            attention_scores=attention_scores,
            gamma=gamma,
            window_size=window_size,
            bsz=batch_size,
            num_key_value_heads=num_heads,
            num_key_value_groups=1
        )
        
        # With single query, no variance component
        assert hh_scores.shape == (batch_size, num_heads, key_len - window_size)
        assert torch.all(hh_scores >= 0)


class TestCalculateEntropy:
    """Test entropy calculation"""
    
    def test_uniform_distribution(self):
        """Test entropy for uniform distribution"""
        # Uniform distribution should have maximum entropy
        n = 10
        uniform_attn = torch.ones(n) / n
        entropy = calculate_entropy(uniform_attn)
        
        # Entropy of uniform distribution is log(n)
        expected_entropy = np.log(n)
        assert abs(entropy.item() - expected_entropy) < 0.01
    
    def test_deterministic_distribution(self):
        """Test entropy for deterministic distribution"""
        # One-hot distribution should have minimum entropy (close to 0)
        attn = torch.zeros(10)
        attn[0] = 1.0
        entropy = calculate_entropy(attn)
        
        # Should be very close to 0
        assert entropy.item() < 0.01
    
    def test_shape_handling(self):
        """Test entropy calculation with different shapes"""
        # 1D tensor
        attn_1d = torch.softmax(torch.rand(20), dim=0)
        entropy_1d = calculate_entropy(attn_1d)
        assert entropy_1d.dim() == 0  # Scalar
        
        # 2D tensor
        attn_2d = torch.softmax(torch.rand(5, 10), dim=-1)
        entropy_2d = calculate_entropy(attn_2d)
        assert entropy_2d.dim() == 0  # Scalar


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
