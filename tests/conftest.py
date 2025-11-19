"""
Test configuration and fixtures
"""

import pytest
import torch


@pytest.fixture
def device():
    """Fixture to provide device for testing"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def small_attention_scores():
    """Fixture to provide small attention score tensor for testing"""
    batch_size = 2
    num_heads = 4
    query_len = 8
    key_len = 32
    
    # Create random attention scores (after softmax)
    attn = torch.rand(batch_size, num_heads, query_len, key_len)
    return torch.nn.functional.softmax(attn, dim=-1)


@pytest.fixture
def cache_config_args():
    """Fixture to provide cache configuration arguments"""
    class Args:
        cache_size = 1024
        window_size = 32
        tau1 = 0.1
        tau2 = 0.3
        tau3 = 0.6
        gamma = 1.0
        quant_type = "mix"
        compress = True
    return Args()


@pytest.fixture
def model_args():
    """Fixture to provide model arguments for testing"""
    class Args:
        model = "Meta-Llama-3.1-8B-Instruct"
        e = False
        compress = True
        cache_size = 1024
        window_size = 32
        tau1 = 0.1
        tau2 = 0.3
        tau3 = 0.6
        gamma = 1.0
        quant_type = "mix"
        outdir = "test_output"
    return Args()
