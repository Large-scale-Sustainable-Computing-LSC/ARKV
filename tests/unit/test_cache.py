"""
Unit tests for AKCB cache implementations
"""

import pytest
import torch
from akcb.config import ADCacheConfig
from akcb.cache.adaptive_cache import AdaptiveCache


class TestADCacheConfig:
    """Test configuration class"""
    
    def test_basic_config(self):
        """Test basic configuration creation"""
        class Args:
            cache_size = 4096
            window_size = 128
            tau1 = 0.1
            tau2 = 0.3
            tau3 = 0.6
            gamma = 1.0
            quant_type = "mix"
            compress = True
        
        config = ADCacheConfig(Args())
        
        assert config.cache_size == 4096
        assert config.window_size == 128
        assert config.key_size == 4096 - 128
        assert config.quant_type == "mix"
        assert config.compress == True
    
    def test_layer_num_update(self):
        """Test updating layer numbers"""
        class Args:
            cache_size = 2048
            window_size = 64
            tau1 = 0.1
            tau2 = 0.3
            tau3 = 0.6
            gamma = 1.0
            quant_type = "quant"
            compress = True
        
        config = ADCacheConfig(Args())
        layer_num = 32
        config.update_layer_num(layer_num)
        
        assert config.layer_num == layer_num
        assert len(config.decoding_compressor) == layer_num
        assert len(config.prefill) == layer_num
        assert len(config.quant_ratio_per_layer) == layer_num


class TestAdaptiveCache:
    """Test adaptive cache factory"""
    
    def test_mix_cache_creation(self):
        """Test creating mixed-precision cache"""
        class Args:
            cache_size = 1024
            window_size = 32
            tau1 = 0.1
            tau2 = 0.3
            tau3 = 0.6
            gamma = 1.0
            quant_type = "mix"
            compress = True
        
        config = ADCacheConfig(Args())
        adaptive_cache = AdaptiveCache(config)
        
        assert adaptive_cache.cache is not None
        assert adaptive_cache.prefill_compressor is not None
        assert adaptive_cache.decoding_compressor is not None
    
    def test_quant_cache_creation(self):
        """Test creating quantization cache"""
        class Args:
            cache_size = 1024
            window_size = 32
            tau1 = 0.1
            tau2 = 0.3
            tau3 = 0.6
            gamma = 1.0
            quant_type = "quant"
            compress = True
        
        config = ADCacheConfig(Args())
        adaptive_cache = AdaptiveCache(config)
        
        assert adaptive_cache.cache is not None
    
    def test_origin_cache_creation(self):
        """Test creating full-precision cache"""
        class Args:
            cache_size = 1024
            window_size = 32
            tau1 = 0.1
            tau2 = 0.3
            tau3 = 0.6
            gamma = 1.0
            quant_type = "origin"
            compress = True
        
        config = ADCacheConfig(Args())
        adaptive_cache = AdaptiveCache(config)
        
        assert adaptive_cache.cache is not None
    
    def test_window_cache_creation(self):
        """Test creating window cache"""
        class Args:
            cache_size = 1024
            window_size = 32
            tau1 = 0.1
            tau2 = 0.3
            tau3 = 0.6
            gamma = 1.0
            quant_type = "window"
            compress = True
        
        config = ADCacheConfig(Args())
        adaptive_cache = AdaptiveCache(config)
        
        assert adaptive_cache.cache is not None
    
    def test_invalid_cache_type(self):
        """Test creating cache with invalid type"""
        class Args:
            cache_size = 1024
            window_size = 32
            tau1 = 0.1
            tau2 = 0.3
            tau3 = 0.6
            gamma = 1.0
            quant_type = "invalid"
            compress = True
        
        config = ADCacheConfig(Args())
        
        with pytest.raises(ValueError):
            adaptive_cache = AdaptiveCache(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
