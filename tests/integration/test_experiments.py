"""
Integration tests for AKCB experiments
"""

import pytest
import torch
import os
import json
import tempfile
from pathlib import Path


class TestExperimentSetup:
    """Test experiment configuration and setup"""
    
    def test_config_files_exist(self):
        """Test that required configuration files exist"""
        config_dir = Path("/var/scratch/jle385/AKCB/experiments/longbench/config")
        
        assert (config_dir / "dataset2prompt.json").exists()
        assert (config_dir / "dataset2maxlen.json").exists()
        assert (config_dir / "model2path.json").exists()
        assert (config_dir / "model2maxlen.json").exists()
    
    def test_config_files_valid_json(self):
        """Test that configuration files are valid JSON"""
        config_dir = Path("/var/scratch/jle385/AKCB/experiments/longbench/config")
        
        for json_file in config_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert isinstance(data, dict)


class TestTelemetry:
    """Test telemetry tracking"""
    
    def test_telemetry_initialization(self):
        """Test telemetry object creation"""
        from experiments.longbench.pred import Telemetry
        
        device = torch.device('cpu')
        tele = Telemetry(device)
        
        assert tele.prompt_tokens == 0
        assert tele.gen_tokens == 0
        assert tele.wall_gen_s == 0.0
        assert len(tele.mem_alloc) == 0
        assert len(tele.ttft_ms) == 0
    
    def test_telemetry_add_tokens(self):
        """Test adding token counts"""
        from experiments.longbench.pred import Telemetry
        
        device = torch.device('cpu')
        tele = Telemetry(device)
        
        tele.add_prompt(100)
        tele.add_gen(50)
        
        assert tele.prompt_tokens == 100
        assert tele.gen_tokens == 50
    
    def test_telemetry_summary(self):
        """Test generating summary statistics"""
        from experiments.longbench.pred import Telemetry
        
        device = torch.device('cpu')
        tele = Telemetry(device)
        
        tele.add_prompt(100)
        tele.add_gen(50)
        tele.add_time(1.5)
        tele.add_ttft(23.5)
        
        summary = tele.summary()
        
        assert 'gen_tokens_per_s' in summary
        assert 'effective_tokens_per_s' in summary
        assert 'ttft_ms' in summary
        assert summary['gen_tokens'] == 50
        assert summary['prompt_tokens'] == 100


class TestFirstTokenTimer:
    """Test first token timing"""
    
    def test_timer_initialization(self):
        """Test timer initialization"""
        from experiments.longbench.pred import FirstTokenTimer
        
        start_len = 100
        timer = FirstTokenTimer(start_len)
        
        assert timer.start_len == start_len
        assert timer.t0 is None
        assert timer.first_token_ms is None
    
    def test_timer_call(self):
        """Test timer callback"""
        from experiments.longbench.pred import FirstTokenTimer
        
        start_len = 10
        timer = FirstTokenTimer(start_len)
        timer.start()
        
        # Simulate generation
        input_ids = torch.randint(0, 1000, (1, start_len + 1))
        scores = torch.randn(1, 1000)
        
        result = timer(input_ids, scores)
        
        assert result == False  # Should not stop generation
        assert timer.first_token_ms is not None


class TestCacheConfigIntegration:
    """Test cache configuration integration"""
    
    def test_cache_config_from_args(self):
        """Test creating cache config from arguments"""
        from experiments.longbench.pred import init_cache_config
        
        class Args:
            model = "Meta-Llama-3.1-8B-Instruct"
            cache_size = 1024
            window_size = 32
            tau1 = 0.1
            tau2 = 0.3
            tau3 = 0.6
            gamma = 1.0
            quant_type = "mix"
            compress = True
        
        config = init_cache_config(Args())
        
        assert config.cache_size == 1024
        assert config.window_size == 32
        assert config.quant_type == "mix"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
