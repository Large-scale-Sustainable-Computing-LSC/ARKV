# AKCB: Adaptive KV Caches under Budget

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

**AKCB (Adaptive KV Caches under Budget)** is a lightweight, plug-and-play, and adaptive mixed-precision KV cache management framework for Large Language Models (LLMs). It addresses the challenge of increasing inference costs in long-context LLM applications by intelligently managing Key-Value (KV) cache memory through dynamic token eviction and quantization.

## Key Features

- **Adaptive Mixed-Precision**: Dynamically balances full-precision, quantized, and evicted tokens based on layer-specific patterns
- **Lightweight & Plug-and-Play**: Minimal computational overhead with no model retraining required
- **Memory Efficient**: Achieves up to **4× reduction** in KV cache size while maintaining ~97% of baseline performance
- **Performance Preserving**: Matches full-precision accuracy on most short-context tasks and maintains high quality on long-context benchmarks
- **High Throughput**: Achieves nearly the same token generation rate as full-cache baseline

## How It Works

AKCB operates through three main stages:

1. **Statistical Collection**: Gathers basic statistical features of each attention layer and token during inference
2. **Ratio Determination**: Generates tunable original-quantization ratios per layer to determine which tokens remain at full precision vs. quantized
3. **State Assignment**: Uses heavy-hitter scores to estimate token importance and assign each token to one of three states:
   - **Original**: Full precision (high importance)
   - **Quantization**: Lower precision (moderate importance)
   - **Eviction**: Removed from cache (low importance)

## Project Structure

```
AKCB/
├── akcb/                          # Core AKCB implementation
│   ├── calculator.py              # Heavy-hitter score calculation
│   ├── config.py                  # Configuration management
│   └── cache/                     # Cache implementations
│       ├── adaptive_cache.py      # Adaptive cache orchestration
│       ├── mix_cache.py          # Mixed-precision cache
│       ├── origin_cache.py       # Full-precision cache
│       ├── quant_cache.py        # Quantized cache
│       └── window_cache.py       # Sliding window cache
├── model/                         # Modified model architectures
│   ├── modify_qwen3.py           # Qwen3 with AKCB support
│   └── modle_llama.py            # Llama with AKCB support
└── experiments/                   # Evaluation scripts
    ├── lmeval/                   # LM Evaluation Harness integration
    │   └── eval.py
    └── longbench/                # LongBench evaluation
        ├── pred.py               # Inference and prediction
        ├── eval.py               # Evaluation metrics
        └── metrics.py            # Performance metrics
```

## Installation

### Method 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/JianlongLei/AKCB.git
cd AKCB

# Create and activate conda environment
conda env create -f environment.yml
conda activate akcb

# Install AKCB in development mode
pip install -e .
```

Or use the setup script:

```bash
bash setup.sh
conda activate akcb
pip install -e .
```

### Method 2: Using pip

```bash
# Clone the repository
git clone https://github.com/JianlongLei/AKCB.git
cd AKCB

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# For evaluation support
pip install -e ".[eval]"
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### Basic Usage

```python
from akcb.config import ADCacheConfig
from akcb.cache.adaptive_cache import AdaptiveCache

# Configure AKCB
class Args:
    cache_size = 4096
    window_size = 128
    tau1 = 0.1
    tau2 = 0.3
    tau3 = 0.6
    gamma = 1.0
    quant_type = "mix"  # Options: "mix", "quant", "origin", "window"
    compress = True

config = ADCacheConfig(Args())

# Initialize adaptive cache
adaptive_cache = AdaptiveCache(config)
```

### Running Experiments

**LongBench Evaluation:**
```bash
cd experiments/longbench
python pred.py --model_path <path_to_model> \
               --cache_size 4096 \
               --window_size 128 \
               --quant_type mix
```

**LM Eval Harness:**
```bash
cd experiments/lmeval
python eval.py --model_path <path_to_model> \
               --cache_size 4096 \
               --window_size 128
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `cache_size` | Maximum KV cache size | 4096 |
| `window_size` | Size of the sliding window for recent tokens | 128 |
| `tau1`, `tau2`, `tau3` | Thresholds for adaptive ratio calculation | 0.1, 0.3, 0.6 |
| `gamma` | Weight for variance in heavy-hitter score | 1.0 |
| `quant_type` | Cache type: "mix", "quant", "origin", "window" | "mix" |
| `compress` | Enable compression | True |

## Supported Models

- **Llama 3**: Full support with modified attention mechanisms
- **Qwen 3**: Full support with modified attention mechanisms

## Performance

AKCB has been evaluated on a comprehensive suite of benchmarks:

- **Long-context tasks**: Maintains ~97% of baseline performance with 4× cache reduction
- **Short-context tasks**: Matches full-precision accuracy in most cases
- **Throughput**: Achieves nearly the same token generation rate as full-cache baseline
- **Memory savings**: Significant reduction in memory usage on resource-constrained hardware

## Citation

If you use AKCB in your research, please cite:

```bibtex
@article{akcb2025,
  title={AKCB: Adaptive KV Caches under Budget for Efficient Long-Context LLM Inference},
  author={Lei, Jianlong and others},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This work builds upon the Transformer architecture and integrates with popular frameworks like Hugging Face Transformers.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

---

**Note**: AKCB enables sustainable deployment of LLMs on resource-constrained hardware without architectural changes or model retraining, making long-context inference more practical and accessible.
