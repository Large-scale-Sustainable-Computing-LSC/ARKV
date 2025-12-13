# AKCB

AKCB (**A**daptive **K**V **C**aches under **B**udget) is a lightweight research codebase for mixed-precision / adaptive KV-cache management during LLM inference.

It provides:
- Cache configuration via `akcb.config.ADCacheConfig`
- Cache implementations in `akcb/cache/` (mix/quant/origin/window)
- Attention-based utilities in `akcb/calculator.py` (heavy-hitter scoring, entropy, kurtosis)

## Install

### Option A: pip editable install (recommended)

From the repo root:

```bash
pip install -U pip
pip install -e .
```

Install optional extras when needed:

```bash
pip install -e ".[dev]"     # pytest, formatting tools
pip install -e ".[eval]"    # lm-eval + metrics deps
pip install -e ".[flash]"   # flash-attn (requires compatible CUDA/GPU)
pip install -e ".[quant]"   # bitsandbytes
```

### Option B: conda environment

An example `environment.yml` is provided:

```bash
conda env create -f environment.yml
conda activate akcb
pip install -e .
```

## Quick sanity check

```bash
python -c "import akcb; from akcb.config import ADCacheConfig; print('ok')"
```

## Run tests

```bash
pip install -e ".[dev]"
pytest
```

## Package layout

- `akcb/config.py`: cache hyperparameters container (`ADCacheConfig`)
- `akcb/cache/`: cache implementations
  - `adaptive_cache.py`: cache factory / high-level wrapper
  - `mix_cache.py`, `quant_cache.py`, `origin_cache.py`, `window_cache.py`
- `akcb/model/`: model-specific attention patching utilities
  - `modle_llama.py` (Llama 3 attention replacement)
  - `modify_qwen3.py` (Qwen attention replacement)

## Notes

- GPU/CUDA requirements depend on whether you enable FlashAttention / bfloat16 inference.
- The repository also contains experiment scripts under `experiments/` in some branches/setups; if your working tree includes them, they can be executed after installing `.[eval]`.
