# ARKV: Adaptive and Resource-Efficient KV Cache Management under Limited Memory Budget for Long-Context Inference in LLMs

The high-level overview of the ARKV framework is as follows:
![System Model](figures/arkv_framework.png)

ARKV consists of three key components: (1) Per-layer Original Quantization (OQ) ratio estimation to determine each layer’s compression sensitivity, (2) Token importance scoring based on online attention statistics, and (3) Tri-state cache assignment to fit a memory budget via selective precision control. 

* Prefill phase: Each layer’s Origin budget is determined by an OQ score combining entropy, variance, and kurtosis of attention distributions. These scores are normalized into an OQ ratio, ensuring the limited cache budget is fairly distributed across layers.
* Decoding phase: At each step, hh scores are computed only on the evictable region $[0 : K − W)$. A keep set is selected, then divided into Original and Quantization according to the budget, with the remainder evicted. The most recent W , the window protection region, tokens are always retained in Original.
* Reconstruction Before Attention: Quantized entries are dequantized on-the-fly using per-token scales and merged with Original entries in their original sequence order. This guarantees a logically contiguous KV cache for attention.

# Abstract

Large Language Models (LLMs) are increasingly deployed in scenarios demanding ultra-long context reasoning, such as agentic workflows and deep research understanding. However, long-context inference is constrained by the KV cache, a transient memory structure that grows linearly with sequence length and batch size, quickly dominating GPU memory usage. Existing memory reduction techniques, including eviction and quantization, often rely on static heuristics and suffer from degraded quality under tight budgets. In this paper, we propose ARKV, a lightweight and adaptive framework that dynamically allocates precision levels to cached tokens based on per-layer attention dynamics and token-level importance. During a short prefill phase, ARKV estimates the original quantization (OQ) ratio of each layer by computing statistical scores such as attention entropy, variance and kurtosis. During decoding, tokens are assigned to one of three states—Original (full precision), Quantization (low precision), or Eviction—according to a fast heavy-hitter scoring strategy. Our experiments on LLaMA3 and Qwen3 models across diverse long- and short-context tasks demonstrate that ARKV preserves ~97% of baseline accuracy on long-context benchmarks while reducing KV memory usage by 4x, with minimal throughput loss. On short-context tasks, ARKV matches full-precision baselines; on GSM8K math reasoning, it significantly outperforms uniform quantization. These results highlight the practical viability of ARKV for scalable LLM deployment, offering fine-grained, data-driven memory control without retraining or architectural modifications.

# Dependencies

> Note: Dependencies are defined in `environment.yml`. Where `>=` means the minimum required version; the exact version installed in your environment may be higher.

| Name | Version | Purpose |
|------|---------|---------|
| `python` | `3.10` | Runtime |
| `pytorch` (`torch`) | `>=2.0.0` | Core deep learning framework |
| `pytorch-cuda` | `11.8` | CUDA runtime for PyTorch (conda) |
| `cudatoolkit` | `11.8` | CUDA toolkit (conda) |
| `transformers` | `>=4.35.0` | HuggingFace Transformers integration |
| `datasets` | `>=2.14.0` | Benchmark / dataset loading |
| `accelerate` | `>=0.24.0` | Multi-device / inference utilities |
| `tokenizers` | `>=0.14.0` | Fast tokenization backend |
| `numpy` | `>=1.24.0` | Numerical computing |
| `tqdm` | (un-pinned) | Progress bars |
| `sentencepiece` | `>=0.1.99` | Tokenizer model support |
| `protobuf` | `>=3.20.0` | Serialization dependency (Transformers ecosystem) |
| `lm-eval` | `>=0.4.0` | (Optional) evaluation harness |
| `rouge` | `>=1.0.1` | (Optional) ROUGE metric |
| `jieba` | `>=0.42.1` | (Optional) Chinese tokenization (some eval pipelines) |
| `fuzzywuzzy` | `>=0.18.0` | (Optional) fuzzy string matching |
| `python-Levenshtein` | `>=0.21.0` | (Optional) fast edit distance |
| `flash-attn` | `>=2.3.0` | (Optional) FlashAttention acceleration |
| `bitsandbytes` | `>=0.41.0` | (Optional) quantization backend |
| `pytest` | `>=8.0.0` | (Optional) testing |
| `black` | `>=23.0.0` | (Optional) formatting |
| `isort` | `>=5.12.0` | (Optional) import sorting |
| `flake8` | `>=6.0.0` | (Optional) linting |

# Environment Setup

The recommended way to set up a runnable environment is to use the provided setup script, which:

1. Creates/updates a conda environment from `environment.yml` (or another env spec you provide), and
2. Installs this repo in editable mode via `pip install -e .`.

## Quick start (recommended)

```bash
./scripts/setup_env.sh
conda activate arkv
```

## Using an exported env spec (optional)

If you generated a conda export (for example, `conda env export --no-builds > env.txt`), you can
use it directly:

```bash
./scripts/setup_env.sh --file env.txt
```

If the export uses a different environment name (e.g., `name: adkv`) but you want to create/update
the `arkv` environment, pass `--name`:

```bash
./scripts/setup_env.sh --file env.txt --name arkv
```

Note: conda exports sometimes include a machine-specific `prefix:` entry. The setup script strips
that line automatically to keep the environment spec portable.

# Repository Structure

This repository is organized around the ARKV runtime pipeline (config → attention patching → prefill scoring → decoding compression), plus an experiments folder for evaluation and analysis.


## Core package: `arkv/`

- `arkv/config.py`
	- Defines `ADCacheConfig`, the central runtime configuration (cache size, window size, `tau1/tau2/tau3`, `gamma`, cache type).
	- Initializes per-layer state such as `prefill` flags and `decoding_compressor` handles.

- `arkv/calculator.py`
	- Prefill-time statistics used to estimate layer sensitivity / OQ preference (entropy, variance, kurtosis).
	- Decoding-time token importance utilities such as heavy-hitter scoring.

- `arkv/cache/`
	- Cache implementations and compressors.
	- `adaptive_cache.py`: a small factory that selects the cache/compressor classes based on `quant_type`.
	- `origin_cache.py`: baseline full-precision cache path.
	- `window_cache.py`: sliding-window style cache / protection window logic.
	- `quant_cache.py`: quantized cache path.
	- `mix_cache.py`: mixed cache combining Original / Quantized regions.
	- Each cache file typically contains both:
		- **Prefill compressor**: builds the initial compressed cache after prompt prefill.
		- **Decoding compressor (layer-wise)**: updates and compresses cache incrementally per decoding step.

- `arkv/model/`
	- HuggingFace Transformers integration points.
	- `modle_llama.py`: patches Llama attention forward to run ARKV logic (prefill scoring + decoding compression) while staying API-compatible with Transformers.
	- `modify_qwen3.py`: analogous patch for Qwen3.

## Experiments: `experiments/`

The `experiments/` directory contains scripts and notebooks used to reproduce evaluation and analysis.

- `experiments/eval/`
	- `eval.py`: evaluation script built on lm-eval, with telemetry (tokens/s) and cache-related logging.
	- `train.py`: Optuna-based hyperparameter tuning for `tau1/tau2/tau3/gamma`.
	- `config/`: hyperparameter presets (e.g., `hypers.json`).

- `experiments/analysis/`
	- Jupyter notebooks for aggregating and plotting results.
	- `data/`: CSV outputs used by notebooks.

- `experiments/LongBench/`
	- LongBench evaluation utilities (`eval.py`, `pred.py`, `metrics.py`) and dataset-specific configs.

### Run examples

After environment setup:

```bash
conda activate arkv
```

Run lm-eval evaluation (example):

```bash
python experiments/eval/eval.py \
	--tasks gsm8k,mmlu,commonsense_qa \
	--quant_type mix \
	--sampling_ratio 0.2 \
	--outdir tps_results
```

Run Optuna tuning (example):

```bash
python experiments/eval/train.py \
	--tasks gsm8k,mmlu,commonsense_qa,longbench \
	--limit 10 \
	--n_trials 10 \
	--cache_size 1024
```



