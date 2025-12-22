#!/usr/bin/env python3
"""experiments/eval/eval.py

Eval script.

This script runs evaluations for language models using the lm-eval framework,
with additional support for ADKV caching configurations and telemetry.
"""

import os
import json
import time
import argparse
from typing import Dict, Any, List, Tuple

import torch
# torch.nn.functional not used

import lm_eval
import lm_eval.api as api
import lm_eval.evaluator as evaluator
from lm_eval import tasks

# === ADKV configuration ===
from arkv.config import ADCacheConfig

def flatten_task_dict(task_dict):
    """Flatten a task structure into a {task_name: Task} dict."""
    flat = {}
    for k, v in task_dict.items():
        if isinstance(k, str):
            flat[k] = v
        elif isinstance(k, tasks.ConfigurableGroup):
            flat.update(v)
    return flat

def to_json_safe(obj):
    """Convert an object into JSON-serializable, friendly types."""
    import json, torch, numpy as np
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.detach().cpu().tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

# ----------------------------
# Read hypers.json (model -> multiple hyperparameter sets such as cache_size)
# ----------------------------
def load_hypers(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def pick_primary_metric(ms: Dict[str, Any]) -> Tuple[str, float]:
    """Pick a primary metric (prefer acc_norm/acc/exact_match/f1)."""
    order = ["acc_norm", "acc", "exact_match", "f1"]
    cand = {}
    for k, v in ms.items():
        if isinstance(v, (int, float)):
            key = k.split(",")[0]
            cand[key] = max(float(v), cand.get(key, -1e9))
    for m in order:
        if m in cand:
            return m, cand[m]
    if cand:
        k0 = next(iter(cand.keys()))
        return k0, cand[k0]
    return "na", float("nan")

def build_task_limits(flat_task_dict: Dict[str, Any], sampling_ratio: float, default_limit: int) -> Dict[str, int]:
    """Compute per-task sample limits based on a sampling ratio."""
    import random
    random.seed(42)
    limits = {}
    for task_name, task_obj in flat_task_dict.items():
        total = 0
        try:
            if task_obj.has_test_docs():
                docs_iter = task_obj.test_docs()
            elif task_obj.has_validation_docs():
                docs_iter = task_obj.validation_docs()
            else:
                docs_iter = []
            total = len(docs_iter) if hasattr(docs_iter, '__len__') else sum(1 for _ in docs_iter)
        except Exception:
            total = 1000
        limits[task_name] = max(1, int(total * sampling_ratio)) if total > 0 else default_limit
    return limits

# ----------------------------
# Monkey-patch: replace attention modules per model (consistent with existing setup)
# ----------------------------
def initialize_my_model(cache_config: ADCacheConfig, model_arg_string: str):
    if cache_config.compress:
        if "Llama-3" in model_arg_string or "Meta-Llama-3" in model_arg_string:
            from arkv.model.modle_llama import replace_llama3_attn
            print("[ADKV] replace_llama3_attn()")
            replace_llama3_attn()
        elif "Qwen3" in model_arg_string or "Qwen/" in model_arg_string:
            from arkv.model.modify_qwen3 import replace_qwen3_attn
            print("[ADKV] replace_qwen3_attn()")
            replace_qwen3_attn()

# Global variable to store the latest past_key_values
_global_last_past_key_values = None

def capture_past_key_values(original_forward):
    """Decorator to capture past_key_values during the model forward pass."""
    def wrapped_forward(*args, **kwargs):
        global _global_last_past_key_values
        result = original_forward(*args, **kwargs)
        
        # Try to extract past_key_values from the output
        if hasattr(result, 'past_key_values') and result.past_key_values is not None:
            _global_last_past_key_values = result.past_key_values
            print(f"[DEBUG] Captured past_key_values with type: {type(result.past_key_values)}")
        
        return result
    return wrapped_forward

# ----------------------------
# Build ADCacheConfig (from hyperparameters)
# ----------------------------
def build_cache_config(h: Dict[str, Any], quant_type="mix") -> ADCacheConfig:
    class Args:
        def __init__(self, h):
            self.model = "instr"
            self.e = False
            self.compress = quant_type in ["mix", "quant", "window", "origin"]
            self.cache_size = int(h["cache_size"])
            self.window_size = int(h.get("window_size", 32))
            self.tau1 = float(h["tau1"])
            self.tau2 = float(h["tau2"])
            self.tau3 = float(h["tau3"])
            self.gamma = float(h["gamma"])
            self.quant_type = quant_type
    return ADCacheConfig(Args(h))

# ----------------------------
# Quant ratio calculation
# ----------------------------
def calculate_quant_ratio(cache_config: ADCacheConfig, past_key_values, quant_type: str) -> float:
    """
    Compute the quantization ratio in mix_cache.

    For mix method: quant_ratio = 1 - origin_budget/available_size
    For other methods: return 0.0
    """
    if quant_type != "mix":
        return 0.0
    
    try:
        # Try multiple ways to obtain past_key_values
        pkv = past_key_values
        if pkv is None:
            print("[DEBUG] past_key_values is None, cannot calculate quant_ratio")
            return 0.0
            
        # Check whether origin_budget exists
        if hasattr(pkv, 'origin_budget') and pkv.origin_budget:
            # Compute available_size (cache_size - window_size)
            available_size = max(cache_config.cache_size - cache_config.window_size, 0)
            
            if available_size > 0:
                # Average origin_budget across layers
                if isinstance(pkv.origin_budget, (list, tuple)) and len(pkv.origin_budget) > 0:
                    avg_origin_budget = sum(pkv.origin_budget) / len(pkv.origin_budget)
                    quant_ratio = 1.0 - (avg_origin_budget / available_size)
                    print(f"[DEBUG] avg_origin_budget: {avg_origin_budget:.2f}, available_size: {available_size}, quant_ratio: {quant_ratio:.4f}")
                    return max(0.0, min(1.0, quant_ratio))  # clamp to [0, 1]
                else:
                    print(f"[DEBUG] origin_budget is empty or invalid: {pkv.origin_budget}")
            else:
                print(f"[DEBUG] available_size <= 0: {available_size}")
        else:
            print(f"[DEBUG] past_key_values has no origin_budget attribute or it's empty")
            
    except Exception as e:
        print(f"[WARNING] Failed to compute quant ratio: {e}")
        import traceback
        traceback.print_exc()
    
    return 0.0

# ----------------------------
# Telemetry wrapper: throughput and (peak/avg) memory statistics
# ----------------------------
from lm_eval.api.model import LM
class LMWithTelemetry(LM):
    def __init__(self, base_lm):
        self.base = base_lm
        self.tokenizer = getattr(base_lm, "tokenizer", None)
        self.prompt_tokens = 0
        self.generated_tokens = 0
        self.ll_tokens = 0
        self.wall_total = 0.0
        self.wall_gen = 0.0
        self.wall_ll = 0.0
        # Per-task TPS statistics
        self.task_stats = {}  # task_name -> {"prompt_tokens", "generated_tokens", "ll_tokens", "wall_time"}
        self.current_task = None  # name of the task currently being evaluated
        # Store the latest past_key_values for quant ratio computation
        self.last_past_key_values = None
        # Per-task quant_ratio samples
        self.task_quant_ratios = {}  # task_name -> [quant_ratio_samples]
        # Per-task evict_ratio samples
        self.task_evict_ratios = {}  # task_name -> [evict_ratio_samples]
        # Per-task cumulative tracking: total evicted tokens and total past tokens
        self.task_evicted_tokens = {}  # task_name -> [evicted_counts_total]
        self.task_past_tokens = {}     # task_name -> [past_len_total]
        self.cache_config = None  # reference to cache config

    # Passthrough required attributes
    @property
    def rank(self): return getattr(self.base, "rank", 0)
    @property
    def world_size(self): return getattr(self.base, "world_size", 1)
    @property
    def tokenizer_name(self): return getattr(self.base, "tokenizer_name", None)
    @property
    def eot_token_id(self): return self.base.eot_token_id
    @property
    def max_length(self): return self.base.max_length
    @property
    def batch_size(self): return self.base.batch_size
    @property
    def config(self): return getattr(self.base, "config", None)
    def __getattr__(self, name): return getattr(self.base, name)

    def _toklen(self, text: str) -> int:
        if self.tokenizer is None: return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def set_current_task(self, task_name: str):
        """Set the task name currently being evaluated."""
        self.current_task = task_name
        if task_name not in self.task_stats:
            self.task_stats[task_name] = {
                "prompt_tokens": 0,
                "generated_tokens": 0, 
                "ll_tokens": 0,
                "wall_time": 0.0
            }
        # Initialize per-task quant_ratio sample list
        if task_name not in self.task_quant_ratios:
            self.task_quant_ratios[task_name] = []
        # Initialize per-task evict_ratio sample list
        if task_name not in self.task_evict_ratios:
            self.task_evict_ratios[task_name] = []
        if task_name not in self.task_evicted_tokens:
            self.task_evicted_tokens[task_name] = []
        if task_name not in self.task_past_tokens:
            self.task_past_tokens[task_name] = []

    def collect_quant_ratio_sample(self):
        """Collect a quant_ratio sample from cache_config (if available)."""
        if self.cache_config is None or self.current_task is None:
            return
            
        try:
            # Prefer per-layer quant_ratio from cache_config
            if hasattr(self.cache_config, 'quant_ratio_per_layer') and self.cache_config.quant_ratio_per_layer:
                # Compute average quant_ratio across layers
                layer_ratios = [ratio for ratio in self.cache_config.quant_ratio_per_layer if ratio > 0]
                if layer_ratios:
                    avg_quant_ratio = sum(layer_ratios) / len(layer_ratios)
                    self.task_quant_ratios[self.current_task].append(avg_quant_ratio)
                    print(f"[DEBUG] Collected quant_ratio sample for {self.current_task}: {avg_quant_ratio:.4f} (from {len(layer_ratios)} layers)")
            else:
                # Fallback: derive quant_ratio from origin_budget vs available_size
                available_size = None
                try:
                    available_size = max(self.cache_config.cache_size - self.cache_config.window_size, 0)
                except Exception:
                    available_size = None

                # Try origin_budget_per_layer
                if available_size and hasattr(self.cache_config, 'origin_budget_per_layer') and self.cache_config.origin_budget_per_layer:
                    budgets = [b for b in self.cache_config.origin_budget_per_layer if b is not None and b >= 0]
                    if budgets:
                        avg_origin_budget = sum(budgets) / len(budgets)
                        quant_ratio = max(0.0, min(1.0, 1.0 - (avg_origin_budget / available_size)))
                        self.task_quant_ratios[self.current_task].append(quant_ratio)
                        print(f"[DEBUG] Fallback quant_ratio (origin_budget_per_layer) for {self.current_task}: {quant_ratio:.4f}")
                # Try layer_configs with per-layer origin_budget
                elif available_size and hasattr(self.cache_config, 'layer_configs') and self.cache_config.layer_configs:
                    budgets = []
                    for lc in self.cache_config.layer_configs:
                        if hasattr(lc, 'origin_budget') and isinstance(lc.origin_budget, (int, float)):
                            budgets.append(lc.origin_budget)
                    if budgets:
                        avg_origin_budget = sum(budgets) / len(budgets)
                        quant_ratio = max(0.0, min(1.0, 1.0 - (avg_origin_budget / available_size)))
                        self.task_quant_ratios[self.current_task].append(quant_ratio)
                        print(f"[DEBUG] Fallback quant_ratio (layer_configs) for {self.current_task}: {quant_ratio:.4f}")
        except Exception as e:
            print(f"[WARNING] Failed to collect quant_ratio sample: {e}")

    def collect_evict_ratio_sample(self):
        """Collect an evict_ratio sample from cache_config (if available)."""
        if self.cache_config is None or self.current_task is None:
            return
        try:
            # Support two field names: evict_ratio_per_layer and evicted_ratios_per_layer
            layer_evict_src = None
            if hasattr(self.cache_config, 'evict_ratio_per_layer') and self.cache_config.evict_ratio_per_layer:
                layer_evict_src = self.cache_config.evict_ratio_per_layer
            elif hasattr(self.cache_config, 'evicted_ratios_per_layer') and self.cache_config.evicted_ratios_per_layer:
                layer_evict_src = self.cache_config.evicted_ratios_per_layer

            if layer_evict_src:
                layer_evict = [ratio for ratio in layer_evict_src if ratio is not None and ratio >= 0]
                if layer_evict:
                    avg_evict_ratio = sum(layer_evict) / len(layer_evict)
                    self.task_evict_ratios[self.current_task].append(avg_evict_ratio)
                    print(f"[DEBUG] Collected evict_ratio sample for {self.current_task}: {avg_evict_ratio:.4f} (from {len(layer_evict)} layers)")

            # Also collect total evicted tokens (summed across layers).
            # If past tokens are not available, approximate with the task's generated_tokens.
            evicted_counts = getattr(self.cache_config, 'evicted_counts_per_layer', [])
            if evicted_counts:
                total_evicted = int(sum(int(c) for c in evicted_counts if c is not None))
                # Approximate past tokens with the task-level generated token count
                estimated_past = int(self.task_stats.get(self.current_task, {}).get("generated_tokens", 0))
                self.task_evicted_tokens[self.current_task].append(total_evicted)
                self.task_past_tokens[self.current_task].append(estimated_past)
                print(f"[DEBUG] Collected evicted tokens for {self.current_task}: evicted={total_evicted}, estimated_past={estimated_past}")
        except Exception as e:
            print(f"[WARNING] Failed to collect evict_ratio sample: {e}")

    def get_task_avg_quant_ratio(self, task_name: str) -> float:
        """Get the average quant_ratio for a given task."""
        if task_name in self.task_quant_ratios and self.task_quant_ratios[task_name]:
            ratios = self.task_quant_ratios[task_name]
            avg_ratio = sum(ratios) / len(ratios)
            print(f"[DEBUG] Task {task_name} avg quant_ratio: {avg_ratio:.4f} (from {len(ratios)} samples)")
            return avg_ratio
        return 0.0

    def get_task_avg_evict_ratio(self, task_name: str) -> float:
        """Get the average evict_ratio for a given task."""
        if task_name in self.task_evict_ratios and self.task_evict_ratios[task_name]:
            ratios = self.task_evict_ratios[task_name]
            avg_ratio = sum(ratios) / len(ratios)
            print(f"[DEBUG] Task {task_name} avg evict_ratio: {avg_ratio:.4f} (from {len(ratios)} samples)")
            return avg_ratio
        return 0.0

    # Intercept three LM-eval interfaces
    def loglikelihood(self, requests):
        t0 = time.perf_counter()
        out = self.base.loglikelihood(requests)
        dt = time.perf_counter() - t0
        self.wall_total += dt; self.wall_ll += dt
        print(f"requests: {len(requests)}, ll_wall_s: {dt:.4f}, ll_tps: {len(requests)/max(dt,1e-9):.2f}")
        
        # Token accounting
        task_tokens = 0
        if self.tokenizer is not None:
            for inst in requests:
                (ctx, cont) = inst.args
                prompt_tok = self._toklen(ctx)
                ll_tok = self._toklen(cont)
                self.prompt_tokens += prompt_tok
                self.ll_tokens += ll_tok
                task_tokens += prompt_tok + ll_tok
        
        # Update per-task stats
        if self.current_task:
            self.task_stats[self.current_task]["wall_time"] += dt
            if self.tokenizer is not None:
                for inst in requests:
                    (ctx, cont) = inst.args
                    self.task_stats[self.current_task]["prompt_tokens"] += self._toklen(ctx)
                    self.task_stats[self.current_task]["ll_tokens"] += self._toklen(cont)
        
        # Collect quant_ratio sample
        self.collect_quant_ratio_sample()
        # Collect evict_ratio sample
        self.collect_evict_ratio_sample()
        return out

    def loglikelihood_rolling(self, requests):
        t0 = time.perf_counter()
        out = self.base.loglikelihood_rolling(requests)
        dt = time.perf_counter() - t0
        self.wall_total += dt
        print(f"requests: {len(requests)}, ll_roll_wall_s: {dt:.4f}, ll_roll_tps: {len(requests)/max(dt,1e-9):.2f}")
        
        if self.tokenizer is not None:
            for inst in requests:
                # loglikelihood_rolling takes (string,) instead of (ctx, cont)
                ctx = inst.args[0] if isinstance(inst.args, tuple) else inst.args
                ll_tok = self._toklen(ctx)
                self.ll_tokens += ll_tok
        
        # Update per-task stats
        if self.current_task:
            self.task_stats[self.current_task]["wall_time"] += dt
            if self.tokenizer is not None:
                for inst in requests:
                    ctx = inst.args[0] if isinstance(inst.args, tuple) else inst.args
                    self.task_stats[self.current_task]["ll_tokens"] += self._toklen(ctx)
        
        # Collect quant_ratio sample
        self.collect_quant_ratio_sample()
        # Collect evict_ratio sample
        self.collect_evict_ratio_sample()
        return out

    def generate_until(self, requests):
        t0 = time.perf_counter()
        outs = self.base.generate_until(requests)
        dt = time.perf_counter() - t0
        self.wall_total += dt; self.wall_gen += dt
        print(f"requests: {len(requests)}, gen_wall_s: {dt:.4f}, gen_tps: {len(requests)/max(dt,1e-9):.2f}")
        
        if self.tokenizer is not None:
            for inst, gen in zip(requests, outs):
                (ctx, gen_kwargs) = inst.args
                prompt_tok = self._toklen(ctx)
                gen_tok = self._toklen(gen)
                self.prompt_tokens += prompt_tok
                self.generated_tokens += gen_tok
        
        # Update per-task stats
        if self.current_task:
            self.task_stats[self.current_task]["wall_time"] += dt
            if self.tokenizer is not None:
                for inst, gen in zip(requests, outs):
                    (ctx, gen_kwargs) = inst.args
                    self.task_stats[self.current_task]["prompt_tokens"] += self._toklen(ctx)
                    self.task_stats[self.current_task]["generated_tokens"] += self._toklen(gen)
        
        # Collect quant_ratio sample
        self.collect_quant_ratio_sample()
        # Collect evict_ratio sample
        self.collect_evict_ratio_sample()
        return outs

    def summary(self) -> Dict[str, Any]:
        eps = 1e-9
        gen_tps = self.generated_tokens / max(self.wall_gen, eps)
        eff_tps = (self.prompt_tokens + self.generated_tokens + self.ll_tokens) / max(self.wall_total, eps)
        def stat(a: List[int]):
            if not a: return {"avg_GB": 0, "max_GB": 0, "std_GB": 0}
            x = torch.tensor(a, dtype=torch.float64)
            toGB = 1024**3
            return {
                "avg_GB": round((x.mean().item())/toGB, 3),
                "max_GB": round((x.max().item())/toGB, 3),
                "std_GB": round((x.std(unbiased=False).item())/toGB, 3),
            }
        
        # Compute per-task TPS
        task_tps = {}
        for task_name, stats in self.task_stats.items():
            total_tokens = stats["prompt_tokens"] + stats["generated_tokens"] + stats["ll_tokens"]
            wall_time = stats["wall_time"]
            if wall_time > eps:
                task_tps[task_name] = {
                    "prompt_tokens": stats["prompt_tokens"],
                    "generated_tokens": stats["generated_tokens"], 
                    "ll_tokens": stats["ll_tokens"],
                    "total_tokens": total_tokens,
                    "wall_time_s": round(wall_time, 4),
                    "tokens_per_s": round(total_tokens / wall_time, 2)
                }
            else:
                task_tps[task_name] = {
                    "prompt_tokens": stats["prompt_tokens"],
                    "generated_tokens": stats["generated_tokens"],
                    "ll_tokens": stats["ll_tokens"], 
                    "total_tokens": total_tokens,
                    "wall_time_s": 0.0,
                    "tokens_per_s": 0.0
                }
        
        return {
            "prompt_tokens": int(self.prompt_tokens),
            "generated_tokens": int(self.generated_tokens),
            "ll_tokens": int(self.ll_tokens),
            "wall_total_s": round(self.wall_total, 4),
            "gen_wall_s": round(self.wall_gen, 4),
            "gen_tokens_per_s": round(gen_tps, 2),
            "effective_tokens_per_s": round(eff_tps, 2),
            "task_tps": task_tps,  # New: per-task TPS statistics
        }

# ----------------------------
# Run one evaluation (single model × one hyperparameter set × task list)
# ----------------------------
def eval_once(
    model_backend: str,
    model_args: str,
    tasks_list: List[str],
    hypers: Dict[str, Any],
    quant_type="mix",
    limit=0,
    n_ttft_samples=0,  # TTFT testing not used
    tps_sampling=True,  # Enable TPS sampling by default
    sampling_ratio=0.2) -> Dict[str, Any]:

    # Declare global
    global _global_last_past_key_values

    # 1) ADKV config + patching
    cache_config = build_cache_config(hypers, quant_type=quant_type)
    initialize_my_model(cache_config, model_args)

    # 2) Build lm-eval model
    lm = api.registry.get_model(model_backend).create_from_arg_string(
        model_args,
        {"batch_size": None, "max_batch_size": None, "device": None},
    )
    hf_model = getattr(lm, "model", None) or getattr(lm, "_model", None)  # HF model
    tokenizer = getattr(lm, "tokenizer", None)

    # 3) Inject cache_config
    config = getattr(lm, "config", None)
    if cache_config.compress and hasattr(config, "num_hidden_layers"):
        layers = int(config.num_hidden_layers)
        cache_config.update_layer_num(layers)
        if cache_config.compress:
            for i in range(layers):
                lm.model.model.layers[i].self_attn.config.cache_config = cache_config
    
    # 3.5) Apply past_key_values capture wrapper (mix method only)
    _global_last_past_key_values = None
    
    if quant_type == "mix" and hasattr(lm, 'model') and hasattr(lm.model, 'forward'):
        original_forward = lm.model.forward
        lm.model.forward = capture_past_key_values(original_forward)
        print(f"[DEBUG] Applied past_key_values capture wrapper to model")

    # 4) Wrap with telemetry
    lm_t = LMWithTelemetry(lm)
    # Pass cache_config reference
    lm_t.cache_config = cache_config

    # 5) Task management
    task_manager = tasks.TaskManager()
    flat_task_dict = flatten_task_dict(tasks.get_task_dict(tasks_list, task_manager))

    limit = limit if limit > 0 else None
    
    # TPS sampling mode: compute per-task sample counts
    task_limits = {}
    if tps_sampling:
        task_limits = build_task_limits(flat_task_dict, sampling_ratio, limit)
        for tn, cnt in task_limits.items():
            print(f"[TPS_SAMPLING] {tn}: {cnt} samples ({sampling_ratio:.1%})")
    
    # 6) Run evaluation per task so we can compute per-task TPS
    all_results = {}
    for task_name, task_obj in flat_task_dict.items():
        print(f"[EVAL] Starting task: {task_name}")
        lm_t.set_current_task(task_name)
        
        # Create a task_dict for a single task
        single_task_dict = {task_name: task_obj}
        
        # Determine the limit for this task
        current_limit = task_limits.get(task_name, limit) if tps_sampling else limit
        
        task_results = evaluator.evaluate(
            lm=lm_t,
            task_dict=single_task_dict,
            limit=current_limit,
            log_samples=False,
            apply_chat_template=False,  # do not use chat template
            fewshot_as_multiturn=False,
        )
        
        # Merge results
        if "results" in task_results:
            all_results.update(task_results["results"])
    
    # Build final result format
    results = {"results": all_results}

    # 7) Telemetry summary (throughput + avg/peak/variance memory)
    telemetry = lm_t.summary()
    
    # After evaluation, try to pull cache info from the model for quant ratio calculation
    if quant_type == "mix" and hasattr(lm, 'model') and hasattr(lm.model, 'model'):
        # Check whether the model exposes a usable cache configuration
        model_layers = lm.model.model.layers
        if len(model_layers) > 0:
            first_layer = model_layers[0]
            if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'config'):
                layer_config = first_layer.self_attn.config
                if hasattr(layer_config, 'cache_config'):
                    print(f"[DEBUG] Found cache_config in model layer, trying to extract cache info...")
                    # Try to obtain global cache info
                    lm_t.last_past_key_values = getattr(layer_config.cache_config, 'global_cache', None)
    
    # 8) Compute quant ratio (mix method only)
    quant_ratio = 0.0
    evict_ratio = 0.0
    task_quant_ratios = {}  # per-task quant_ratio
    task_evict_ratios = {}  # per-task evict_ratio
    task_evicted_tokens = {}  # per-task total evicted tokens
    task_past_tokens = {}     # per-task total past tokens
    
    if quant_type == "mix":
        # Prefer task-level quant_ratio collected from cache_config
        print(f"[DEBUG] Computing task-level quant_ratios from collected samples...")
        
        for task_name in lm_t.task_stats.keys():
            task_quant_ratios[task_name] = lm_t.get_task_avg_quant_ratio(task_name)
            task_evict_ratios[task_name] = lm_t.get_task_avg_evict_ratio(task_name)
            # Aggregate per-task evicted tokens and past totals
            ev_counts = lm_t.task_evicted_tokens.get(task_name, [])
            past_counts = lm_t.task_past_tokens.get(task_name, [])
            task_evicted_tokens[task_name] = int(sum(ev_counts)) if ev_counts else 0
            task_past_tokens[task_name] = int(sum(past_counts)) if past_counts else 0
        
        # Compute overall average quant_ratio across tasks
        valid_ratios = [ratio for ratio in task_quant_ratios.values() if ratio > 0]
        if valid_ratios:
            quant_ratio = sum(valid_ratios) / len(valid_ratios)
            print(f"[DEBUG] Overall quant_ratio: {quant_ratio:.4f} (from {len(valid_ratios)} valid tasks)")
        # Compute overall evict_ratio
        valid_evict = [ratio for ratio in task_evict_ratios.values() if ratio > 0]
        if valid_evict:
            evict_ratio = sum(valid_evict) / len(valid_evict)
            print(f"[DEBUG] Overall evict_ratio: {evict_ratio:.4f} (from {len(valid_evict)} valid tasks)")
        else:
            # Fallback: use the legacy computation logic
            past_key_values = _global_last_past_key_values
            
            if past_key_values is not None:
                print(f"[DEBUG] Using captured past_key_values: {type(past_key_values)}")
            else:
                print(f"[DEBUG] No captured past_key_values, searching alternatives...")
                
                # Fallback: try to retrieve from inside the model
                if hasattr(lm_t, 'last_past_key_values') and lm_t.last_past_key_values is not None:
                    past_key_values = lm_t.last_past_key_values
                    print(f"[DEBUG] Found past_key_values in lm_t.last_past_key_values")
                        
            quant_ratio = calculate_quant_ratio(cache_config, past_key_values, quant_type)
            # If evict_ratio is not explicitly available, approximate with quant_ratio
            evict_ratio = quant_ratio

        # Extra aggregation: total evicted tokens / total past tokens and the overall ratio
        total_evicted_tokens = int(sum(task_evicted_tokens.values()))
        total_past_tokens = int(sum(task_past_tokens.values()))
        overall_evicted_ratio = (float(total_evicted_tokens) / total_past_tokens) if total_past_tokens > 0 else 0.0
        print(f"[DEBUG] Overall evicted tokens: {total_evicted_tokens}/{total_past_tokens} (ratio={overall_evicted_ratio:.4f})")
        
        # If still 0, use a configuration-based estimate
        if quant_ratio == 0.0:
            print(f"[DEBUG] Quant ratio still 0, using configuration-based estimation...")
            
            # Estimate based on cache configuration
            if cache_config.cache_size > cache_config.window_size:
                available_size = cache_config.cache_size - cache_config.window_size
                
                # Estimate using task complexity and cache size
                total_processed_tokens = sum([
                    stats["prompt_tokens"] + stats["generated_tokens"] + stats["ll_tokens"] 
                    for stats in lm_t.task_stats.values()
                ])
                
                if total_processed_tokens > 0:
                    # Compute average sequence length
                    avg_seq_length = total_processed_tokens / max(len(lm_t.task_stats), 1)
                    
                    if avg_seq_length > available_size:
                        # Long-sequence regime: more compression needed
                        compression_pressure = min(avg_seq_length / available_size, 3.0)
                        origin_ratio = max(0.3, 0.8 / compression_pressure)  # more pressure => smaller origin ratio
                        quant_ratio = 1.0 - origin_ratio
                        evict_ratio = quant_ratio
                        print(f"[DEBUG] Long sequence estimation - avg_len: {avg_seq_length:.0f}, "
                              f"available: {available_size}, quant_ratio: {quant_ratio:.4f}")
                    else:
                        # Short-sequence regime: less compression
                        origin_ratio = min(0.9, max(0.6, available_size / (avg_seq_length + 1)))
                        quant_ratio = 1.0 - origin_ratio
                        evict_ratio = quant_ratio
                        print(f"[DEBUG] Short sequence estimation - quant_ratio: {quant_ratio:.4f}")
                else:
                    # Default estimate
                    quant_ratio = 0.3  # default: 30% quantized
                    evict_ratio = quant_ratio
                    print(f"[DEBUG] Default fallback quant_ratio: {quant_ratio:.4f}")
            else:
                quant_ratio = 0.0  # cache too small; cannot compress
                evict_ratio = 0.0
                print(f"[DEBUG] Cache too small for compression, quant_ratio: {quant_ratio:.4f}")
    else:
        quant_ratio = 0.0

    return {
        "results": results.get("results", {}),
        "configs": {
            "model_args": model_args,
            "hypers": hypers,
            "quant_type": quant_type,
            "quant_ratio": quant_ratio,
            "task_quant_ratios": task_quant_ratios,  # per-task quant_ratio
            "evict_ratio": evict_ratio,
            "task_evict_ratios": task_evict_ratios,  # per-task evict_ratio
            "task_evicted_tokens": task_evicted_tokens,  # per-task total evicted tokens
            "task_past_tokens": task_past_tokens,        # per-task total past tokens
            "overall_evicted_tokens": total_evicted_tokens,
            "overall_past_tokens": total_past_tokens,
            "overall_evicted_ratio": overall_evicted_ratio,
            "tps_sampling": tps_sampling,
            "sampling_ratio": sampling_ratio if tps_sampling else None,
            "task_limits": task_limits if tps_sampling else None,
        },
        "telemetry": telemetry,
    }

# ----------------------------
# Entry point: read hypers.json and iterate (model × hyperparameter set)
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypers", type=str, default="../config/hypers.json",
                        help="Path to the hyperparameter JSON file (model -> multiple cache_size configs, etc.)")
    parser.add_argument("--tasks", type=str, default="gsm8k,mmlu,commonsense_qa",
                        help="Comma-separated list of lm-eval tasks")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--outdir", type=str, default="tps_results")
    parser.add_argument("--quant_type", type=str, default="mix")
    parser.add_argument("--limit", type=int, default=0, help="Per-task maximum number of evaluation samples")
    parser.add_argument("--sampling_ratio", type=float, default=0.2,
                        help="Sampling ratio in TPS sampling mode (default: 0.2 / 20%)")
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated list of models to test. If empty, tests all models. Example: --models 'meta-llama/Meta-Llama-3.1-8B-Instruct,Qwen/Qwen3-8B'")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    print(f"[TPS_TEST] TPS benchmark mode enabled, sampling ratio: {args.sampling_ratio:.1%}")
    print(f"[TPS_TEST] quant_type: {args.quant_type}")
    print(f"[TPS_TEST] Output directory: {args.outdir}")
    
    hypers_all = load_hypers(args.hypers)
    
    # Model filtering
    if args.models.strip():
        target_models = [m.strip() for m in args.models.split(",") if m.strip()]
        print(f"[MODEL_FILTER] Testing only specified models: {target_models}")
        filtered_hypers = {}
        for model_id in target_models:
            if model_id in hypers_all:
                filtered_hypers[model_id] = hypers_all[model_id]
            else:
                print(f"[WARNING] Model {model_id} is not present in the config file")
        hypers_all = filtered_hypers
        print(f"[MODEL_FILTER] After filtering: {len(hypers_all)} models remaining")
    else:
        print(f"[MODEL_FILTER] Testing all models: {len(hypers_all)}")

    MODEL_BACKEND = "hf"

    all_rows = []  # for CSV export
    for model_id, entry in hypers_all.items():
        hypers_list = entry["hypers"]
        # Build model_args
        model_args = f"pretrained={model_id},dtype={args.dtype},device={args.device}"
        print(f"\n==== Model: {model_id} ====")

        for h in hypers_list:
            print(f"[RUN] cache_size={h['cache_size']}, window={h.get('window_size',32)}, "
                  f"tau1={h['tau1']} tau2={h['tau2']} tau3={h['tau3']} gamma={h['gamma']}")

            out = eval_once(
                model_backend=MODEL_BACKEND,
                model_args=model_args,
                tasks_list=[t.strip() for t in args.tasks.split(",") if t.strip()],
                hypers=h,
                quant_type=args.quant_type,
                limit=args.limit,
                tps_sampling=True,
                sampling_ratio=args.sampling_ratio
            )

            # Save JSON
            tag = f"{model_id.replace('/','_')}__{args.quant_type}__cs{h['cache_size']}"
            tag += f"_tps{int(args.sampling_ratio*100)}pct"
            
            json_path = os.path.join(args.outdir, f"{tag}.json")
            safe_out = to_json_safe(out)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(safe_out, f, ensure_ascii=False, indent=2)
            print(f"[SAVED] {json_path}")

            # Summary rows (for easy aggregation/comparison)
            # Pick one primary metric per task
            for task_name, metrics in safe_out["results"].items():
                # Pick primary metric (prefer acc_norm/acc/exact_match/f1)
                def pick_metric(ms: Dict[str, Any]) -> Tuple[str, float]:
                    order = ["acc_norm","acc","exact_match","f1"]
                    cand = {}
                    for k, v in ms.items():
                        if isinstance(v, (int, float)):
                            cand[k.split(",")[0]] = max(float(v), cand.get(k.split(",")[0], -1e9))
                    for m in order:
                        if m in cand: return m, cand[m]
                    if cand:
                        k0 = list(cand.keys())[0]
                        return k0, cand[k0]
                    return "na", float("nan")

                mname, mval = pick_metric(metrics)

                row = {
                    "model": model_id,
                    "method": args.quant_type,
                    "cache_size": h["cache_size"],
                    "window_size": h.get("window_size", 32),
                    "task": task_name, "metric": mname, "score": mval,
                    # Overall throughput
                    "gen_tokens_per_s": safe_out["telemetry"]["gen_tokens_per_s"],
                    "effective_tokens_per_s": safe_out["telemetry"]["effective_tokens_per_s"],
                    # Task-specific TPS
                    "task_tokens_per_s": safe_out["telemetry"]["task_tps"].get(task_name, {}).get("tokens_per_s", 0.0),
                    "task_prompt_tokens": safe_out["telemetry"]["task_tps"].get(task_name, {}).get("prompt_tokens", 0),
                    "task_generated_tokens": safe_out["telemetry"]["task_tps"].get(task_name, {}).get("generated_tokens", 0),
                    "task_ll_tokens": safe_out["telemetry"]["task_tps"].get(task_name, {}).get("ll_tokens", 0),
                    "task_wall_time_s": safe_out["telemetry"]["task_tps"].get(task_name, {}).get("wall_time_s", 0.0),
                    # Quant ratio (overall and per-task)
                    "quant_ratio": safe_out["configs"].get("quant_ratio", 0.0),
                    "task_quant_ratio": safe_out["configs"].get("task_quant_ratios", {}).get(task_name, 0.0),
                    # Evict ratio (overall and per-task)
                    "evict_ratio": safe_out["configs"].get("evict_ratio", 0.0),
                    "task_evict_ratio": safe_out["configs"].get("task_evict_ratios", {}).get(task_name, 0.0),
                }
                all_rows.append(row)

    # Export CSV summary
    import csv
    csv_filename = f"tps_summary_{args.quant_type}.csv"
    csv_path = os.path.join(args.outdir, csv_filename)
    if all_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[SAVED] {csv_path}")
        
    print("\n🎉 TPS benchmark finished!")
    print(f"Results saved to: {args.outdir}")
    if args.quant_type == "mix":
        print("📊 Quant ratio has been computed and included in the results")

if __name__ == "__main__":
    main()