import os
import json
import time
import math
import argparse
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F

import lm_eval
import lm_eval.api as api
import lm_eval.evaluator as evaluator
from lm_eval import tasks

# === AKCB configuration ===
from akcb.config import ADCacheConfig

def flatten_task_dict(task_dict):
    """
    Flatten the structure returned by get_task_dict:
    - Normal tasks:  { "gsm8k": Task }
    - Task groups:   { ConfigurableGroup(mmlu): { "mmlu_subject1": Task, ... } }
    Return unified { "task_name": Task } dictionary.
    """
    flat = {}
    for k, v in task_dict.items():
        if isinstance(k, str):
            flat[k] = v
        elif isinstance(k, tasks.ConfigurableGroup):
            for subname, subtask in v.items():
                flat[subname] = subtask
    return flat


def to_json_safe(obj):
    """Convert object to JSON-safe type (ensure dict keys are str; convert tensor/np to basic types)"""
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
# Chat Template Handling: Templates for different models
# ----------------------------
def get_system_prompt_and_template_for_model(model_arg_string: str):
    """
    Return appropriate system prompt and whether to use chat template based on model
    Returns: (system_prompt, use_chat_template)
    """
    model_lower = model_arg_string.lower()
    
    # Llama 3.1 and 3.2 series
    if any(x in model_lower for x in ["llama-3.1", "llama-3.2", "meta-llama-3"]):
        system_prompt = "You are a helpful assistant."
        return system_prompt, True
    
    # Qwen 3 series
    elif any(x in model_lower for x in ["qwen3", "qwen/qwen2.5"]):
        system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        return system_prompt, True
    
    # Other models don't use chat template
    else:
        return None, False

def apply_model_chat_template(text: str, tokenizer, model_arg_string: str, system_prompt: str = None):
    """
    Apply appropriate chat template for different models
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        return text
    
    model_lower = model_arg_string.lower()
    
    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})
    
    try:
        # Llama 3.x series uses official template
        if any(x in model_lower for x in ["llama-3.1", "llama-3.2", "meta-llama-3"]):
            return tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        
        # Qwen 3.x series uses official template
        elif any(x in model_lower for x in ["qwen3", "qwen/qwen2.5"]):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        # Default handling
        else:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False, 
                add_generation_prompt=True
            )
            
    except Exception as e:
        print(f"[WARNING] Chat template application failed: {e}, using original text")
        return text

# ----------------------------
# Load hypers.json (model -> multiple cache_size hyperparameters)
# ----------------------------
def load_hypers(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------
# monkey-patch: Replace attention by model (consistent with existing code)
# ----------------------------
def initialize_my_model(cache_config: ADCacheConfig, model_arg_string: str):
    if cache_config.compress:
        if "Llama-3" in model_arg_string or "Meta-Llama-3" in model_arg_string:
            from akcb.model.modle_llama import replace_llama3_attn
            print("[akcb] replace_llama3_attn()")
            replace_llama3_attn()
        elif "Qwen3" in model_arg_string or "Qwen/" in model_arg_string:
            from akcb.model.modify_qwen3 import replace_qwen3_attn
            print("[akcb] replace_qwen3_attn()")
            replace_qwen3_attn()

# ----------------------------
# Build ADCacheConfig (using hyperparameters)
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
# Telemetry wrapper: throughput and memory peak/average
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
        self.mem_samples_alloc = []
        self.mem_samples_reserved = []
        # Per-task TPS statistics
        self.task_stats = {}  # task_name -> {"prompt_tokens", "generated_tokens", "ll_tokens", "wall_time"}
        self.current_task = None  # Currently evaluating task name
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    # Transparent properties
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

    def _sample_mem(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.mem_samples_alloc.append(torch.cuda.memory_allocated())
            self.mem_samples_reserved.append(torch.cuda.memory_reserved())
    
    def set_current_task(self, task_name: str):
        """Set the currently evaluating task name"""
        self.current_task = task_name
        if task_name not in self.task_stats:
            self.task_stats[task_name] = {
                "prompt_tokens": 0,
                "generated_tokens": 0, 
                "ll_tokens": 0,
                "wall_time": 0.0
            }

    # Three interface interceptions
    def loglikelihood(self, requests):
        t0 = time.perf_counter()
        out = self.base.loglikelihood(requests)
        dt = time.perf_counter() - t0
        self.wall_total += dt; self.wall_ll += dt
        print(f"requests: {len(requests)}, ll_wall_s: {dt:.4f}, ll_tps: {len(requests)/max(dt,1e-9):.2f}")
        
        # Statistics for current task tokens
        task_tokens = 0
        if self.tokenizer is not None:
            for inst in requests:
                (ctx, cont) = inst.args
                prompt_tok = self._toklen(ctx)
                ll_tok = self._toklen(cont)
                self.prompt_tokens += prompt_tok
                self.ll_tokens += ll_tok
                task_tokens += prompt_tok + ll_tok
        
        # Update current task statistics
        if self.current_task:
            self.task_stats[self.current_task]["wall_time"] += dt
            if self.tokenizer is not None:
                for inst in requests:
                    (ctx, cont) = inst.args
                    self.task_stats[self.current_task]["prompt_tokens"] += self._toklen(ctx)
                    self.task_stats[self.current_task]["ll_tokens"] += self._toklen(cont)
        
        self._sample_mem()
        return out

    def loglikelihood_rolling(self, requests):
        t0 = time.perf_counter()
        out = self.base.loglikelihood_rolling(requests)
        dt = time.perf_counter() - t0
        self.wall_total += dt
        print(f"requests: {len(requests)}, ll_wall_s: {dt:.4f}, ll_tps: {len(requests)/max(dt,1e-9):.2f}")
        
        if self.tokenizer is not None:
            for inst in requests:
                ctx = inst.args
                ll_tok = self._toklen(ctx)
                self.ll_tokens += ll_tok
        
        # Update current task statistics
        if self.current_task:
            self.task_stats[self.current_task]["wall_time"] += dt
            if self.tokenizer is not None:
                for inst in requests:
                    ctx = inst.args
                    self.task_stats[self.current_task]["ll_tokens"] += self._toklen(ctx)
        
        self._sample_mem()
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
        
        # Update current task statistics
        if self.current_task:
            self.task_stats[self.current_task]["wall_time"] += dt
            if self.tokenizer is not None:
                for inst, gen in zip(requests, outs):
                    (ctx, gen_kwargs) = inst.args
                    self.task_stats[self.current_task]["prompt_tokens"] += self._toklen(ctx)
                    self.task_stats[self.current_task]["generated_tokens"] += self._toklen(gen)
        
        self._sample_mem()
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
        
        # Calculate TPS for each task
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
            "memory_allocated": stat(self.mem_samples_alloc),
            "memory_reserved": stat(self.mem_samples_reserved),
            "task_tps": task_tps,  # New: Per-task TPS statistics
        }

# ----------------------------
# TTFT: Sample N items per task, generate 1 token each, measure first token latency
# ----------------------------
def measure_ttft_per_task(hf_model, tokenizer, task, n_samples=8, apply_chat_template=True, system_prompt=None, model_arg_string=""):
    """
    Returns: {'count', 'lat_ms': {avg, p50, max, min, std}}
    One by one with batch=1 to ensure accurate TTFT.
    """
    device = next(hf_model.parameters()).device
    lat = []
    # Get document iterator
    if task.has_validation_docs():
        docs_iter = task.validation_docs()
    elif task.has_test_docs():
        docs_iter = task.test_docs()
    else:
        return {"count": 0, "lat_ms": {"avg": None, "p50": None, "max": None, "min": None, "std": None}}

    for i, doc in enumerate(docs_iter):
        if i >= n_samples: break
        prompt = task.doc_to_text(doc)

        if apply_chat_template:
            text = apply_model_chat_template(prompt, tokenizer, model_arg_string, system_prompt)
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt").to(device)
        torch.cuda.synchronize(device) if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = hf_model.generate(
                **inputs, 
                do_sample=False, 
                max_new_tokens=1,
                pad_token_id = tokenizer.pad_token_id if tokenizer is not None else 0,
            )
        torch.cuda.synchronize(device) if torch.cuda.is_available() else None
        dt = (time.perf_counter() - t0) * 1000.0
        lat.append(dt)

    if not lat:
        return {"count": 0, "lat_ms": {"avg": None, "p50": None, "max": None, "min": None, "std": None}}

    import numpy as np
    arr = np.array(lat, dtype=np.float64)
    return {
        "count": len(lat),
        "lat_ms": {
            "avg": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "max": float(arr.max()),
            "min": float(arr.min()),
            "std": float(arr.std()),
        }
    }

# ----------------------------
# Evaluate once (single model × single hyperparameter set × task list)
# ----------------------------
def eval_once(
    model_backend: str,
    model_args: str,
    tasks_list: List[str],
    hypers: Dict[str, Any],
    quant_type="mix",
    limit=0,
    apply_chat_template=False,
    fewshot_as_multiturn=False,
    n_ttft_samples=8,
    tps_sampling=False,
    sampling_ratio=0.2) -> Dict[str, Any]:

    # 1) AKCB configuration + replacement
    cache_config = build_cache_config(hypers, quant_type=quant_type)
    initialize_my_model(cache_config, model_args)

    # 2) Construct lm-eval model
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

    # 4) Wrap Telemetry
    lm_t = LMWithTelemetry(lm)

    # 5) Task management
    task_manager = tasks.TaskManager()
    task_dict = tasks.get_task_dict(tasks_list, task_manager)
    flat_task_dict = flatten_task_dict(task_dict)

    # 5.5) Get model-specific system prompt and chat template settings
    system_prompt, use_chat_template = get_system_prompt_and_template_for_model(model_args)
    final_apply_chat_template = apply_chat_template and use_chat_template
    
    if final_apply_chat_template:
        print(f"[CHAT_TEMPLATE] Using chat template, system prompt: {system_prompt}")
    else:
        print(f"[CHAT_TEMPLATE] Not using chat template")late")

    # 6) TTFT: Measure first token latency for samples in each task (batch=1, max_new_tokens=1)
    ttft = {}
    for name, t in flat_task_dict.items():
        try:
            ttft[name] = measure_ttft_per_task(hf_model, tokenizer, t, n_samples=n_ttft_samples,
                                               apply_chat_template=final_apply_chat_template, 
                                               system_prompt=system_prompt,
                                               model_arg_string=model_args)
        except Exception as e:
            print(f"[TTFT ERROR] task {name}: {e}")
    limit = limit if limit > 0 else None
    
    # TPS sampling mode: Calculate sampling count for each task
    task_limits = {}
    if tps_sampling:
        import random
        random.seed(42)  # Fix random seed for reproducibility
        
        for task_name, task_obj in flat_task_dict.items():
            # Get total number of samples for the task
            total_samples = 0
            try:
                if task_obj.has_test_docs():
                    # For large tasks, avoid loading all data into memory
                    docs_iter = task_obj.test_docs()
                    if hasattr(docs_iter, '__len__'):
                        total_samples = len(docs_iter)
                    else:
                        # If iterator, count elements (may be slow)
                        total_samples = sum(1 for _ in docs_iter)
                elif task_obj.has_validation_docs():
                    docs_iter = task_obj.validation_docs()
                    if hasattr(docs_iter, '__len__'):
                        total_samples = len(docs_iter)
                    else:
                        total_samples = sum(1 for _ in docs_iter)
            except Exception as e:
                print(f"[WARNING] Cannot get sample count for task {task_name}: {e}")
                total_samples = 1000  # Default assumption
            
            if total_samples > 0:
                sampled_count = max(1, int(total_samples * sampling_ratio))
                task_limits[task_name] = sampled_count
                print(f"[TPS_SAMPLING] {task_name}: {sampled_count}/{total_samples} samples ({sampling_ratio:.1%})")
            else:
                task_limits[task_name] = limit
    
    # 7) Formal evaluation - Evaluate task by task to calculate TPS for each task
    all_results = {}
    for task_name, task_obj in flat_task_dict.items():
        print(f"[EVAL] Starting task: {task_name}")
        lm_t.set_current_task(task_name)
        
        # Create task_dict for single task
        single_task_dict = {task_name: task_obj}
        
        # Determine limit for this task
        current_limit = task_limits.get(task_name, limit) if tps_sampling else limit
        
        task_results = evaluator.evaluate(
            lm=lm_t,
            task_dict=single_task_dict,
            limit=current_limit,
            log_samples=False,
            apply_chat_template=final_apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
        )
        
        # Merge results
        if "results" in task_results:
            all_results.update(task_results["results"])
    
    # Construct final result format
    results = {"results": all_results}

    # 8) Telemetry summary (throughput + memory average/peak/variation)
    telemetry = lm_t.summary()

    return {
        "results": results.get("results", {}),
        "configs": {
            "model_args": model_args,
            "hypers": hypers,
            "apply_chat_template": final_apply_chat_template,
            "system_prompt": system_prompt if final_apply_chat_template else None,
            "tps_sampling": tps_sampling,
            "sampling_ratio": sampling_ratio if tps_sampling else None,
            "task_limits": task_limits if tps_sampling else None,
        },
        "telemetry": telemetry,
        "ttft": ttft,
    }

# ----------------------------
# Main entry: Read hypers.json, iterate (model × hyperparameter set) evaluation
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypers", type=str, default="config/hypers.json",
                        help="Hyperparameter file path (JSON), listing multiple cache_size groups by model")sting multiple cache_size groups by model")
    parser.add_argument("--tasks", type=str, default="gsm8k,mmlu,commonsense_qa",
                        help="Comma-separated lm-eval task list")val task list")
    parser.add_argument("--apply_chat_template", action="store_true",
                        help="Recommended for instruct models")truct models")
    parser.add_argument("--fewshot_as_multiturn", action="store_true",
                        help="Fewshot as multi-turn dialogue (only valid for apply_chat_template)")for apply_chat_template)")
    # parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation, 0=auto")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--outdir", type=str, default="eval_runs")
    parser.add_argument("--n_ttft", type=int, default=8, help="每个任务用于 TTFT 的样本数")
    parser.add_argument("--quant_type", type=str, default="mix")
    parser.add_argument("--limit", type=int, default=0, help="Upper limit of evaluation samples per task")f evaluation samples per task")
    parser.add_argument("--tps_sampling", action="store_true", 
                        help="Enable TPS sampling mode: 20%% sampling per task, output to tps subfolder") per task, output to tps subfolder")
    parser.add_argument("--sampling_ratio", type=float, default=0.2,
                        help="Sampling ratio for TPS sampling mode, default 0.2 (20%%)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    # TPS 采样模式：创建单独的输出目录
    if args.tps_sampling:
        tps_outdir = os.path.join(args.outdir, "tps")
        os.makedirs(tps_outdir, exist_ok=True)
        print(f"[TPS_SAMPLING] Enable TPS sampling mode, sampling ratio: {args.sampling_ratio:.1%}")
        print(f"[TPS_SAMPLING] Output directory: {tps_outdir}")
    else:
        tps_outdir = args.outdir
    
    hypers_all = load_hypers(args.hypers)

    MODEL_BACKEND = "hf"

    all_rows = []  # 用于导出 CSV
    all_ttft_rows = [] # 用于导出 TTFT CSV
    for model_id, entry in hypers_all.items():
        hypers_list = entry["hypers"]
        # 构造 model_args
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
                apply_chat_template=args.apply_chat_template,
                fewshot_as_multiturn=args.fewshot_as_multiturn,
                n_ttft_samples=args.n_ttft,
                tps_sampling=args.tps_sampling,
                sampling_ratio=args.sampling_ratio
            )

            # 保存 JSON
            # print(f"[RESULT] out:{out}")
            tag = f"{model_id.replace('/','_')}__cs{h['cache_size']}"
            if args.tps_sampling:
                tag += f"_tps{int(args.sampling_ratio*100)}pct"
            
            json_path = os.path.join(tps_outdir, f"{tag}.json")
            safe_out = to_json_safe(out)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(safe_out, f, ensure_ascii=False, indent=2)
            print(f"[SAVED] {json_path}")

            # 摘要行（便于汇总对比）
            # 取每任务一个主指标
            for task_name, metrics in safe_out["results"].items():
                # 选主指标（优先 acc_norm/acc/exact_match/f1）
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
                    "cache_size": h["cache_size"],
                    "window_size": h.get("window_size", 32),
                    "tau1": h["tau1"], "tau2": h["tau2"], "tau3": h["tau3"], "gamma": h["gamma"],
                    "task": task_name, "metric": mname, "score": mval,
                    # 整体吞吐
                    "gen_tokens_per_s": safe_out["telemetry"]["gen_tokens_per_s"],
                    "effective_tokens_per_s": safe_out["telemetry"]["effective_tokens_per_s"],
                    # 任务专属 TPS
                    "task_tokens_per_s": safe_out["telemetry"]["task_tps"].get(task_name, {}).get("tokens_per_s", 0.0),
                    "task_prompt_tokens": safe_out["telemetry"]["task_tps"].get(task_name, {}).get("prompt_tokens", 0),
                    "task_generated_tokens": safe_out["telemetry"]["task_tps"].get(task_name, {}).get("generated_tokens", 0),
                    "task_ll_tokens": safe_out["telemetry"]["task_tps"].get(task_name, {}).get("ll_tokens", 0),
                    "task_wall_time_s": safe_out["telemetry"]["task_tps"].get(task_name, {}).get("wall_time_s", 0.0),
                    # 显存
                    "mem_alloc_avg_GB": safe_out["telemetry"]["memory_allocated"]["avg_GB"],
                    "mem_alloc_max_GB": safe_out["telemetry"]["memory_allocated"]["max_GB"],
                    "mem_alloc_std_GB": safe_out["telemetry"]["memory_allocated"]["std_GB"],
                    "mem_res_avg_GB": safe_out["telemetry"]["memory_reserved"]["avg_GB"],
                    "mem_res_max_GB": safe_out["telemetry"]["memory_reserved"]["max_GB"],
                    "mem_res_std_GB": safe_out["telemetry"]["memory_reserved"]["std_GB"],
                }
                # TTFT
                if task_name in safe_out["ttft"]:
                    row.update({
                        "ttft_avg_ms": safe_out["ttft"][task_name]["lat_ms"]["avg"] if "lat_ms" in safe_out["ttft"][task_name] else None,
                        "ttft_p50_ms": safe_out["ttft"][task_name]["lat_ms"]["p50"] if "lat_ms" in safe_out["ttft"][task_name] else None,
                        "ttft_max_ms": safe_out["ttft"][task_name]["lat_ms"]["max"] if "lat_ms" in safe_out["ttft"][task_name] else None,
                        "ttft_min_ms": safe_out["ttft"][task_name]["lat_ms"]["min"] if "lat_ms" in safe_out["ttft"][task_name] else None,
                        "ttft_std_ms": safe_out["ttft"][task_name]["lat_ms"]["std"] if "lat_ms" in safe_out["ttft"][task_name] else None,
                    })
                all_rows.append(row)
                

    # 导出 CSV 汇总
    import csv
    csv_filename = "tps_summary.csv" if args.tps_sampling else "summary.csv"
    csv_path = os.path.join(tps_outdir, csv_filename)
    if all_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[SAVED] {csv_path}")
        
    # TPS 采样模式：额外保存 TPS 专用的详细统计文件
    if args.tps_sampling and all_rows:
        tps_detail_rows = []
        for row in all_rows:
            task_name = row["task"]
            # 查找对应的详细 TPS 数据
            for model_id, entry in hypers_all.items():
                for h in entry["hypers"]:
                    if (row["model"] == model_id and 
                        row["cache_size"] == h["cache_size"] and
                        row["tau1"] == h["tau1"] and 
                        row["tau2"] == h["tau2"] and
                        row["tau3"] == h["tau3"] and
                        row["gamma"] == h["gamma"]):
                        
                        # 构造详细的 TPS 行
                        tps_row = {
                            "model": model_id,
                            "cache_size": h["cache_size"],
                            "window_size": h.get("window_size", 32),
                            "tau1": h["tau1"], "tau2": h["tau2"], "tau3": h["tau3"], "gamma": h["gamma"],
                            "task": task_name,
                            "task_tokens_per_s": row["task_tokens_per_s"],
                            "task_prompt_tokens": row["task_prompt_tokens"],
                            "task_generated_tokens": row["task_generated_tokens"],
                            "task_ll_tokens": row["task_ll_tokens"],
                            "task_total_tokens": row["task_prompt_tokens"] + row["task_generated_tokens"] + row["task_ll_tokens"],
                            "task_wall_time_s": row["task_wall_time_s"],
                            "sampling_ratio": args.sampling_ratio,
                        }
                        tps_detail_rows.append(tps_row)
                        break
        
        # 保存 TPS 详细文件
        tps_detail_path = os.path.join(tps_outdir, "tps_detailed_stats.csv")
        if tps_detail_rows:
            with open(tps_detail_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(tps_detail_rows[0].keys()))
                writer.writeheader()
                writer.writerows(tps_detail_rows)
            print(f"[SAVED] TPS详细统计: {tps_detail_path}")
    
if __name__ == "__main__":
    main()
