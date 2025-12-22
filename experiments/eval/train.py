# optuna_tune_adkv.py
import os
import json
import math
import argparse
from typing import Dict, Any, List, Tuple

import optuna
import torch

import lm_eval
import lm_eval.api as api
import lm_eval.evaluator as evaluator
from lm_eval import tasks

# === Your config class ===
from arkv.config import ADCacheConfig

# ----------------------------
# Utility: metric aggregation (pick one “primary metric” per task, then average)
# ----------------------------
PRIMARY_METRICS_ORDER = [
    "acc_norm", "acc", "exact_match", "f1",
    "qa_f1_score", "qa_f1_zh_score",
    "rouge_score", "rouge_zh_score",
    "retrieval_score", "retrieval_zh_score",
    "code_sim_score", "classification_score", "count_score",
]

def pick_primary_metric(metrics: Dict[str, float]) -> Tuple[str, float]:
    """
    lm-eval returns per-task metric keys typically shaped like 'acc,none' / 'exact_match,strict-match'.
    This function selects one primary metric based on a priority list.
    """
    # Extract the prefix before the comma: 'acc,none' -> 'acc'
    m2v: Dict[str, float] = {}
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            continue
        name = k.split(",")[0].strip()
        # If a task has multiple variants under the same name (e.g., strict vs flexible), take the max.
        m2v[name] = max(float(v), m2v.get(name, float("-inf")))
    for name in PRIMARY_METRICS_ORDER:
        if name in m2v:
            return name, float(m2v[name])
    # Fallback: pick any numeric metric
    if m2v:
        k0 = next(iter(m2v.keys()))
        return k0, float(m2v[k0])
    return "na", float("nan")

def aggregate_results(results_dict: Dict[str, Any]) -> float:
    """
    Aggregate lm-eval evaluate() results into a single score (uniform average across tasks).
    """
    task_scores: List[float] = []
    for task_name, metrics in results_dict.get("results", {}).items():
        _, val = pick_primary_metric(metrics)
        if math.isfinite(val):
            task_scores.append(val)
    return float(sum(task_scores) / len(task_scores)) if task_scores else float("nan")

# ----------------------------
# Monkey-patch entry point (decide by model repo/name)
# ----------------------------
def initialize_my_model(cache_config: ADCacheConfig, model_arg_string: str):
    # Keep the same heuristic as before: model arg string contains "Llama-3" or "Qwen3"
    if cache_config.compress:
        if "Llama-3" in model_arg_string:
            from arkv.model.modle_llama import replace_llama3_attn
            replace_llama3_attn()
        elif "Qwen3" in model_arg_string:
            from arkv.model.modify_qwen3 import replace_qwen3_attn
            replace_qwen3_attn()

# ----------------------------
# Build cache_config (from trial hyperparameters)
# ----------------------------
def build_cache_config(
    cache_size: int,
    window_size: int,
    tau1: float,
    tau2: float,
    tau3: float,
    gamma: float,
    quant_type: str = "mix",
    compress: bool = True,
) -> ADCacheConfig:
    class Args:
        # Keep fields aligned with your DummyArgs structure
        def __init__(self):
            self.model = "Meta-Llama-3.1-8B-Instruct"
            self.e = False
            self.compress = compress
            self.cache_size = cache_size
            self.window_size = window_size
            self.tau1 = tau1
            self.tau2 = tau2
            self.tau3 = tau3
            self.gamma = gamma
            self.quant_type = quant_type
    return ADCacheConfig(Args())

# ----------------------------
# Single run: given model + cache_config, return evaluation output
# ----------------------------
def run_once(
    model_backend: str,
    model_args: str,
    tasks_list: List[str],
    limit: int,
    cache_config: ADCacheConfig,
) -> Dict[str, Any]:
    # Apply monkey-patch first
    initialize_my_model(cache_config, model_args)

    # Build lm-eval model
    lm = api.registry.get_model(model_backend).create_from_arg_string(
        model_args,
        {
            "batch_size": None,
            "max_batch_size": None,
            "device": None,
        },
    )
    print(f"[INFO] Built model: {lm}")

    # Inject cache_config
    config = getattr(lm, "config", None)
    if hasattr(config, "num_hidden_layers"):
        layers = int(config.num_hidden_layers)
        cache_config.update_layer_num(layers)
        if cache_config.compress:
            # Llama/Qwen：lm.model.model.layers[i].self_attn
            for i in range(layers):
                try:
                    lm.model.model.layers[i].self_attn.config.cache_config = cache_config
                except Exception as e:
                    raise RuntimeError(f"Failed to inject cache_config at layer {i}: {e}")
    else:
        print("[WARN] Can't find num_hidden_layers on config; skip layer injection update_layer_num")

    # Build task dict
    task_manager = tasks.TaskManager()
    task_dict = tasks.get_task_dict(tasks_list, task_manager)

    # Evaluate
    eout = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        log_samples=False,
        # Note: if you need chat templates, switch to simple_evaluate(apply_chat_template=True)
    )
    return eout

# ----------------------------
# Optuna objective (search tau1, tau2, tau3, gamma)
# ----------------------------
def make_objective(
    model_backend: str,
    model_args: str,
    tasks_list: List[str],
    limit: int,
    cache_size: int,
    window_size: int,
    quant_type: str = "mix",
):
    def objective(trial: optuna.Trial) -> float:
        # Search space (adjust ranges as needed)
        tau1 = trial.suggest_float("tau1", 0.1, 5.0)
        tau2 = trial.suggest_float("tau2", 0.1, 5.0)
        tau3 = trial.suggest_float("tau3", 0.1, 5.0)
        gamma = trial.suggest_float("gamma", 100.0, 300.0)

        cache_config = build_cache_config(
            cache_size=cache_size,
            window_size=window_size,
            tau1=tau1, tau2=tau2, tau3=tau3, gamma=gamma,
            quant_type=quant_type,
            compress=True,
        )

        results = run_once(
            model_backend=model_backend,
            model_args=model_args,
            tasks_list=tasks_list,
            limit=limit,
            cache_config=cache_config,
        )

        score = aggregate_results(results)
          # Store details on the trial for later analysis
        trial.set_user_attr("raw_results", json.dumps(results.get("results", {}), ensure_ascii=False))
        trial.set_user_attr("cache_size", cache_size)
        trial.set_user_attr("model_args", model_args)
        print(f"[TRIAL {trial.number}] score={score:.4f}  (cache_size={cache_size})  "
              f"tau1={tau1:.3f} tau2={tau2:.3f} tau3={tau3:.3f} gamma={gamma:.3f}")
        return score
    return objective

# ----------------------------
# Main program: create separate studies for each (model × cache_size) combo
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, default="gsm8k,mmlu,commonsense_qa,longbench",
                        help="Comma-separated task names (must be supported by lm-eval)")
    parser.add_argument("--limit", type=str, default="10", help="Samples per task (small sample for speed)")
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--quant_type", type=str, default="mix")
    parser.add_argument("--outdir", type=str, default="optuna_runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_size", type=str, default="1024")
    parser.add_argument("--model", type=str, 
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct", 
                        choices=[
                            "meta-llama/Meta-Llama-3.1-8B-Instruct",
                            "Qwen/Qwen3-4B-Instruct-2507",
                        ])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)

    # Model list (dtype/device can be adjusted as needed)
    MODEL_BACKEND = "hf"
    MODEL_ARGS_LIST = [
        "pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,dtype=bfloat16,device=cuda",
        "pretrained=Qwen/Qwen3-4B-Instruct-2507,dtype=bfloat16,device=cuda",
        "pretrained=meta-llama/Llama-3.2-3B-Instruct,dtype=bfloat16,device=cuda",
    ]
    CACHE_SIZES = [int(t.strip()) for t in args.cache_size.split(",") if t.strip()]

    tasks_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    limit = 10
    if args.limit.isdigit():
        limit = int(args.limit) if int(args.limit) > 1 else float(args.limit)

    # If you include code-execution tasks (e.g., HumanEval/MBPP), you must explicitly allow them (and run in a sandbox)
    # os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    # Also pass confirm_run_unsafe_code=True to evaluate/simple_evaluate (this script does not run such tasks by default)

    summaries = []

    for model_args in MODEL_ARGS_LIST:
        # Create an independent study per model × cache_size; optimize in the maximize direction
        for cs in CACHE_SIZES:
            study_name = f"adkv_tune__model={model_args.split(',')[0]}__cache={cs}"
            print("\n" + "="*80)
            print(f"[STUDY] {study_name}")
            print("="*80)

            study = optuna.create_study(direction="maximize", study_name=study_name)
            study.optimize(
                make_objective(
                    model_backend=MODEL_BACKEND,
                    model_args=model_args,
                    tasks_list=tasks_list,
                    limit=limit,
                    cache_size=cs,
                    window_size=args.window_size,
                    quant_type=args.quant_type,
                ),
                n_trials=args.n_trials,
                gc_after_trial=True,
            )

            best = {
                "study": study_name,
                "best_value": study.best_value,
                "best_params": study.best_trial.params,
                "cache_size": cs,
                "model_args": model_args,
            }
            summaries.append(best)

            # Save study results
            out_json = os.path.join(args.outdir, f"{study_name.replace('/','_')}.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({
                    "study": study_name,
                    "best_value": study.best_value,
                    "best_params": study.best_trial.params,
                    "all_trials": [
                        {"number": t.number, "value": t.value, "params": t.params,
                         "user_attrs": t.user_attrs}
                        for t in study.trials
                    ]
                }, f, ensure_ascii=False, indent=2)
            print(f"[SAVED] {out_json}")

    # Print summary
    print("\n=== SUMMARY (best of each combo) ===")
    for s in summaries:
        print(f"{s['study']}\n  best={s['best_value']:.4f}  params={s['best_params']}\n")

if __name__ == "__main__":
    main()
