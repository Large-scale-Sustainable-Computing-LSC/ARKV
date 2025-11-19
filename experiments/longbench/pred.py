import os
from datasets import load_dataset
import torch
import json
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import time
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

from akcb.config import ADCacheConfig

# ====== TTFT: First Token Timer ======
class FirstTokenTimer(StoppingCriteria):
    def __init__(self, start_len: int):
        super().__init__()
        self.start_len = start_len
        self.t0 = None
        self.first_token_ms = None
    def start(self):
        self.t0 = time.perf_counter()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Record first token time when sequence length first exceeds start_len (don't actually stop)
        if self.first_token_ms is None and input_ids.shape[1] > self.start_len:
            self.first_token_ms = (time.perf_counter() - self.t0) * 1000.0
        return False

# ====== Telemetry: Track throughput/memory/TTFT ======
class Telemetry:
    def __init__(self, device):
        self.prompt_tokens = 0
        self.gen_tokens = 0
        self.wall_gen_s = 0.0
        self.mem_alloc = []
        self.mem_res = []
        self.ttft_ms = []
        self.device = device

    def sample_mem(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            self.mem_alloc.append(torch.cuda.memory_allocated(self.device))
            self.mem_res.append(torch.cuda.memory_reserved(self.device))

    def add_prompt(self, n): self.prompt_tokens += int(n)
    def add_gen(self, n): self.gen_tokens += int(n)
    def add_time(self, dt): self.wall_gen_s += float(dt)
    def add_ttft(self, ms):
        if ms is not None:
            self.ttft_ms.append(float(ms))

    def summary(self):
        import numpy as np
        eps = 1e-9
        gen_tps = self.gen_tokens / max(self.wall_gen_s, eps)
        eff_tps = (self.prompt_tokens + self.gen_tokens) / max(self.wall_gen_s, eps)
        def mstat(arr):
            if not arr: return {"avg_GB": None, "max_GB": None, "std_GB": None}
            x = torch.tensor(arr, dtype=torch.float64); toGB = 1024**3
            return {
                "avg_GB": round((x.mean().item())/toGB, 3),
                "max_GB": round((x.max().item())/toGB, 3),
                "std_GB": round((x.std(unbiased=False).item())/toGB, 3),
            }
        def tstat(arr):
            if not arr: return {"avg": None, "p50": None, "max": None, "min": None, "std": None}
            a = np.array(arr, dtype=np.float64)
            return {
                "avg": float(a.mean()), "p50": float(np.percentile(a, 50)),
                "max": float(a.max()),  "min": float(a.min()),
                "std": float(a.std())
            }
        return {
            "gen_tokens_per_s": round(gen_tps, 2),
            "effective_tokens_per_s": round(eff_tps, 2),
            "memory_allocated": mstat(self.mem_alloc),
            "memory_reserved":  mstat(self.mem_res),
            "ttft_ms": tstat(self.ttft_ms),
            "ttft_ms_samples": self.ttft_ms,   # Sample-level list
            "gen_tokens": int(self.gen_tokens),
            "prompt_tokens": int(self.prompt_tokens),
            "gen_wall_s": round(self.wall_gen_s, 4),
        }

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Meta-Llama-3.1-8B-Instruct", 
                        choices=["Qwen3-4B-Instruct-2507", 
                                 "Meta-Llama-3.1-8B-Instruct",
                                 "Llama-3.2-3B-Instruct",
                                 "Qwen3-8B",])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--compress', action='store_true', help="Comrpess kv cache with")
    # parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cache_size', type=int, default=1024)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--tau1', type=float, default=1.0)
    parser.add_argument('--tau2', type=float, default=1.0)
    parser.add_argument('--tau3', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=200.0)
    parser.add_argument('--quant_type', type=str, default="mix", choices=["mix", "quant", "origin", "window"])
    parser.add_argument('--outdir', type=str, default="pred",help="Output directory")
    return parser.parse_args(args)

def init_cache_config(args):
    cache_config = ADCacheConfig(args)
    if cache_config.compress:
        model_name = args.model
        model2hypers = json.load(open("config/model2hypers.json", "r"))
        if model_name in model2hypers:
            hypers = model2hypers[model_name][f"{cache_config.cache_size}"]
            cache_config.tau1 = hypers['tau1']
            cache_config.tau2 = hypers['tau2']
            cache_config.tau3 = hypers['tau3']
            cache_config.gamma = hypers['gamma']
            print(f"Using hyperparameters for {model_name}: tau1 {cache_config.tau1}, "
                  f"tau2 {cache_config.tau2}, tau3 {cache_config.tau3}, "
                  f"gamma {cache_config.gamma}, quant_type {cache_config.quant_type}")
    return cache_config

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    # if "chatglm3" in model_name:
    #     prompt = tokenizer.build_chat_input(prompt)
    # elif "chatglm" in model_name:
    #     prompt = tokenizer.build_prompt(prompt)
    # elif "longchat" in model_name or "vicuna" in model_name:
    #     from fastchat.model import get_conversation_template
    #     conv = get_conversation_template("vicuna")
    #     conv.append_message(conv.roles[0], prompt)
    #     conv.append_message(conv.roles[1], None)
    #     prompt = conv.get_prompt()
    # elif "llama2" in model_name:
    #     prompt = f"[INST]{prompt}[/INST]"
    # elif "xgen" in model_name:
    #     header = (
    #         "A chat between a curious human and an artificial intelligence assistant. "
    #         "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
    #     )
    #     prompt = header + f" ### Human: {prompt}\n###"
    # elif "internlm" in model_name:
    #     prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    
    if "llama3" in model_name:
        print("======== llama3 build chat ========")
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(rank, world_size, data, max_length, 
             max_gen, prompt_format, dataset, device, 
             model_name, model2path, out_path, cache_config
):
    print(f"Process dataset {dataset}, rank {rank}")
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, cache_config)
    
    tele = Telemetry(device)
    
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        tele.add_prompt(context_length)
        
        # ==== TTFT Timer ====
        tt = FirstTokenTimer(start_len=context_length)
        sc = StoppingCriteriaList([tt])
        tt.start()
        
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            t0 = time.perf_counter()
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                stopping_criteria=sc,
            )[0]
            torch.cuda.synchronize(device) if torch.cuda.is_available() else None
            dt = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                pad_token_id = tokenizer.eos_token_id,
                stopping_criteria=sc,
            )[0]
            torch.cuda.synchronize(device) if torch.cuda.is_available() else None
            dt = time.perf_counter() - t0
        
        tele.add_time(dt)                 # NEW
        tele.add_ttft(tt.first_token_ms) # NEW
        tele.sample_mem()  
        gen_tokens = int(output.shape[-1] - context_length)
        tele.add_gen(gen_tokens)
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
        if cache_config.compress:
            cache_config.refresh_model_settings()
    
    tele_path = out_path.replace(".jsonl", f".telemetry_rank{rank}.json")
    with open(tele_path, "w", encoding="utf-8") as f:
        json.dump(tele.summary(), f, ensure_ascii=False, indent=2)
    # dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device, cache_config):
    if cache_config.compress:
        if "Llama-3" in model_name:
            from akcb.model.modle_llama import replace_llama3_attn
            print("======== llama3 replace flashllama attn ========")
            replace_llama3_attn()
        elif "Qwen3" in model_name:
            from akcb.model.modify_qwen3 import replace_qwen3_attn
            print("======== qwen3 replace flashllama attn ========")
            replace_qwen3_attn()
        # tokenizer = LlamaTokenizer.from_pretrained(path)
        # model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype,
        attn_implementation="flash_attention_2"
    ).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    config = AutoConfig.from_pretrained(path)
    if hasattr(config, 'num_hidden_layers'):
        layers = config.num_hidden_layers
    cache_config.update_layer_num(layers)
    if cache_config.compress:
        for i in range(layers):
            model.model.layers[i].self_attn.config.cache_config = cache_config

    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    cache_config = init_cache_config(args)
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
        #             "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
        #             "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        datasets = ["narrativeqa"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    outdir = args.outdir
    # predict on each dataset
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(f"{outdir}_e") and args.e:
        os.makedirs(f"{outdir}_e")
    for dataset in datasets:
        cache_size = cache_config.cache_size if cache_config.compress else "full"
        cache_type = cache_config.quant_type if cache_config.compress else "full"
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test[:10]')
            if not os.path.exists(f"{outdir}_e/{model_name}/{cache_size}_{cache_type}"):
                os.makedirs(f"{outdir}_e/{model_name}/{cache_size}_{cache_type}")
            out_path = f"{outdir}_e/{model_name}/{cache_size}_{cache_type}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test[:10]')
            if not os.path.exists(f"{outdir}/{model_name}/{cache_size}_{cache_type}"):
                os.makedirs(f"{outdir}/{model_name}/{cache_size}_{cache_type}")
            out_path = f"{outdir}/{model_name}/{cache_size}_{cache_type}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
                        max_gen, prompt_format, dataset, device, model_name, model2path, out_path, cache_config))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        
        # ====== Aggregate telemetry from all ranks, write one CSV row ======
        import glob, numpy as np, csv, json

        tele_files = sorted(glob.glob(out_path.replace(".jsonl", ".telemetry_rank*.json")))
        tele_all = []
        for fp in tele_files:
            with open(fp, "r", encoding="utf-8") as f:
                tele_all.append(json.load(f))

        # Throughput (keep original logic)
        gen_tokens    = sum(t["gen_tokens"] for t in tele_all)
        prompt_tokens = sum(t["prompt_tokens"] for t in tele_all)
        wall_gen_s    = sum(t["gen_wall_s"] for t in tele_all)  # Or use max(...) for more conservative estimate
        gen_tps = gen_tokens / max(wall_gen_s, 1e-9)
        eff_tps = (gen_tokens + prompt_tokens) / max(wall_gen_s, 1e-9)

        # Memory (keep original logic)
        def agg_mem(key):
            vals_avg = [t[key]["avg_GB"] for t in tele_all if t[key]["avg_GB"] is not None]
            vals_max = [t[key]["max_GB"] for t in tele_all if t[key]["max_GB"] is not None]
            vals_std = [t[key]["std_GB"] for t in tele_all if t[key]["std_GB"] is not None]
            return (
                float(np.mean(vals_avg)) if vals_avg else None,
                float(np.max(vals_max)) if vals_max else None,
                float(np.mean(vals_std)) if vals_std else None
            )
        mem_alloc_avg, mem_alloc_max, mem_alloc_std = agg_mem("memory_allocated")
        mem_res_avg,   mem_res_max,   mem_res_std   = agg_mem("memory_reserved")

        # Key: Merge sample-level TTFT from all ranks before computing statistics
        ttft_samples = []
        for t in tele_all:
            ttft_samples += t.get("ttft_ms_samples", []) or []

        if ttft_samples:
            a = np.array(ttft_samples, dtype=np.float64)
            ttft_avg = float(a.mean())
            ttft_p50 = float(np.percentile(a, 50))
            ttft_max = float(a.max())
            ttft_min = float(a.min())
            ttft_std = float(a.std())
        else:
            ttft_avg = ttft_p50 = ttft_max = ttft_min = ttft_std = None

        # Write CSV: Add five TTFT fields to header
        directory = f"{outdir}" if not args.e else f"{outdir}_e"
        summary_csv = f"{directory}/{model_name}/{cache_size}_{cache_type}/summary.csv"
        header = [
            "model","dataset","cache_size","window_size","tau1","tau2","tau3","gamma","quant_type",
            "gen_tokens_per_s","effective_tokens_per_s",
            "mem_alloc_avg_GB","mem_alloc_max_GB","mem_alloc_std_GB",
            "mem_res_avg_GB","mem_res_max_GB","mem_res_std_GB",
            "ttft_avg_ms","ttft_p50_ms","ttft_max_ms","ttft_min_ms","ttft_std_ms"  # New fields
        ]
        row = {
            "model": model_name,
            "dataset": dataset,
            "cache_size": cache_config.cache_size if cache_config.compress else "full",
            "window_size": cache_config.window_size,
            "tau1": cache_config.tau1, "tau2": cache_config.tau2, "tau3": cache_config.tau3, "gamma": cache_config.gamma,
            "quant_type": cache_config.quant_type if cache_config.compress else "full",
            "gen_tokens_per_s": round(gen_tps, 2),
            "effective_tokens_per_s": round(eff_tps, 2),
            "mem_alloc_avg_GB": mem_alloc_avg, "mem_alloc_max_GB": mem_alloc_max, "mem_alloc_std_GB": mem_alloc_std,
            "mem_res_avg_GB": mem_res_avg,   "mem_res_max_GB": mem_res_max,   "mem_res_std_GB": mem_res_std,
            "ttft_avg_ms": ttft_avg, "ttft_p50_ms": ttft_p50, "ttft_max_ms": ttft_max, "ttft_min_ms": ttft_min, "ttft_std_ms": ttft_std,
        }
        need_header = not os.path.exists(summary_csv)
        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=header)
            if need_header: wr.writeheader()
            wr.writerow(row)
        print(f"[SUMMARY] {row}")