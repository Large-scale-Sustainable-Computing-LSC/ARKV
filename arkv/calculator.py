from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math

def calculate_heavy_hitter(
    attention_scores: torch.Tensor, 
    gamma: float, 
    window_size: int,
    bsz: int,
    num_key_value_heads: int,
    num_key_value_groups: int,
):
    """
    Input:
        Calculate heavy hitter scores from attention scores.
        attention_scores: [B, H, Q, K] attention scores after softmax
        gamma: weight for variance
        window_size: int, size of the sliding window
        bsz: batch size
        num_key_value_heads: number of key/value heads
        num_key_value_groups: number of key/value groups
    Return:
        hh_score: [B, H, G, S-window_size] heavy hitter scores for each group
    """
    B, H, q_len, K = attention_scores.shape
    L = K - window_size
    if L <= 0:
        return attention_scores.new_empty(bsz, num_key_value_heads, 0)
    attn_mean = attention_scores.mean(dim=-2)
    if q_len > 1:
        attn_var = attention_scores.var(dim=-2)
        attn_cache = attn_mean + gamma * attn_var
    else:
        attn_cache = attn_mean
    attn_cache = attn_cache[:, :, :-window_size]
    attn_cache = F.avg_pool1d(attn_cache, kernel_size=5, padding=5//2, stride=1)
    attn_cache = attn_cache.reshape(bsz, num_key_value_heads, num_key_value_groups, -1)
    hh_score = attn_cache.mean(dim=-2)
    return hh_score


def calculate_entropy(attention_scores):
    attention_scores = attention_scores.to(torch.float32)
    entropy = -torch.sum(attention_scores * torch.log(attention_scores + 1e-10))  
    entropy= entropy.to(dtype=torch.float32)
    return entropy

def calculate_kurtosis(
    attn: torch.Tensor, 
    eps: float = 1e-8, 
    fisher: bool = False
) -> torch.Tensor:
    """
    Compute kurtosis on dim=-2 (LQ); consistent with your current variance dimension.
    attn: [B, H, LQ, LK]
    Returns: scalar (same aggregation method as var: sum over B/H/LK)
    """
    attn = attn.to(torch.float32)
    # Central moments on LQ dimension
    mean_q = attn.mean(dim=-2, keepdim=True)                 # [B,H,1,LK]
    delta  = attn - mean_q
    m2 = (delta.pow(2)).mean(dim=-2).clamp_min(eps)          # [B,H,LK]
    m4 = (delta.pow(4)).mean(dim=-2)                         # [B,H,LK]

    kurt = m4 / (m2 ** 2)                                    # Pearson kurtosis (normal=3)
    if fisher:
        kurt = kurt - 3.0                                    # Fisher excess kurtosis (normal=0)

    # Same aggregation method as your var: sum over B,H,LK -> scalar
    kurt_scalar = kurt.sum(0).sum(0).sum(0)
    return kurt_scalar

def disp_var_kurt_preference(
    attn: torch.Tensor,
    tau1: float,
    tau2: float,
    tau3: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute per-layer preference weights from attention using disp+var+kurt with temperature exponents.
    - Input attention is already softmaxed (probabilities).
    - Supports shapes:
        [B, H, Lq, Lk]       -> returns scalar weight (0D tensor)
    - No per-metric normalization; only a final max-normalization across layers.

    Args:
        attn: attention probabilities. Last dim is key_len (Lk). If a layer axis exists, it is dim 0.
        tau1, tau2, tau3: temperature exponents for disp, var, kurt (tau>1 compresses; tau<1 sharpens).
        eps: small constant for numerical stability.

    Returns:
        weights: returns a 0D tensor in [0,1].
    """
    # B, H, Lq, Lk = attn.shape
    # p = attn.clamp_min(eps)
    # p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)

    # logL = torch.log(torch.tensor(float(Lk), device=p.device, dtype=p.dtype))
    # H_ent = -(p * p.log()).sum(dim=-1)                    # [B,H,Lq]
    # disp_raw = (logL - H_ent).clamp_min(eps)

    # mean_p = 1.0 / float(Lk)
    # var_raw = ((p - mean_p) ** 2).mean(dim=-1).clamp_min(eps)  # [B,H,Lq]

    # m4 = ((p - mean_p) ** 4).mean(dim=-1)
    # kurt = (m4 / (var_raw ** 2 + eps))
    # kurt_excess = (kurt - 3.0).clamp_min(eps)

    # disp_w = disp_raw.pow(1.0 / max(tau1, eps))
    # var_w  = var_raw.pow (1.0 / max(tau2, eps))
    # kurt_w = kurt_excess.pow(1.0 / max(tau3, eps))

    # fused = disp_w * var_w * kurt_w                       # [B,H,Lq]
    # score = fused.mean(dim=(0, 1, 2))                     # scalar
    # # Single layer -> max-normalization is degenerate; return 1.0 if positive else 0.0
    # return torch.where(score > 0, score.new_tensor(1.0), score.new_tensor(0.0))
    disp = calculate_entropy(attn)
    var = torch.var(attn, dim=-2).sum(0).sum(0).sum(0)
    kurt = calculate_kurtosis(attn, eps=eps, fisher=False)
    pref_score = (disp**(1/tau1)*var**(1/tau2)*kurt**(1/tau3)).cpu().numpy()
    # pref_score = (disp**(1/tau1)*var**(1/tau2)).cpu().numpy()
    return pref_score
    


def adjust_budgets(origin_budget, seq_len, layer_budget):

    b = np.asarray(origin_budget, dtype=int)
    cap = int(max(min(layer_budget, seq_len), 0))
    b = np.clip(b, 0, cap)
    return b.tolist()

def heads_diff(
    left: torch.Tensor,
    right: torch.Tensor,
):
    """
    Get the difference set of left - right.
    left: (B, H, K)
    right: (B, H, O)
    return: diff: (B, H, K-O)
    """
    eq = left.unsqueeze(-1) == right.unsqueeze(-2)
    mask_not_in_o = ~eq.any(dim=-1)
    diff = left.masked_select(mask_not_in_o)
    diff = diff.view(left.size(0), left.size(1), -1)
    return diff 


def heads_union(
    left: torch.Tensor,
    right: torch.Tensor,
):
    """
    Get the union set of left U right.
    left: (B, H, K)
    right: (B, H, O)
    return: union: (B, H, K+O) -> (B, H, U<=K+O)
    """
    return torch.unique(torch.cat([left, right], dim=-1), dim=-1)

def build_unified_positions(o_sort: torch.Tensor, q_sort: torch.Tensor):
    """
    Input:
        o_sort: [B, H, Lo]  sorted origin indices in ascending order
        q_sort: [B, H, Lq]  sorted quantized indices in ascending order
    Return:
      new_o_idx: [B, H, Lo] or None  (when Lo=0)
      new_q_idx: [B, H, Lq] or None  (when Lq=0)
    """
    B, H, Lo = o_sort.shape
    Lq = q_sort.shape[-1]
    device = o_sort.device

    # If both sides are empty
    if Lo == 0 and Lq == 0:
        return None, None

    # If q is empty, only need to return 0..Lo-1
    if Lq == 0:
        new_o_idx = torch.arange(Lo, device=device).view(1, 1, -1).expand(B, H, -1)
        return new_o_idx, None

    # If o is empty, only need to return 0..Lq-1
    if Lo == 0:
        new_q_idx = torch.arange(Lq, device=device).view(1, 1, -1).expand(B, H, -1)
        return None, new_q_idx

    # --- Normal case: Lo>0 and Lq>0 ---
    merged = torch.cat([o_sort, q_sort], dim=-1)  # [B,H,Lo+Lq]
    is_quant = torch.cat([
        torch.zeros((B, H, Lo), dtype=torch.bool, device=device),
        torch.ones ((B, H, Lq), dtype=torch.bool, device=device),
    ], dim=-1)

    order = torch.argsort(merged, dim=-1, stable=True)               # [B,H,Lo+Lq]
    is_quant_sorted = torch.gather(is_quant, dim=-1, index=order)    # [B,H,Lo+Lq]

    final_pos_sorted = torch.arange(merged.size(-1), device=device, dtype=torch.long)
    final_pos_sorted = final_pos_sorted.view(1, 1, -1).expand(B, H, -1)  # [B,H,Lo+Lq]

    new_o_idx = final_pos_sorted.masked_select(~is_quant_sorted).view(B, H, Lo)
    new_q_idx = final_pos_sorted.masked_select( is_quant_sorted).view(B, H, Lq)
    return new_o_idx, new_q_idx

def quantize_tensor(
    x: torch.Tensor,
    type: str = "fp8",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if type == "fp8":
        return quantize_fp8_tensor(x)
    else:
        raise ValueError(f"Unsupported quantization type: {type}")
    
def dequantize_tensor(
    q: torch.Tensor, 
    scale: torch.Tensor, 
    out_dtype: torch.dtype,
    type: str = "fp8",
) -> torch.Tensor:
    if type == "fp8":
        return dequantize_fp8(q, scale, out_dtype)
    elif type == "int4":
        return dequantize_int4(q, scale, out_dtype)
    else:
        raise ValueError(f"Unsupported quantization type: {type}")

def quantize_fp8_tensor(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize per-token (seq axis) to FP8-like dtype with a per-token scale.
    x: [B, H, T, D]
    Return:
      q: [B, H, T, D] (torch.float8_e4m3fn or simulated)
      s: [T] or [T, 1, 1] broadcastable to [B, H, T, D]
    """
    with torch.no_grad():
        # amax over (B,H,D) for each token T
        amax = x.abs().amax(dim=(0, 1, 3))  # [T]
        qmax = 448.0  # "large enough" int range for simulation; replace with real fp8 range if needed
        scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
        q = (x / scale.view(1,1,-1,1)).clamp(-qmax, qmax).to(torch.float8_e4m3fn)
    return q, scale

def dequantize_fp8(
    q: torch.Tensor, 
    scale: torch.Tensor, 
    out_dtype: torch.dtype
) -> torch.Tensor:
    """
    Dequantize q with per-token scale.
    q: [B, H, T, D], scale: [T] or broadcastable to [B,H,T,D]
    """
    return (q.to(torch.float32) * scale.view(1,1,-1,1)).to(out_dtype)

def append_new_indices(
    old_indices: torch.Tensor,
    new_start: int,
    new_end: int,
) -> torch.Tensor:
    """
    Append new_indices to old_indices (if not None).
    old_indices: [B, H, T, D]
    new_start: int
    new_end: int
    return: combined: [B, H, T + new_end - new_start, D]
    """
    if old_indices is None:
        raise ValueError("old_indices should not be None")
    new_indices = torch.arange(new_start, new_end, device=old_indices.device)
    # print(f"old_indices: {old_indices.shape}")
    B, H, _, D = old_indices.shape
    new_indices = new_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B, H, -1, D)
    combined = torch.cat([old_indices, new_indices], dim=2)
    return combined


