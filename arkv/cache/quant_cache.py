import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import DynamicCache, Cache, HybridCache, CacheLayerMixin
from typing import Any, Dict, List, Optional, Tuple, Union
from arkv.config import ADCacheConfig
from arkv.calculator import adjust_budgets, append_new_indices, build_unified_positions, calculate_heavy_hitter, dequantize_tensor, quantize_tensor


class QuantLayer(CacheLayerMixin):
    
    
    def __init__(self):
        self.dtype, self.device = None, None
        self.origin_keys = None
        self.origin_values = None
        self.quant_keys = None
        self.quant_values = None
        self.origin_idx = None
        self.quant_idx = None
        self.key_scales = None
        self.value_scales = None

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.origin_keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.origin_values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.quant_keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.quant_values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.origin_idx = torch.tensor([], dtype=torch.long, device=self.device)
        self.quant_idx = torch.tensor([], dtype=torch.long, device=self.device)
        self.key_scales = torch.tensor([], dtype=torch.float, device=self.device)
        self.value_scales = torch.tensor([], dtype=torch.float, device=self.device)

    def get_kv_cache(self):
        """
        Retrieve the full key and value cache for a given layer by merging the origin and quantized caches.
        Returns:
            A tuple of (key_cache, value_cache) for the specified layer.
        """
        if self.origin_keys is None or self.origin_keys.numel() == 0:
            return (None, None)

        o_k = self.origin_keys          # [B, H, O, D]
        o_v = self.origin_values        # [B, H, O, D]
        q_k = self.quant_keys           # [B, H, Q, D] or None
        q_v = self.quant_values         # [B, H, Q, D] or None

        o_indices = self.origin_idx      # [B, H, O]
        q_indices = self.quant_idx       # [B, H, Q] or None

        # Unified sequence length L = O + Q (new_o_idx/new_q_idx should map to 0..L-1 when generated)
        B, H, O, D = o_k.shape
        Q = q_k.shape[2] if (q_k is not None) else 0
        L = O + Q

        # ---- Keys ----
        if q_k is None:
            f_k = o_k
        else:
            dq_k = dequantize_tensor(q_k, self.key_scales, o_k.dtype)  # [B, H, Q, D]
            # Initialize unified KV
            f_k = o_k.new_zeros((B, H, L, D))
            # Put origin in the position specified by new_o_idx
            try:
                f_k.scatter_(dim=2, index=o_indices, src=o_k)
            except Exception as e:
                print(f"Error in scatter o_k: o_indices: {o_indices}, o_k: {o_k}")
                raise e
            # Put quant in the position specified by new_q_idx
            if q_indices is not None and Q > 0:
                f_k.scatter_(dim=2, index=q_indices, src=dq_k)

        # ---- Values ----
        if q_v is None:
            f_v = o_v
        else:
            dq_v = dequantize_tensor(q_v, self.value_scales, o_v.dtype)  # [B, H, Q, D]
            f_v = o_v.new_zeros((B, H, L, D))
            f_v.scatter_(dim=2, index=o_indices, src=o_v)
            if q_indices is not None and Q > 0:
                f_v.scatter_(dim=2, index=q_indices, src=dq_v)
        
        return (f_k, f_v)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if self.origin_keys is None:
            self.lazy_initialization(key_states)
        
        if key_states is None or value_states is None:
            return self.get_kv_cache(layer_idx)

        new_T = key_states.shape[-2]
            
        if self.origin_keys.numel() == 0:
            B, H, _, D = key_states.shape
            new_idx = torch.arange(0, new_T, device=key_states.device, dtype=torch.int32)
            new_idx = new_idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B, H, -1, D)  # [B, H, new_T, D]
            self.set_layer_info(
                o_k=key_states, o_v=value_states,
                o_indices=new_idx,
            )
        else:
            ori_keys = torch.cat(
                [self.origin_keys, key_states], dim=-2
            )
            ori_values = torch.cat(
                [self.origin_values, value_states], dim=-2
            )
            o_indices = self.origin_idx
            q_indices = self.quant_idx
            start = o_indices.shape[-2]
            if q_indices is not None and q_indices.numel() > 0:
                start += q_indices.shape[-2]
            end = start + new_T
            new_indices = append_new_indices(
                old_indices=o_indices,
                new_start=start,
                new_end=end
            )
            self.set_layer_info(
                o_k=ori_keys, o_v=ori_values,
                o_indices=new_indices,
            )

        return self.get_kv_cache()
    
    
    def set_layer_info(
        self,
        o_k: Optional[torch.Tensor] = None,
        o_v: Optional[torch.Tensor] = None,
        q_k: Optional[torch.Tensor] = None,
        q_v: Optional[torch.Tensor] = None,
        o_indices: Optional[torch.Tensor] = None,
        q_indices: Optional[torch.Tensor] = None,
        k_scales: Optional[torch.Tensor] = None,
        v_scales: Optional[torch.Tensor] = None,
    ):
        """Record per-layer quant payload + indices + scales, and origin indices."""
        # Ensure slots exist

        self.origin_keys    = o_k
        self.origin_values  = o_v
        self.quant_keys     = q_k
        self.quant_values   = q_v
        self.origin_idx     = o_indices
        self.quant_idx      = q_indices
        self.value_scales   = v_scales
        self.key_scales     = k_scales

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask"""
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        if self.origin_keys is None or self.origin_keys.numel() == 0:
            return 0
        o_len = self.origin_keys.shape[-2]
        # print(f"[ARKV] MixCache: get_seq_length layer {layer_idx} origin length {o_len}")
        if self.quant_keys is not None:
            # print(f"[ARKV] MixCache: get_seq_length layer {layer_idx} quant length {self.quant_key_cache[layer_idx].shape[-2]}")
            o_len += self.quant_keys.shape[-2]
        return o_len

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length of the cache object. DynamicLayer does not have a maximum length."""
        return -1

    def crop(self, max_length: int) -> None:
        """
        Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be negative
        to remove `max_length` tokens.
        """
        print(f"[ARKV] MixCache: crop to max_length {max_length}")
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        B, H, _, D = key_states.shape
        key_states, value_states = self.get_kv_cache()
        new_idx = torch.arange(0, max_length, device=key_states.device, dtype=torch.int32)
        new_idx = new_idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B, H, -1, D)  # [B, H, new_T, D]
        self.set_layer_info(
            o_k=key_states[..., :max_length, :],
            o_v=value_states[..., :max_length, :],
            o_indices=new_idx,
        )


    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat the cache `repeats` times in the batch dimension."""
        print(f"[ARKV] MixCache: batch_repeat_interleave by {repeats}")
        key_states, value_states = self.get_kv_cache()
        if key_states.numel() > 0:
            key_states = key_states.repeat_interleave(repeats, dim=0)
            value_states = value_states.repeat_interleave(repeats, dim=0)
            B, H, L, D = key_states.shape
            new_idx = torch.arange(0, L, device=key_states.device, dtype=torch.int32)
            new_idx = new_idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B, H, -1, D)  # [B, H, L, D]
            self.set_layer_info(
                o_k=key_states,
                o_v=value_states,
                o_indices=new_idx,
            )
        
    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Only keep the `indices` in the batch dimension of the cache."""
        print(f"[ARKV] MixCache: batch_select_indices {indices}")
        key_states, value_states = self.get_kv_cache()
        if key_states.numel() > 0:
            key_states = key_states[indices, ...]
            value_states = value_states[indices, ...]
            B, H, L, D = key_states.shape
            new_idx = torch.arange(0, L, device=key_states.device, dtype=torch.int32)
            new_idx = new_idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B, H, -1, D)  # [B, H, L, D]
            self.set_layer_info(
                o_k=key_states,
                o_v=value_states,
                o_indices=new_idx,
            )


class QuantCache(Cache):
    """
    Cache for storing quantized key-value pairs along with their scales and indices.
    Only window_size most recent tokens are stored in original precision.
    """
    
    def __init__(self) -> None:
        self.layers: List[QuantLayer] = []

        self._seen_tokens = 0  # global running length (tokens)
        self.pref_scores:  List[torch.Tensor] = []
        self.evict_scores: List[torch.Tensor] = []
        self.origin_budget: List[int] = []
        self.gamma = 0.5
        self.mix_type = "quant"  # "mix", "quant", "origin"
        self.o_dtype = torch.float16
    
    def get_quant_ratio(self, layer_idx: int) -> float:
        """Get the quantization ratio for a given layer."""
        return 0.0
    
    def get_kv_cache(self, layer_idx: int):
        """
        Retrieve the full key and value cache for a given layer by merging the origin and quantized caches.
        Returns:
            A tuple of (key_cache, value_cache) for the specified layer.
        """
        return self.layers[layer_idx].get_kv_cache()
        
    
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self.layers):
            return self.get_kv_cache(layer_idx)
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield self.get_kv_cache(layer_idx)

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.layers)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx >= len(self.layers):
            self.layers.append(QuantLayer())
        if key_states is None or value_states is None:
            return self.layers[layer_idx].get_kv_cache()
        self.layers[layer_idx].update(key_states, value_states)
        return self.layers[layer_idx].get_kv_cache()
    
    def set_layer_info(
        self,
        layer_idx: int,
        o_k: Optional[torch.Tensor],
        o_v: Optional[torch.Tensor],
        q_k: Optional[torch.Tensor],
        q_v: Optional[torch.Tensor],
        q_indices: Optional[torch.Tensor],
        k_scales: Optional[torch.Tensor],
        v_scales: Optional[torch.Tensor],
        o_indices: Optional[torch.Tensor],
    ):
        """Record per-layer quant payload + indices + scales, and origin indices."""
        # Ensure slots exist
        self.layers[layer_idx].set_layer_info(
            o_k=o_k, o_v=o_v,
            q_k=q_k, q_v=q_v,
            o_indices=o_indices,
            q_indices=q_indices,
            k_scales=k_scales,
            v_scales=v_scales,
        )


    def update_score(
        self,
        pref_score: torch.Tensor,
        evict_score: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ):
        self.pref_scores.append(pref_score)
        self.evict_scores.append(evict_score)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return self.layers[layer_idx].get_seq_length()

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. QuantCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `MixeCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += (self.get_kv_cache(layer_idx),)
        return legacy_cache
    
    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "QuantCache":
        """Converts a cache in the legacy cache format into an equivalent `QuantCache`. Used for
        backward compatibility."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
    
    @classmethod
    def from_dynamic_cache(cls, past_key_values: Optional[DynamicCache] = None) -> "QuantCache":
        cache = cls()
        if past_key_values is not None:
            for layer_idx, cache_layer in enumerate(past_key_values.layers):
                key_states = cache_layer.keys
                value_states = cache_layer.values
                cache.update(key_states, value_states, layer_idx)
        return cache
    

    def log_state(self, place, logging=False, logging_place=False):
        if not logging_place:
            return
        print(f"[ARKV] QuantCache at {place}: {len(self)} layers")
        if not logging:
            return
        for i in range(len(self)):
            o_k = self.origin_key_cache[i]
            o_v = self.origin_value_cache[i]
            q_k = self.quant_key_cache[i] if i < len(self.quant_key_cache) else None
            q_v = self.quant_value_cache[i] if i < len(self.quant_value_cache) else None
            o_idx = self.origin_indices[i]
            q_idx = self.quant_indices[i]
            print(f"  Layer {i}: origin {o_k.shape if o_k is not None else None}, quant {q_k.shape if q_k is not None else None}\n")
            print(f"           origin_idx {o_idx.shape if o_idx is not None else None}, quant_idx {q_idx.shape if q_idx is not None else None}\n")
            print(f"           origin_idx_min: {o_idx.min().item() if o_idx is not None and o_idx.numel()>0 else None}, max: {o_idx.max().item() if o_idx is not None and o_idx.numel()>0 else None}\n")
            print(f"           quant_idx_min: {q_idx.min().item() if q_idx is not None and q_idx.numel()>0 else None}, max: {q_idx.max().item() if q_idx is not None and q_idx.numel()>0 else None}\n")
            print(f"           key_scales {self.key_scales[i].shape if self.key_scales[i] is not None else None}, "
                  f"value_scales {self.value_scales[i].shape if self.value_scales[i] is not None else None}\n")
        print(f"  Pref scores: {[s.item() for s in self.pref_scores]}\n")
        print(f"  Evict scores: {[s.shape if isinstance(s, torch.Tensor) else (s[0].shape, s[1].shape) for s in self.evict_scores]}\n")

class QuantPrefillKVCompressor:
    def __init__(
        self,
        cache_config: ADCacheConfig,
    ):
        self.window_size = cache_config.window_size
        self.total_size = cache_config.key_size * cache_config.layer_num
        self.cache_size = cache_config.cache_size

    def __call__(self, past_key_values, seq_len):
        # print(f"[ARKV] MixPrefillKVCache: total_size {self.total_size}, window_size {self.window_size}, seq_len {seq_len}")
        # If context is not longer than cache_size + window, nothing to do.

        # Per-layer total budget is identical.
        # 'available_size' is the capacity for the *past* part (excluding the window).
        layer_budget_total = self.cache_size
        available_size = max(self.cache_size - self.window_size, 0)

        # Preference scores per layer. Use the max to normalize into [0, 1].
        pref_scores = past_key_values.pref_scores
        # print(f"[ARKV] MixPrefillKVCache: pref_scores {[s.item() for s in pref_scores]}")
        max_pref = max(pref_scores) if len(pref_scores) > 0 else 0.0

        # Compute per-layer origin budgets for the PAST part (exclude window).
        # ratio = pref / max_pref; higher score => more FP16 in the past segment.
        if max_pref > 0:
            origin_budgets = [int(round((s / max_pref) * available_size)) for s in pref_scores]
        else:
            # Degenerate case: all scores are zero; give everyone the same split (e.g., 0).
            origin_budgets = [0 for _ in pref_scores]

        # Clip each origin budget to legal range:
        #   0 <= origin_budget_i <= min(available_size, seq_len - window_size)
        # (No cross-layer redistribution; each layer is independent now.)
        origin_budgets = adjust_budgets(
            origin_budgets,
            seq_len=max(seq_len - self.window_size, 0),
            layer_budget=available_size,
        )

        # Apply per-layer eviction with the computed *past* FP16 budget.
        # Also record the identical per-layer *total* layer budget (for bookkeeping).
        for layer_idx, budget in enumerate(origin_budgets):
            # Safety clamp (harmless if already clamped in adjust_budgets).
            budget = min(budget, max(seq_len - self.window_size, 0))
            if past_key_values.origin_budget is None or len(past_key_values.origin_budget) < len(past_key_values):
                past_key_values.origin_budget.append(budget)
            else:
                past_key_values.origin_budget[layer_idx] = budget
            # Store identical per-layer total budget (not the past budget).
            # past_key_values.layer_budget[layer_idx] = layer_budget_total
        layer_idx = len(origin_budgets) - 1  # Only evict the last layer (most recent)
        past_key_values = self.evict_layer_kvcache(
            past_key_values,
            layer_idx,
            available_size,
        )

        return past_key_values

    def evict_layer_kvcache(self, past_key_values: QuantCache, layer_idx: int, layer_budget: int):
        # print(f"[ARKV] MixPrefillKVCache: evict layer {layer_idx} with layer_budget:{layer_budget}")
        layer = past_key_values.layers[layer_idx]
        origin_keys, origin_values = layer.origin_keys, layer.origin_values
        B, H, o_len, D = origin_keys.shape
        if o_len <= self.cache_size:
            return past_key_values
        hh_score = past_key_values.evict_scores[layer_idx]
        if layer_budget > hh_score.shape[-1]:
            layer_budget = hh_score.shape[-1]

        indices = hh_score.topk(layer_budget, dim=-1).indices
        hh_score_compress = hh_score.gather(dim=2, index=indices)
        past_key_values.evict_scores[layer_idx] = hh_score_compress

        indices = indices.unsqueeze(-1).expand(-1, -1, -1, D)

        q_k = origin_keys[:, :, :-self.window_size, :].gather(dim=2, index=indices)
        q_v = origin_values[:, :, :-self.window_size, :].gather(dim=2, index=indices)
        o_k = origin_keys[:, :, -self.window_size:, :]
        o_v = origin_values[:, :, -self.window_size:, :]

        k_scale = None
        v_scale = None
        if (q_k.numel() != 0) and (q_v.numel() != 0):
            q_k, k_scale = quantize_tensor(q_k)
            q_v, v_scale = quantize_tensor(q_v)

        new_T = q_k.shape[2]
        # Full origin indices: [new_T-window_size .. new_T-1]
        o_idx = torch.arange(new_T, new_T + self.window_size, device=o_k.device, dtype=torch.int32)
        o_idx = o_idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B, H, -1, D)  # [B, H, window_size, D]
        # Full quant indices: [0..new_T-1]
        q_idx = torch.arange(0, new_T, device=q_k.device, dtype=torch.int32)
        q_idx = q_idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B, H, -1, D)  # [B, H, T, D]

        past_key_values.set_layer_info(
            layer_idx = layer_idx,
            o_k = o_k, o_v = o_v,
            q_k = q_k, q_v = q_v,
            o_indices = o_idx, q_indices = q_idx,
            k_scales = k_scale, v_scales = v_scale,
        )

        return past_key_values
        

class QuantDecodingKVCompressorLayerWise:
    def __init__(
        self,
        cache_config: ADCacheConfig,
    ):
        """
        During decoding:
          - Always keep last `window_size` tokens as origin.
          - From earlier "past", keep at most `hh_size` tokens (importance-based).
          - Split the kept past into {origin, quant} by `origin_ratio`.
          - Quant tokens are stored in QuantCache.quant_* with per-token scales & 1D indices.
        """
        self.key_size = cache_config.key_size
        self.window_size = cache_config.window_size
        self.cache_size = cache_config.cache_size
        # self.hh_score = None  # optional running score (if you want cumulative update)

    @torch.no_grad()
    def __call__(self, past_key_values: "QuantCache", attn_score: torch.Tensor, layer_idx: int):
        # print(f"[ARKV] MixDecodingKVCache_LayerWise: layer {layer_idx}")
        """
        Quant+Evict (simple): keep last window as origin; from past keep top-hh_size by importance;
        split kept past into {origin, quant} by origin_ratio; store K/V and final positions via
        o_indices (origin positions) and q_indices (quant positions). No o_write/q_write.
        """
        K, V = past_key_values.get_kv_cache(layer_idx)   # [B, Hkv, T, D]
        device = K.device
        B, Hkv, T, D = K.shape
        nh = attn_score.shape[1]
        Gkv = nh // Hkv  # num key-value groups
        # print(f"B={B}, Hkv={Hkv}, nh={nh}, Gkv={Gkv}, T={T}, D={D}")

        # If total length does not exceed target cache (= hh_size + window_size), do nothing
        o_len = past_key_values.get_seq_length(layer_idx)
        if o_len <= self.cache_size:
            # print(f"layer {layer_idx} no need to evict: {o_len} <= {self.cache_size}")
            return past_key_values

        # 1) Only score the "past" segment (window does not participate in scoring)
        past_len = max(o_len - self.window_size, 0)
        if past_len == 0:
            # Only window: directly record and return
            full_idx = torch.arange(T, device=K.device, dtype=torch.int32)
            full_idx = full_idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B, Hkv, -1, D)  # [B, H, T, D]
            past_key_values.set_layer_info(
                layer_idx,
                o_k=K, o_v=V,
                q_k=None, q_v=None,
                q_indices=None, k_scales=None, v_scales=None,
                o_indices=full_idx,
            )
            # print(f"layer {layer_idx} no past to evict: past_len=0")
            return past_key_values
        
        keep_size = int(past_len / 4 * 3)  # keep at least 75% of past tokens
        hh_scores = calculate_heavy_hitter(
            attn_score,  # [B, nh, q_len, past_len]
            gamma=past_key_values.gamma,
            window_size=self.window_size,
            bsz=B,
            num_key_value_heads=Hkv,
            num_key_value_groups=Gkv,
        )
        
        hh_size = hh_scores.shape[-1]
        # keep_size = min(keep_size, hh_size)
        if hh_size < keep_size:
            raise ValueError(f"hh_size={hh_size} < keep_size={keep_size}, cannot evict")
              
        # 1) Calculate indices by hh_score and origin_budget
        quant_budget = past_len
        
        keep_indices = hh_scores.topk(keep_size, dim=-1).indices  # [B, Hkv, k]
        q_indices = keep_indices
        o_indices = torch.arange(past_len, o_len, device=device, dtype=torch.int32)  # [window_size]
        o_indices = o_indices.unsqueeze(0).unsqueeze(0).expand(B, Hkv, -1)  # [B, Hkv, window_size]

        # 2) Calculate new o_idx, q_idx
        o_idx, q_idx = build_unified_positions(o_indices, q_indices)
        o_idx = o_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # [B, Hkv, o + window_size, D]
        q_idx = q_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # [B, Hkv, q, D]
        
        # 3) Keep kv with o_indices, q_indices
        o_indices = o_indices.unsqueeze(-1).expand(-1, -1, -1, D)  # [B, Hkv, o, D]
        q_indices = q_indices.unsqueeze(-1).expand(-1, -1, -1, D)  # [B, Hkv, q, D]
        o_k = K.gather(dim=2, index=o_indices)
        o_v = V.gather(dim=2, index=o_indices)
        q_k = K.gather(dim=2, index=q_indices) if quant_budget > 0 else None
        q_v = V.gather(dim=2, index=q_indices) if quant_budget > 0 else None
        
        # 4) Quantize q_k, q_v
        # print(f"o_k: {o_k.shape}, o_v: {o_v.shape}, q_k: {q_k.shape if q_k is not None else None}, q_v: {q_v.shape if q_v is not None else None}")
        if quant_budget > 0:
            q_k, k_scale = quantize_tensor(q_k)
            q_v, v_scale = quantize_tensor(q_v)
        else:
            q_k, k_scale, q_idx = None, None, None
            q_v, v_scale = None, None
        
        # 5) Update QuantCache with new origin + quant
        past_key_values.set_layer_info(
            layer_idx,
            o_k=o_k, o_v=o_v,
            q_k=q_k, q_v=q_v,
            q_indices=q_idx,
            k_scales=k_scale, v_scales=v_scale,
            o_indices=o_idx,
        )
        
        return past_key_values

