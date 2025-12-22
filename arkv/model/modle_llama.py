import math
from typing import Optional, Callable

import torch
from torch import nn

import transformers

from transformers.cache_utils import Cache, DynamicCache
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)

from arkv.cache.adaptive_cache import AdaptiveCache

from arkv.calculator import calculate_heavy_hitter, disp_var_kurt_preference

def replace_llama3_attn():
    transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attention_forward
    

@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor]:
    
    cache_config = self.config.cache_config
    ad_cache = AdaptiveCache(cache_config)
    
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
    
    
    key_test = repeat_kv(key_states, self.num_key_value_groups)
    bsz, q_len, _ = hidden_states.size()
    tmp_size = min(q_len, cache_config.window_size)
    if cache_config.prefill[self.layer_idx]:

        tmp_attn_weights = torch.matmul(query_states[..., -tmp_size:, :], key_test.transpose(2, 3)) / math.sqrt(self.head_dim)

        if q_len !=1:
            # mask = torch.full((tmp_size, tmp_size), torch.finfo(tmp_attn_weights.dtype).min, device=tmp_attn_weights.device)
            # mask_cond = torch.arange(mask.size(-1), device=tmp_attn_weights.device)
            # mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            # mask = mask.to(tmp_attn_weights.device)
            # tmp_attention_mask = mask[None, None, :, :]
            # tmp_attn_weights[:, :, -tmp_size:, -tmp_size:] += tmp_attention_mask
            tri = torch.tril(torch.ones(tmp_size, tmp_size, device=tmp_attn_weights.device))
            local_mask = (1.0 - tri) * torch.finfo(tmp_attn_weights.dtype).min
            tmp_attn_weights[:, :, -tmp_size:, -tmp_size:] += local_mask[None, None, :, :]

        tmp_attn_weights = nn.functional.softmax(tmp_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        oq_score = disp_var_kurt_preference(
            tmp_attn_weights[:,:,-tmp_size:,:-tmp_size],
            cache_config.tau1,
            cache_config.tau2,
            cache_config.tau3
        )
            
        #compute preference score and hh score
        attention_score = tmp_attn_weights[:, :, -tmp_size:, :] 
        # print(f"cache_config.gamma: {cache_config.gamma}, type: {type(cache_config.gamma)}, value: {cache_config.gamma}")
        past_key_values.gamma = cache_config.gamma

        hh_score = calculate_heavy_hitter(
            attention_scores=attention_score, 
            gamma=past_key_values.gamma, 
            window_size=cache_config.window_size,
            bsz=bsz,
            num_key_value_heads=self.config.num_key_value_heads,
            num_key_value_groups=self.num_key_value_groups
        )
        # print(f"hh_score: {hh_score}")
        past_key_values.update_score(oq_score, hh_score)

        # past_key_values.layer_budget.append(cache_config.key_size[self.layer_idx])
        cache_config.prefill[self.layer_idx] = False
        # if not any(cache_config.prefill):
        prefill_compressor = ad_cache.prefill_cpr()(cache_config)
        past_key_values = prefill_compressor(past_key_values, q_len)

    if cache_config.decoding_compressor[self.layer_idx] is None:
        cache_config.decoding_compressor[self.layer_idx] = ad_cache.decoding_cpr()(cache_config)
        cache_config.quant_ratio_per_layer[self.layer_idx] = past_key_values.get_quant_ratio(self.layer_idx)
    else:
        tmp_attn_weights = torch.matmul(query_states[..., -tmp_size:, :], key_test.transpose(2, 3)) / math.sqrt(self.head_dim)
        tmp_attn_weights = nn.functional.softmax(tmp_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention_score = tmp_attn_weights[:, :, -tmp_size:, :] 
        past_key_values = cache_config.decoding_compressor[self.layer_idx](past_key_values, attention_score, self.layer_idx)
    
    # return attn_output, attn_weights
    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


@check_model_inputs
# @auto_docstring
def llama_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)
    if use_cache and past_key_values is None:
        dynamicCache = DynamicCache(config=self.config)
        cache_config = self.config.cache_config
        cache_config.refresh_model_settings()
        ad_cache = AdaptiveCache(cache_config)
        past_key_values = ad_cache.cache().from_dynamic_cache(dynamicCache)
    elif past_key_values is not None and isinstance(past_key_values, DynamicCache):
        cache_config = self.config.cache_config
        cache_config.refresh_model_settings()
        ad_cache = AdaptiveCache(cache_config)
        past_key_values = ad_cache.cache().from_dynamic_cache(past_key_values)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position: torch.Tensor = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


