# coding=utf-8
# GPT-OSS model profiling wrapper for LLMServingSim.
# Adapted from mixtral.py profiler wrapper.
#
# Architecture: Sparse MoE with top-k routing, GQA, SwiGLU experts, RMSNorm.
# Key difference from Mixtral: explicit head_dim (not derived from hidden_size / num_heads),
# attention_bias=True, 32 experts with top-4 routing, SiLU activation with alpha=1.702.

from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs

from profiler.common.timer import Timer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class GptOssConfig:
    """Minimal config stub used by the profiler.

    When profiling via ``AutoConfig.from_pretrained`` the real HuggingFace
    config object is used instead. This class only documents the expected
    fields so that type-checkers and readers can follow the code.
    """
    model_type = "gpt_oss"


# ---------------------------------------------------------------------------
# Expert MLP  (SwiGLU: w1/w3 gate, act, w2 down)
# ---------------------------------------------------------------------------
class GptOssExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        # gpt-oss uses bias=True in expert linears
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=True)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=True)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=True)

        self.act_fn = ACT2FN[getattr(config, "hidden_act", "silu")]

        self._w1_timer = Timer(name="expert.w1")
        self._w2_timer = Timer(name="expert.w2")
        self._w3_timer = Timer(name="expert.w3")
        self._act_timer = Timer(name="act_fn")

    def forward(self, hidden_states):
        with self._w1_timer:
            w1 = self.w1(hidden_states)

        with self._w3_timer:
            w3 = self.w3(hidden_states)

        with self._act_timer:
            act = self.act_fn(w1) * w3

        with self._w2_timer:
            current_hidden_states = self.w2(act)

        return current_hidden_states


# ---------------------------------------------------------------------------
# Sparse MoE block
# ---------------------------------------------------------------------------
class GptOssSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating — gpt-oss uses bias=True in the router
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=True)

        # Single expert instance for profiling (same as mixtral profiler pattern)
        self.experts = GptOssExpertMLP(config)

        self._gate_timer = Timer(name="gate")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with self._gate_timer:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
            router_logits = self.gate(hidden_states)

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(hidden_states.dtype)

        # Profile one expert (times num_local_experts scaling done in profiler main)
        final_hidden_states = self.experts(hidden_states)

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class GptOssRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# Rotary embedding helpers
# ---------------------------------------------------------------------------
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class GptOssAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.is_causal = True

        # gpt-oss uses attention_bias=True
        bias = getattr(config, "attention_bias", True)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim // config.tp_size, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim // config.tp_size, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim // config.tp_size, bias=bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim // config.tp_size, config.hidden_size, bias=bias)

        self._q_proj_timer = Timer(name="q_proj")
        self._k_proj_timer = Timer(name="k_proj")
        self._v_proj_timer = Timer(name="v_proj")
        self._rope_timer = Timer(name="rope")
        self._o_proj_timer = Timer(name="o_proj")

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        with self._q_proj_timer:
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        with self._k_proj_timer:
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        with self._v_proj_timer:
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        with self._rope_timer:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        with self._o_proj_timer:
            attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------
class GptOssDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GptOssAttention(config, layer_idx)
        self.block_sparse_moe = GptOssSparseMoeBlock(config)
        self.input_layernorm = GptOssRMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-5))
        self.post_attention_layernorm = GptOssRMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-5))

        self._input_layernorm_timer = Timer(name="input_layernorm")
        self._post_layernorm_timer = Timer(name="post_layernorm")

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        residual = hidden_states
        with self._input_layernorm_timer:
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        with self._post_layernorm_timer:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Rotary embedding (reuses Llama-style RoPE compatible with YaRN)
# ---------------------------------------------------------------------------
class GptOssRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Pre-trained model base
# ---------------------------------------------------------------------------
class GptOssPreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GptOssDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class GptOssModel(GptOssPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size // config.tp_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GptOssDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GptOssRMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-5))
        self.rotary_emb = GptOssRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

        self._embedding_timer = Timer(name="embedding")
        self._final_layernorm_timer = Timer(name="final_layernorm")

    @check_model_inputs()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            with self._embedding_timer:
                inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        from transformers.masking_utils import create_causal_mask
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
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        with self._final_layernorm_timer:
            hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


# ---------------------------------------------------------------------------
# Causal LM head
# ---------------------------------------------------------------------------
class GptOssForCausalLM(GptOssPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GptOssModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size // config.tp_size, bias=False)

        self.post_init()

        self._lm_head_timer = Timer(name="lm_head")

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> MoeCausalLMOutputWithPast:
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        with self._lm_head_timer:
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        return MoeCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
        )


__all__ = [
    "GptOssForCausalLM",
    "GptOssModel",
    "GptOssPreTrainedModel",
]
