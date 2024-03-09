
# Flax implementation from transformers for consistency with pretrained model
# Modified https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_flax_clip.py#L969 to be standalone


import numpy as np
import jax
import jax.numpy as jnp

import flax
from flax import struct
import flax.linen as nn
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict


from typing import Optional, Union, List, Any, Callable, Tuple, Sequence
from functools import partial

from einops import rearrange, reduce, repeat

import os
import pickle

PyTreeDef = type(jax.tree_util.tree_structure(None))



@struct.dataclass
class CLIPTextConfig:
    vocab_size: int = 49408
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 77
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 0.00001
    dropout: float = 0.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    bos_token_id: int = 49406
    eos_token_id: int = 49407
    dtype: jnp.dtype = jnp.float32 # TODO


def get_pretrained_config() -> CLIPTextConfig:
    pretrained_config_dict = {
        "attention_dropout": 0.0,
        "bos_token_id": 49406,
        "dropout": 0.0,
        "eos_token_id": 49407,
        "hidden_act": "quick_gelu",
        "hidden_size": 512,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 2048,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 77,
        "num_attention_heads": 8,
        "num_hidden_layers": 12,
        "vocab_size": 49408
    }
    return CLIPTextConfig(**pretrained_config_dict)



class FlaxCLIPTextEmbeddings(nn.Module):
    config: CLIPTextConfig

    def setup(self):
        embed_dim = self.config.hidden_size

        self.token_embedding = nn.Embed(self.config.vocab_size, embed_dim, embedding_init=jax.nn.initializers.normal())
        self.position_embedding = nn.Embed(
            self.config.max_position_embeddings, embed_dim, embedding_init=jax.nn.initializers.normal()
        )
        self.position_ids = jnp.expand_dims(
            jnp.arange(0, self.config.max_position_embeddings, dtype="i4"), axis=(0, 1)
        )

    def __call__(self, input_ids: jnp.ndarray, position_ids: jnp.ndarray) -> jnp.ndarray:
        input_embeds = self.token_embedding(input_ids.astype("i4"))
        position_embeds = self.position_embedding(position_ids.astype("i4"))
        embeddings = input_embeds + position_embeds
        return embeddings


class FlaxCLIPAttention(nn.Module):
    config: CLIPTextConfig
    causal: bool = True

    def setup(self):
        self.embed_dim = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = self.config.attention_dropout

        self.k_proj = nn.Dense(self.embed_dim, dtype=self.config.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.v_proj = nn.Dense(self.embed_dim, dtype=self.config.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.q_proj = nn.Dense(self.embed_dim, dtype=self.config.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.out_proj = nn.Dense(self.embed_dim, dtype=self.config.dtype, kernel_init=jax.nn.initializers.normal(0.01))

        # self.causal = isinstance(self.config, CLIPTextConfig)
        if self.causal:
            self.causal_mask = make_causal_mask(jnp.ones((1, self.config.max_position_embeddings), dtype="i4"))

    def _split_heads(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        deterministic: bool = True,
        decode: bool = False
    ) -> jnp.ndarray:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        causal_attention_mask = None
        if self.causal:
            query_length, key_length = query.shape[1], key.shape[1]
            causal_attention_mask = self.causal_mask[:, :, key_length - query_length : key_length, :key_length]

        if attention_mask is not None and causal_attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            attention_mask = combine_masks(attention_mask, causal_attention_mask, dtype="i4")
        elif causal_attention_mask is not None:
            attention_mask = causal_attention_mask
        elif attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))


        if decode: # https://github.com/google/flax/blob/main/flax/linen/attention.py#L195
            # detect if we're initializing by absence of existing cache data.
            is_initialized = self.has_variable('cache', 'cached_key')
            cached_key = self.variable('cache', 'cached_key',
                jnp.zeros, key.shape, key.dtype)
            cached_value = self.variable('cache', 'cached_value',
                jnp.zeros, value.shape, value.dtype)
            cache_index = self.variable('cache', 'cache_index',
                lambda: jnp.array(0, dtype=jnp.int32))
            
            if is_initialized:
                *batch_dims, max_length, num_heads, depth_per_head = (
                    cached_key.value.shape)
                # shape check of cached keys against query input
                expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
                if expected_shape != query.shape:
                    raise ValueError('Autoregressive cache shape error, '
                        'expected query shape %s instead got %s.' %
                        (expected_shape, query.shape))
                # update key, value caches with our new 1d spatial slices
                cur_index = cache_index.value
                indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
                key = jax.lax.dynamic_update_slice(cached_key.value, key, indices)
                value = jax.lax.dynamic_update_slice(cached_value.value, value, indices)
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cache_index.value + 1
                # causal mask for cached decoder self-attention:
                # our single query position should only attend to those key
                # positions that have already been generated and cached,
                # not the remaining zero elements.
                attention_mask = combine_masks(
                    attention_mask,
                    jnp.broadcast_to(jnp.arange(max_length) <= cur_index,
                        tuple(batch_dims) + (1, 1, max_length)))
                        


        if attention_mask is not None:
            attention_bias = jax.lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.config.dtype),
                jnp.full(attention_mask.shape, -1e4).astype(self.config.dtype),
            )
        else:
            attention_bias = None

        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout,
            deterministic=deterministic,
            dtype=self.config.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)

        return attn_output


class FlaxCrossAttention(nn.Module):
    config: CLIPTextConfig

    def setup(self):
        self.embed_dim = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = self.config.attention_dropout

        self.k_proj = nn.Dense(self.embed_dim, dtype=self.config.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.v_proj = nn.Dense(self.embed_dim, dtype=self.config.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.q_proj = nn.Dense(self.embed_dim, dtype=self.config.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.out_proj = nn.Dense(self.embed_dim, dtype=self.config.dtype, kernel_init=jax.nn.initializers.normal(0.01))


    def _split_heads(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))


    def _split_heads_cross(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, -1))


    def __call__(
        self,
        hidden_states: jnp.ndarray,
        audio_hidden_state: jnp.ndarray,
        cross_attention_mask: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        query = self.q_proj(hidden_states)
        key = self.k_proj(audio_hidden_state)
        value = self.v_proj(audio_hidden_state)

        query = self._split_heads(query)
        key = self._split_heads_cross(key)
        value = self._split_heads_cross(value)
            

        if cross_attention_mask is not None:
            cross_attention_mask = jnp.expand_dims(cross_attention_mask, axis=(-3,-2))
            attention_bias = jax.lax.select(
                cross_attention_mask > 0,
                jnp.full(cross_attention_mask.shape, 0.0).astype(self.config.dtype),
                jnp.full(cross_attention_mask.shape, -1e4).astype(self.config.dtype),
            )
        else:
            attention_bias = None

        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")


        if query.shape[:-3] != key.shape[:-3]:

            jax.debug.breakpoint()

        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout,
            deterministic=deterministic,
            dtype=self.config.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)

        return attn_output



def quick_gelu(x: jnp.ndarray) -> jnp.ndarray:
    return x * jax.nn.sigmoid(1.702 * x)

ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),
    "quick_gelu": quick_gelu,
}

class FlaxCLIPMLP(nn.Module):
    config: CLIPTextConfig

    def setup(self):
        self.activation_fn = ACT2FN[self.config.hidden_act]
        self.fc1 = nn.Dense(
            self.config.intermediate_size,
            dtype=self.config.dtype,
            kernel_init=jax.nn.initializers.normal(0.01),
        )
        self.fc2 = nn.Dense(self.config.hidden_size, dtype=self.config.dtype, kernel_init=jax.nn.initializers.normal(0.01))

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class FlaxCLIPEncoderLayer(nn.Module):
    config: CLIPTextConfig
    causal: bool = True
    scan: bool = True

    def setup(self):
        self.self_attn = FlaxCLIPAttention(self.config, causal=self.causal)
        self.layer_norm1 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.config.dtype)
        self.mlp = FlaxCLIPMLP(self.config)
        self.layer_norm2 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.config.dtype)

        self.layer_norm3 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.config.dtype)
        self.cross_attn = FlaxCrossAttention(self.config)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        audio_hidden_state: Optional[jnp.ndarray] = None,
        cross_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        decode: bool = False
    ) -> jnp.ndarray:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            decode=decode
        )
        hidden_states = residual + hidden_states

        if audio_hidden_state is not None:

            residual = hidden_states
            hidden_states = self.layer_norm3(hidden_states)
            hidden_states = self.cross_attn(
                hidden_states=hidden_states,
                audio_hidden_state=audio_hidden_state,
                cross_attention_mask=cross_attention_mask,
                deterministic=deterministic,
            )
            hidden_states = residual + hidden_states


        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if self.scan:
            return hidden_states, None

        return hidden_states






class FlaxCLIPLayerCollection(nn.Module):
    config: CLIPTextConfig
    causal: bool = True
    scan: bool = True


    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        deterministic: bool = True,
        audio_hidden_state: Optional[jnp.ndarray] = None,
        cross_attention_mask: Optional[jnp.ndarray] = None,
        decode: bool = False
    ) -> jnp.ndarray:

        if self.scan:
            hidden_states, _ = nn.scan(
                FlaxCLIPEncoderLayer,
                variable_axes={'params': 0, 'cache': 0},
                split_rngs={'params': True, 'dropout': True},
                in_axes=nn.broadcast,
                length=self.config.num_hidden_layers,
            )(self.config, causal=self.causal, scan=True)(
                hidden_states, attention_mask, 
                audio_hidden_state, cross_attention_mask, deterministic, decode
            )
        else:
            for i in range(self.config.num_hidden_layers):
                hidden_states = FlaxCLIPEncoderLayer(self.config, name=str(i), scan=False)(
                    hidden_states, attention_mask, deterministic=deterministic, 
                    audio_hidden_state=audio_hidden_state, cross_attention_mask=cross_attention_mask, decode=decode
                )


        return hidden_states


class FlaxCLIPEncoder(nn.Module):
    config: CLIPTextConfig
    causal: bool = True
    scan: bool = True

    def setup(self):
        self.layers = FlaxCLIPLayerCollection(self.config, causal=self.causal, scan=self.scan)

    def __call__(
        self,
        inputs_embeds: jnp.ndarray,
        attention_mask: jnp.ndarray,
        deterministic: bool = True,
        audio_hidden_state: Optional[jnp.ndarray] = None,
        cross_attention_mask: Optional[jnp.ndarray] = None,
        decode: bool = False
    ) -> jnp.ndarray:
        return self.layers(
            hidden_states=inputs_embeds,
            attention_mask=attention_mask,
            deterministic=deterministic,
            audio_hidden_state=audio_hidden_state,
            cross_attention_mask=cross_attention_mask,
            decode=decode
        )


class CLIPTextModel(nn.Module):
    config: CLIPTextConfig
    scan: bool = True

    def setup(self):
        self.embeddings = FlaxCLIPTextEmbeddings(self.config)
        self.encoder = FlaxCLIPEncoder(self.config, scan=self.scan)
        self.final_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.config.dtype)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
        position_ids: Optional[jnp.ndarray] = None,
        audio_hidden_state: Optional[jnp.ndarray] = None,
        cross_attention_mask: Optional[jnp.ndarray] = None,
        is_train: bool = True,
        decode: bool = False
    ) -> jnp.ndarray:
        
        if position_ids is None:
            position_ids = repeat(jnp.arange(input_ids.shape[-1]), 's -> b s', b=input_ids.shape[0])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            deterministic=not is_train,
            audio_hidden_state=audio_hidden_state,
            cross_attention_mask=cross_attention_mask,
            decode=decode
        )

        last_hidden_state = self.final_layer_norm(last_hidden_state)

        eos_inds = jnp.argmax(input_ids == self.config.eos_token_id, axis=-1)
        text_features = jnp.take_along_axis(last_hidden_state, indices=eos_inds[:, None, None], axis=-2)[:, 0]

        return text_features, last_hidden_state


class CLIPTextDecoder(nn.Module):
    config: CLIPTextConfig
    scan: bool = True

    def setup(self):
        self.encoder = FlaxCLIPEncoder(self.config, scan=self.scan)
        self.final_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.config.dtype)

        self.decoder_proj = nn.Dense(
            self.config.vocab_size, 
            dtype=self.config.dtype, 
            kernel_init=jax.nn.initializers.normal(),
            name='decoder_proj')


    def __call__(
        self,
        text_hidden_state: jnp.ndarray,
        attention_mask: jnp.ndarray,
        audio_hidden_state: jnp.ndarray,
        audio_mask: jnp.ndarray,
        is_train: bool = True,
        decode: bool = False
    ) -> jnp.ndarray:
        

        last_hidden_state = self.encoder(
            inputs_embeds=text_hidden_state,
            attention_mask=attention_mask,
            deterministic=not is_train,
            audio_hidden_state=audio_hidden_state,
            cross_attention_mask=audio_mask,
            decode=decode
        )

        last_hidden_state = self.final_layer_norm(last_hidden_state)
        logits = self.decoder_proj(last_hidden_state)

        return logits


class CLIPTextMatch(nn.Module):
    config: CLIPTextConfig
    causal: bool = True
    scan: bool = True

    def setup(self):
        self.encoder = FlaxCLIPEncoder(self.config, causal=self.causal, scan=self.scan)
        self.final_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.config.dtype)

        self.match_proj = nn.Dense(
            2, 
            dtype=self.config.dtype, 
            kernel_init=jax.nn.initializers.normal(),
            name='match_proj')


    def __call__(
        self,
        text_hidden_state: jnp.ndarray,
        attention_mask: jnp.ndarray,
        input_ids: jnp.ndarray,
        audio_hidden_state: jnp.ndarray,
        audio_mask: jnp.ndarray,
        is_train: bool = True,
    ) -> jnp.ndarray:

        last_hidden_state = self.encoder(
            inputs_embeds=text_hidden_state,
            attention_mask=attention_mask,
            deterministic=not is_train,
            audio_hidden_state=audio_hidden_state,
            cross_attention_mask=audio_mask,
        )

        last_hidden_state = self.final_layer_norm(last_hidden_state)

        eos_inds = jnp.argmax(input_ids == self.config.eos_token_id, axis=-1)
        text_features = jnp.take_along_axis(last_hidden_state, indices=eos_inds[:, None, None], axis=-2)[:, 0]
        match_logits = self.match_proj(text_features)

        return match_logits





def clip_update_pretrained_parameters(
    params: PyTreeDef,
    pretrained_path: str,
    prefix: Optional[Tuple[str]] = None,
) -> PyTreeDef:

    from flax.core import freeze, unfreeze
    from flax.traverse_util import flatten_dict, unflatten_dict

    from utils import load_params

    pretrained_weights = load_params(pretrained_path)

    params = flatten_dict(unfreeze(params))
    pretrained_weights = flatten_dict(pretrained_weights)

    if prefix is None:
        prefix = tuple()

    if any([prefix + ('encoder', 'layers', 'ScanFlaxCLIPEncoderLayer_0') == k[:3+len(prefix)] for k in params.keys()]): # check if scan
        scan_params = {}
        key_list = list(pretrained_weights.keys())
        for key in key_list:
            if key[:2] == ('encoder', 'layers'):
                k = key[:2]+('ScanFlaxCLIPEncoderLayer_0',) + key[3:]
                if k not in scan_params:
                    scan_params[k] = []
                scan_params[k].append(pretrained_weights[key])
                pretrained_weights.pop(key)

        key_list_scan = list(scan_params.keys())
        for key in key_list_scan:
            pretrained_weights[key] = jnp.stack(scan_params[key], axis=0)
            scan_params.pop(key)

    def _update_params(params: PyTreeDef, key: Tuple, weight: Any) -> PyTreeDef:
        key = prefix+key
        if key not in params.keys():
            print(f'Custom model is missing pretrained_key: {key}')
        else:
            if weight.shape != params[key].shape:
                print(weight.shape, params[key].shape)
                raise ValueError(f'Shape mismatch between params for key {key}')
            # print(f'Updated: {key}')
            params[key] = jnp.asarray(weight)

        return params

    for k in pretrained_weights.keys():
        params = _update_params(
            params, key=k, weight=pretrained_weights[k] 
        )

    params = freeze(unflatten_dict(params))

    return params



def save_pretrained_clip_params(
    save_model_path: Union[str, os.PathLike],
    save_tokenizer_path: Optional[Union[str, os.PathLike]] = None
):
    """Saves params and tokenizer"""
    from transformers import FlaxCLIPTextModel, CLIPTokenizerFast

    from utils import save_params

    model = FlaxCLIPTextModel.from_pretrained('openai/clip-vit-base-patch32')
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    save_params(model.params['text_model'], save_model_path)

    if save_tokenizer_path is not None:
        tokenizer.save_pretrained(save_tokenizer_path)

