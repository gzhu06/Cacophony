# MODIFIED FROM # https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/roberta/modeling_flax_roberta.py#L992
# to be standalone. implements unimodal and multimodal models.




# Copyright 2021 The Google Flax Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Callable, Optional, Tuple

import numpy as np

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

from functools import partial

from flax import struct

PyTreeDef = type(jax.tree_util.tree_structure(None))




@struct.dataclass
class RobertaConfig:
    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = 'gelu'
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    initializer_range: int = 0.02
    layer_norm_eps: float = 1e-05
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    position_embedding_type: str = 'absolute'
    # use_cache=True,
    # classifier_dropout=None,
    



ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),
    # "quick_gelu": quick_gelu
}


def create_position_ids_from_input_ids(input_ids, padding_idx):
    mask = (input_ids != padding_idx).astype("i4")

    if mask.ndim > 2:
        mask = mask.reshape((-1, mask.shape[-1]))
        incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask
        incremental_indices = incremental_indices.reshape(input_ids.shape)
    else:
        incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask

    return incremental_indices.astype("i4") + padding_idx


class FlaxRobertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids: jnp.ndarray, token_type_ids: jnp.ndarray, position_ids: jnp.ndarray, attention_mask: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxRobertaSelfAttention(nn.Module):
    config: RobertaConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads` "
                "                   : {self.config.num_attention_heads}"
            )

        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.hidden_size,))

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        key_value_states: Optional[jnp.array] = None,
        deterministic=True,
        decode: bool = False
    ) -> jnp.ndarray:

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        batch_size = hidden_states.shape[0]

        query_states = self.query(hidden_states)
        if is_cross_attention:
            assert not self.causal
            key_states = self.key(key_value_states)
            value_states = self.value(key_value_states)
            query_states = self._split_heads(query_states)
            key_states = self._split_heads(key_states)
            value_states = self._split_heads(value_states)
            if attention_mask is not None:
                attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
                attention_bias = lax.select(
                    attention_mask > 0,
                    jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                    jnp.full(attention_mask.shape, -1e10).astype(self.dtype),
                )
        else:
            key_states = self.key(hidden_states)
            value_states = self.value(hidden_states)
            
            query_states = self._split_heads(query_states)
            key_states = self._split_heads(key_states)
            value_states = self._split_heads(value_states)

            if self.causal and not decode: # need to combine causal mask with attention mask
                query_length, key_length = query_states.shape[1], key_states.shape[1]
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]
                causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
                if attention_mask is not None:
                    attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
                    attention_mask = combine_masks(attention_mask, causal_mask)
                else:
                    attention_mask = causal_mask
            elif decode: # https://github.com/google/flax/blob/main/flax/linen/attention.py#L195 (# decode (implied causal) uses cache for fast generation)
                assert self.causal
                
                # detect if we're initializing by absence of existing cache data.

                if attention_mask is not None:
                    attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

                is_initialized = self.has_variable('cache', 'cached_key')
                cached_key = self.variable('cache', 'cached_key',
                    jnp.zeros, key_states.shape, key_states.dtype)
                cached_value = self.variable('cache', 'cached_value',
                    jnp.zeros, value_states.shape, value_states.dtype)
                cache_index = self.variable('cache', 'cache_index',
                    lambda: jnp.array(0, dtype=jnp.int32))
                
                if is_initialized:
                    *batch_dims, max_length, num_heads, depth_per_head = (
                        cached_key.value.shape)
                    # shape check of cached keys against query input
                    expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
                    if expected_shape != query_states.shape:
                        raise ValueError('Autoregressive cache shape error, '
                            'expected query shape %s instead got %s.' %
                            (expected_shape, query_states.shape))
                    # update key, value caches with our new 1d spatial slices
                    cur_index = cache_index.value
                    indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
                    key_states = jax.lax.dynamic_update_slice(cached_key.value, key_states, indices)
                    value_states = jax.lax.dynamic_update_slice(cached_value.value, value_states, indices)
                    cached_key.value = key_states
                    cached_value.value = value_states
                    cache_index.value = cache_index.value + 1
                    # causal mask for cached decoder self-attention:
                    # our single query position should only attend to those key
                    # positions that have already been generated and cached,
                    # not the remaining zero elements.
                    attention_mask = combine_masks(
                        attention_mask,
                        jnp.broadcast_to(jnp.arange(max_length) <= cur_index,
                            tuple(batch_dims) + (1, 1, max_length)))

            elif attention_mask is not None: # non-causal (and not decode), with attention mask passed in
                attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

            if attention_mask is not None:
                attention_bias = lax.select(
                    attention_mask > 0,
                    jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                    jnp.full(attention_mask.shape, -1e10).astype(self.dtype),
                )
            else:
                attention_bias = None

        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        return attn_output


class FlaxRobertaSelfOutput(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states: jnp.ndarray, input_tensor: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FlaxRobertaAttention(nn.Module):
    config: RobertaConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self = FlaxRobertaSelfAttention(self.config, causal=self.causal, dtype=self.dtype)
        self.output = FlaxRobertaSelfOutput(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        decode: bool = False
    ) -> jnp.ndarray:

        attn_output = self.self(
            hidden_states,
            attention_mask,
            key_value_states=key_value_states,
            deterministic=deterministic,
            decode=decode
        )
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        return hidden_states


class FlaxRobertaIntermediate(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class FlaxRobertaOutput(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states: jnp.ndarray, attention_output: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        return hidden_states


class FlaxRobertaLayer(nn.Module):
    config: RobertaConfig
    causal: bool = True
    dtype: jnp.dtype = jnp.float32
    scan: bool = True

    def setup(self):
        self.attention = FlaxRobertaAttention(self.config, causal=self.causal, dtype=self.dtype)
        self.intermediate = FlaxRobertaIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxRobertaOutput(self.config, dtype=self.dtype)
        self.crossattention = FlaxRobertaAttention(self.config, causal=False, dtype=self.dtype)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        decode: bool = False
    ) -> jnp.ndarray:

        attention_output = self.attention(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            decode=decode
        )

        if encoder_hidden_states is not None:
            attention_output = self.crossattention(
                attention_output,
                attention_mask=encoder_attention_mask,
                key_value_states=encoder_hidden_states,
                deterministic=deterministic,
                decode=decode
            )
            # attention_output = cross_attention_outputs

        hidden_states = self.intermediate(attention_output)
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        if self.scan:
            return hidden_states, None

        return hidden_states


class FlaxRobertaLayerCollection(nn.Module):
    config: RobertaConfig
    causal: bool = True
    dtype: jnp.dtype = jnp.float32
    scan: bool = True

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        decode: bool = False,
    ) -> jnp.ndarray:

        if self.scan:
            hidden_states, _ = nn.scan(
                FlaxRobertaLayer,
                variable_axes={'params': 0, 'cache': 0},
                split_rngs={'params': True, 'dropout': True},
                in_axes=nn.broadcast,
                length=self.config.num_hidden_layers,
            )(self.config, causal=self.causal, dtype=self.dtype, scan=True)(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                deterministic,
                decode
            )
        else:
            for i in range(self.config.num_hidden_layers):
                hidden_states = FlaxRobertaLayer(self.config, causal=self.causal, name=str(i), dtype=self.dtype, scan=False)(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    deterministic,
                    decode=decode
                )

        return hidden_states


class FlaxRobertaEncoder(nn.Module):
    config: RobertaConfig
    causal: bool = True
    dtype: jnp.dtype = jnp.float32
    scan: bool = True

    def setup(self):
        self.layer = FlaxRobertaLayerCollection(
            self.config,
            causal=self.causal,
            dtype=self.dtype,
            scan=self.scan
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        decode: bool = False
    ) -> jnp.ndarray:
        return self.layer(
            hidden_states,
            attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            deterministic=deterministic,
            decode=decode
        )


class AttentionPooler(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.key_proj = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.value_proj = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.query = self.param('attention_pool_query', jax.nn.initializers.normal(self.config.initializer_range), (1, self.config.hidden_size))

    def __call__(self, hidden_states: jnp.ndarray, attention_mask: Optional[jnp.ndarray]) -> jnp.ndarray:
        key = self.key_proj(hidden_states) / jnp.sqrt(self.query.shape[-1])
        value = self.value_proj(hidden_states)
        attn_weights = jnp.einsum('mh,bnh->bmn', self.query, key)
        if attention_mask is not None:
            big_neg = jnp.finfo(self.dtype).min
            attn_weights = jnp.where(attention_mask[:, None], attn_weights, big_neg)
        attn_weights = jax.nn.softmax(attn_weights)
        output = jnp.einsum('bmn,bnh->bmh', attn_weights, value)[:, 0]
        return output


class RobertaModel(nn.Module):
    """Unimodal (causal) model"""
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    scan: bool = True

    def setup(self):
        self.embeddings = FlaxRobertaEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxRobertaEncoder(
            self.config,
            dtype=self.dtype,
            scan=self.scan
        )
        self.pooler = AttentionPooler(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids: Optional[jnp.ndarray] = None,
        is_train: bool = True,
        decode: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        token_type_ids = jnp.zeros_like(input_ids)

        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        hidden_states = self.embeddings(
            input_ids, token_type_ids, position_ids, attention_mask, deterministic=not is_train
        )
        hidden_states = self.encoder(
            hidden_states,
            attention_mask,
            deterministic=not is_train,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            decode=decode
        )

        text_embedding = self.pooler(hidden_states, attention_mask) # TODO(jd) maybe can't pool for decode

        return text_embedding, hidden_states


class RobertaDecoder(nn.Module):
    """Multimodal (causal) decoder"""
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    scan: bool = True

    def setup(self):
        
        self.encoder = FlaxRobertaEncoder(
            self.config,
            dtype=self.dtype,
            scan=self.scan
        )

        self.decoder_proj = nn.Dense(
            self.config.vocab_size, 
            dtype=self.dtype, 
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

        hidden_states = self.encoder(
            text_hidden_state,
            attention_mask,
            deterministic=not is_train,
            encoder_hidden_states=audio_hidden_state,
            encoder_attention_mask=audio_mask,
            decode=decode
        )

        logits = self.decoder_proj(hidden_states)

        return logits

# class RobertaMatch(nn.Module):
#     """Multimodal match"""
#     config: RobertaConfig
#     causal: bool = True
#     dtype: jnp.dtype = jnp.float32
#     scan: bool = True

#     def setup(self):
        
#         self.encoder = FlaxRobertaEncoder(
#             self.config,
#             causal=self.causal,
#             dtype=self.dtype,
#             scan=self.scan
#         )

#         self.match_proj = nn.Dense(
#             2, 
#             dtype=self.dtype, 
#             kernel_init=jax.nn.initializers.normal(),
#             name='match_proj')

#         self.pooler = AttentionPooler(RobertaConfig, dtype=self.dtype)

#     def __call__(
#         self,
#         text_hidden_state: jnp.ndarray,
#         attention_mask: jnp.ndarray,
#         input_ids: jnp.ndarray, # not used here but needed to match other text models 
#         audio_hidden_state: jnp.ndarray,
#         audio_mask: jnp.ndarray,
#         is_train: bool = True,
#     ) -> jnp.ndarray:

#         hidden_states = self.encoder(
#             text_hidden_state,
#             attention_mask,
#             deterministic=not is_train,
#             encoder_hidden_states=audio_hidden_state,
#             encoder_attention_mask=audio_mask,
#         )


#         match_logits = self.match_proj(self.pooler(hidden_states, attention_mask))

#         return match_logits





def roberta_update_pretrained_parameters(
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

    if any([prefix + ('encoder', 'layer', 'ScanFlaxRobertaLayer_0') == k[:3+len(prefix)] for k in params.keys()]): # check if scan
        scan_params = {}
        key_list = list(pretrained_weights.keys())
        for key in key_list:
            if key[:2] == ('encoder', 'layer'):
                k = key[:2]+('ScanFlaxRobertaLayer_0',) + key[3:]
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



def save_pretrained_roberta_params(
    save_model_path: str,
    save_tokenizer_path: Optional[str] = None
):
    """Saves params and tokenizer"""
    from transformers import FlaxRobertaModel, RobertaTokenizerFast

    from utils import save_params

    model = FlaxRobertaModel.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    save_params(model.params, save_model_path)

    if save_tokenizer_path is not None:
        tokenizer.save_pretrained(save_tokenizer_path)
