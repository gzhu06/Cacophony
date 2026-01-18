import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from einops import rearrange, repeat
from typing import Optional, Mapping, Tuple, Union

PyTreeDef = type(jax.tree_util.tree_structure(None))
NORM_EPS = 1e-10

@struct.dataclass
class CACOConfig:
    dtype: jnp.dtype
    logit_scale_init_value: float
    num_attention_pool_heads: int
    use_decoder: bool
    projection_size: Optional[int] = None

class AttentionPooler(nn.Module):
    caco_config: CACOConfig

    @nn.compact
    def __call__(
        self,
        audio_hidden_state: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:

        k, v = jnp.split(nn.Dense(
            2 * audio_hidden_state.shape[-1],
            dtype=self.caco_config.dtype,
        )(audio_hidden_state), 2, axis=-1) # TODO(jd) init?

        q = self.param('query', jax.nn.initializers.normal(0.02), (audio_hidden_state.shape[-1],))

        q = rearrange(q, '(m d) -> m d', m=self.caco_config.num_attention_pool_heads)
        k = rearrange(k, '... (m d) -> ... m d', m=self.caco_config.num_attention_pool_heads)
        v = rearrange(v, '... (m d) -> ... m d', m=self.caco_config.num_attention_pool_heads)

        attn_weights = jnp.einsum('hd, bjhd -> bhj', q / jnp.sqrt(q.shape[-1]), k)
        if mask is not None:
            big_neg = jnp.finfo(self.caco_config.dtype).min
            attn_weights = jnp.where(mask[:, None], attn_weights, big_neg)
        
        attn_weights = jax.nn.softmax(attn_weights)

        output = jnp.einsum('bhj, bjhd- > bhd', attn_weights, v)

        output = nn.Dense(
            self.caco_config.projection_size or audio_hidden_state.shape[-1],
            dtype=self.caco_config.dtype
        )(rearrange(output, 'b h d -> b (h d)'))

        return output

class CACO(nn.Module):

    caco_config: CACOConfig
    audio_module: nn.Module
    text_module: nn.Module
    decoder_module: Optional[nn.Module]

    def setup(self):
        self.logit_scale = self.param('logit_scale', lambda _: jnp.array(self.caco_config.logit_scale_init_value))
        proj_size = self.caco_config.projection_size
        self.text_proj = nn.Dense(
            proj_size,
            dtype=self.caco_config.dtype
        )
        self.audio_attention_pool = AttentionPooler(self.caco_config)

    def get_audio_embedding(
        self,
        audio_patches: jnp.ndarray,
        audio_time_inds: jnp.ndarray,
        audio_freq_inds: jnp.ndarray,
        audio_mask: jnp.ndarray,
        deterministic: bool = False,
        return_hidden_state: bool = True,
        normalize: bool = False
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        audio_hidden_state = self.audio_module(
            x=audio_patches,
            time_inds=audio_time_inds,
            freq_inds=audio_freq_inds,
            mask=audio_mask,
            deterministic=deterministic
        )
        audio_embedding = self.audio_attention_pool(audio_hidden_state, audio_mask)
        if normalize:
            audio_embedding = audio_embedding / jnp.linalg.norm(audio_embedding + NORM_EPS, axis=-1, keepdims=True)
        
        if return_hidden_state:
            return (audio_embedding, audio_hidden_state)

        return audio_embedding

        
    def get_text_embedding(
        self,
        text_input_ids: jnp.ndarray,
        text_mask: jnp.ndarray,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
        return_hidden_state: bool = True,
        decode: bool = False,
        normalize: bool = False
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:

        text_embedding, text_hidden_state = self.text_module(
            input_ids=text_input_ids,
            attention_mask=text_mask,
            position_ids=position_ids,
            is_train=not deterministic,
            decode=decode
        )
        if self.caco_config.projection_size is not None:
            text_embedding = self.text_proj(text_embedding)
        if normalize:
            text_embedding = text_embedding / jnp.linalg.norm(text_embedding + NORM_EPS, axis=-1, keepdims=True)
        if return_hidden_state:
            return (text_embedding, text_hidden_state)
        return text_embedding

    def get_next_decoder_logits(
        self,
        audio_hidden_state: jnp.ndarray,
        audio_mask: jnp.ndarray,
        text_input_ids: jnp.ndarray,
        text_mask: jnp.ndarray,
        position_ids: jnp.ndarray,
        deterministic: bool = False,
    ) -> jnp.ndarray:

        _, text_hidden_state = self.get_text_embedding(
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            decode=True,
        )
        logits = self.decoder_module(
            text_hidden_state=text_hidden_state,
            attention_mask=text_mask,
            audio_hidden_state=audio_hidden_state,
            audio_mask=audio_mask,
            is_train=not deterministic,
            decode=True,
        )[:, 0, :]
        return logits



def decode(
    caco_model: CACO,
    params: PyTreeDef,
    audio_batch: PyTreeDef,
    max_length: int,
    temperature: float,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    rng: jax.Array
) -> jnp.ndarray:

    def _cond_fn(val):
        return (val[2] < max_length - 1) & (jax.lax.psum(jnp.sum(val[3]), axis_name='dp') > 0)


    _, audio_hidden_state = caco_model.apply(
        {'params': params},
        **audio_batch,
        deterministic=True,
        method=caco_model.get_audio_embedding
    )


    def _loop_fn(val):
        cache, input_ids, index, is_generating = val
        current_input_ids = input_ids[:, index]

        current_input_ids = current_input_ids * is_generating + (1 - is_generating) * pad_id# TODO(jd) check

        logits, variables = caco_model.apply(
            {'params': params, 'cache': cache},
            audio_hidden_state=audio_hidden_state,
            audio_mask=audio_batch['audio_mask'],
            text_input_ids=current_input_ids[..., None],
            text_mask=jnp.ones((batch_size, 1)),
            position_ids=jnp.zeros((batch_size, 1)) + index,
            deterministic=True, 
            method=caco_model.get_next_decoder_logits,
            mutable=['cache'],
        )

        updated_cache = variables['cache']

        sampled_ids = jax.random.categorical(jax.random.fold_in(rng, index), logits/temperature, axis=-1)
        updated_input_ids = input_ids.at[:, index+1].set(sampled_ids * is_generating)

        is_generating = is_generating * (sampled_ids != eos_id)
        return (updated_cache, updated_input_ids, index + 1, is_generating)

        
    batch_size = audio_batch['audio_patches'].shape[0]

    def fold_train_rngs(rng: jax.random.KeyArray) -> Mapping[str, jax.random.KeyArray]:
        return {'params': jax.random.fold_in(rng, 0), 'dropout': jax.random.fold_in(rng, 1), 'sample': jax.random.fold_in(rng, 2)}

    rngs = fold_train_rngs(rng)

    initial_variables = caco_model.init(
        rngs=rngs, 
        audio_hidden_state=audio_hidden_state,
        audio_mask=audio_batch['audio_mask'],
        text_input_ids=jnp.ones((batch_size, max_length)),
        text_mask=jnp.ones((batch_size, max_length)),
        position_ids=repeat(jnp.arange(max_length), 's -> b s', b=batch_size),
        deterministic=True, 
        method=caco_model.get_next_decoder_logits
    )

    cache = initial_variables['cache']
    input_ids = jnp.zeros((batch_size, max_length), dtype=jnp.int32)
    input_ids = input_ids.at[:, 0].set(bos_id)
    is_generating = jnp.ones((batch_size,), dtype=jnp.bool_)
    init_val = (cache, input_ids, jnp.array(0, dtype=jnp.int32), is_generating)
    end_val = jax.lax.while_loop(_cond_fn, _loop_fn, init_val)
    
    return end_val[1]