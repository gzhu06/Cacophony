# This contains the MAE ( Masked Autoencoder ) class
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.training import checkpoints
from flax.core import freeze, unfreeze

PyTreeDef = type(jax.tree_util.tree_structure(None))

@struct.dataclass
class AudioTransformerConfig:

    hidden_size: int
    num_layers: int
    num_heads: int

    intermediate_size: int

    patch_size: int # (width * height)
    max_time_ind: int
    num_freq_patches: int

    dropout_rate: float
    drop_path_rate: float

    dtype: jnp.dtype

@struct.dataclass
class AudioMAEConfig:
    encoder_config: AudioTransformerConfig # TODO (maybe separate class)
    decoder_config: AudioTransformerConfig


class DropPath(nn.Module):
    config: AudioTransformerConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:

        if deterministic or self.config.drop_path_rate == 0.:
            return x
        else:
            x = jnp.where(
                jax.random.uniform(self.make_rng('drop_path'), shape=(x.shape[0], *([1] * (x.ndim - 1))))  >= self.config.drop_path_rate,
                x / (1 - self.config.drop_path_rate),
                jnp.zeros_like(x)
            )
            return x
        
class MLP(nn.Module):
    config: AudioTransformerConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True
    ) -> jnp.ndarray:

        x = nn.Dense(self.config.intermediate_size, dtype=self.config.dtype)(x)
        x = nn.silu(x)
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.config.hidden_size, dtype=self.config.dtype)(x)
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic=deterministic)
        return x

class AudioEncoderLayer(nn.Module):
    config: AudioTransformerConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray,
        deterministic: bool = True
    ) -> jnp.ndarray:

        cfg = self.config

        attn_mask = mask[..., None, :]

        h = nn.LayerNorm(dtype=cfg.dtype)(x)
        h = nn.MultiHeadDotProductAttention(num_heads=cfg.num_heads, 
            dtype=cfg.dtype, dropout_rate=cfg.dropout_rate)(h, h, attn_mask[..., None, :, :], deterministic=deterministic)
        x = x + DropPath(cfg)(h, deterministic=deterministic) 
        # x = nn.Dropout(self.config.dropout_rate)(x, deterministic=deterministic)
        
        h = nn.LayerNorm(dtype=cfg.dtype)(x)
        h = MLP(cfg)(h)
        x = x + DropPath(cfg)(h, deterministic=deterministic)
        # x = nn.Dropout(self.config.dropout_rate)(x, deterministic=deterministic)

        return x
    
def get_sin_cos_pos_embed(position_ids: jnp.ndarray, 
                          embed_size: int) -> jnp.ndarray: # TODO check if good
    assert embed_size % 2 == 0
    pos_embed = position_ids[..., None] * jnp.exp(2 * jnp.arange(embed_size // 2, dtype=jnp.float32) * -jnp.log(10000.) / embed_size)
    pos_embed = jnp.concatenate([jnp.sin(pos_embed), jnp.cos(pos_embed)], axis=-1)
    return pos_embed

class AudioEncoder(nn.Module):
    config: AudioTransformerConfig

    @nn.compact # initialises the weights       
    def __call__(
        self,
        x: jnp.ndarray, # (batch_size, patch_seq, in_patch_size)
        time_inds: jnp.ndarray,
        freq_inds: jnp.ndarray,
        mask: jnp.ndarray, # (batch_size, patch_seq),
        deterministic: bool = True
    ) -> jnp.ndarray:

        cfg = self.config

        x = nn.Dense(cfg.hidden_size, dtype=cfg.dtype)(x)

        time_pos_emb = get_sin_cos_pos_embed(time_inds, x.shape[-1]) 
        freq_pos_emb = self.param('freq_positional_embedding', 
                                nn.initializers.normal(stddev=0.02), 
                                (self.config.num_freq_patches, x.shape[-1]))
        x = x + time_pos_emb
        x = x + jnp.take_along_axis(freq_pos_emb[None], freq_inds[..., None], axis=-2)

        # x = nn.Dropout(self.config.dropout_rate)(x, deterministic=deterministic)

        # TODO scan
        for _ in range(cfg.num_layers):
            x = AudioEncoderLayer(cfg)(x, mask, deterministic=deterministic)

        x = nn.LayerNorm(dtype=cfg.dtype)(x)

        return x

        
# Decoder code

class AudioDecoder(nn.Module):
    config: AudioTransformerConfig
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray, # (batch_size, patch_seq, hidden)
        mask: jnp.ndarray,
        time_inds: jnp.ndarray,
        freq_inds: jnp.ndarray,
        restore_time_inds: jnp.ndarray,
        restore_freq_inds: jnp.ndarray,
        restore_mask: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        
        cfg = self.config

        x = nn.Dense(cfg.hidden_size, dtype=cfg.dtype)(x)

        time_pos_emb = get_sin_cos_pos_embed(time_inds, x.shape[-1])
        freq_pos_emb = self.param('freq_positional_embedding', 
                                  nn.initializers.normal(stddev=0.02),
                                  (self.config.num_freq_patches, x.shape[-1]))
        x = x + time_pos_emb
        x = x + jnp.take_along_axis(freq_pos_emb[None], freq_inds[..., None], axis=-2)
        x_restore = self.param('restore_patch', nn.initializers.normal(stddev=0.02), (x.shape[-1],))[None, None]
        restore_time_pos_emb = get_sin_cos_pos_embed(restore_time_inds, x.shape[-1])
        x_restore = x_restore + restore_time_pos_emb
        x_restore = x_restore + jnp.take_along_axis(freq_pos_emb[None], restore_freq_inds[..., None], axis=-2)

        x = jnp.concatenate([x, x_restore], axis=-2)
        # x = nn.Dropout(self.config.dropout_rate)(x, deterministic=deterministic)

        mask = jnp.concatenate([mask, restore_mask], axis=-1)

        # TODO scan
        for _ in range(cfg.num_layers):
            x = AudioEncoderLayer(cfg)(x, mask, deterministic=deterministic)

        x = nn.LayerNorm(dtype=cfg.dtype)(x)

        x = nn.Dense(cfg.patch_size, dtype=cfg.dtype)(x)

        return x

class AudioMAE(nn.Module):
    config: AudioMAEConfig

    @nn.compact        
    def __call__(
        self, 
        x: jnp.ndarray, # (batch_size, patch_seq, in_patch)
        mask: jnp.ndarray,
        time_inds: jnp.ndarray,
        freq_inds: jnp.ndarray,
        restore_time_inds: jnp.ndarray,
        restore_freq_inds: jnp.ndarray,
        restore_mask: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray: 

        x = AudioEncoder(self.config.encoder_config)(
            x,
            mask=mask,
            time_inds=time_inds,
            freq_inds=freq_inds,
            deterministic=deterministic
        )

        x = AudioDecoder(self.config.decoder_config)(
            x,
            mask=mask,
            time_inds=time_inds,
            freq_inds=freq_inds,
            restore_time_inds=restore_time_inds,
            restore_freq_inds=restore_freq_inds,
            restore_mask=restore_mask,
            deterministic=deterministic
        )

        return x
    
def ast_update_pretrained_parameters(
    params: PyTreeDef,
    pretrained_path: str,
) -> PyTreeDef:
    
    params = unfreeze(params)
    params['audio_module'] = checkpoints.restore_checkpoint(pretrained_path, target=None)['0']['params']['AudioEncoder_0']
    return freeze(params)
