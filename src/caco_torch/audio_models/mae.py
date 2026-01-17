import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class AudioTransformerConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    patch_size: int
    max_time_ind: int
    num_freq_patches: int
    dropout_rate: float
    drop_path_rate: float
    dtype: torch.dtype = torch.float32


@dataclass
class AudioMAEConfig:
    encoder_config: AudioTransformerConfig
    decoder_config: AudioTransformerConfig


class DropPath(nn.Module):
    def __init__(self, drop_path_rate: float):
        super().__init__()
        self.drop_path_rate = drop_path_rate

    def forward(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        if deterministic or self.drop_path_rate == 0.:
            return x

        keep_prob = 1 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = (random_tensor >= self.drop_path_rate).float()
        if keep_prob > 0:
            random_tensor = random_tensor / keep_prob
        return x * random_tensor


class MLP(nn.Module):
    def __init__(self, config: AudioTransformerConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        x = self.fc1(x)
        x = F.silu(x)
        x = self.dropout(x) if not deterministic else x
        x = self.fc2(x)
        x = self.dropout(x) if not deterministic else x
        return x


class AudioEncoderLayer(nn.Module):
    def __init__(self, config: AudioTransformerConfig):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attn = nn.MultiheadAttention(
            config.hidden_size,
            config.num_heads,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.drop_path1 = DropPath(config.drop_path_rate)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)
        self.drop_path2 = DropPath(config.drop_path_rate)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = True
    ) -> torch.Tensor:
        # Attention mask: mask shape is (batch, seq_len)
        # PyTorch expects attention mask where True means ignore
        # JAX uses mask where 1 means keep
        attn_mask = mask == 0  # Convert to PyTorch format (True = ignore)

        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=attn_mask, need_weights=False)
        x = x + self.drop_path1(h, deterministic=deterministic)

        h = self.norm2(x)
        h = self.mlp(h, deterministic=deterministic)
        x = x + self.drop_path2(h, deterministic=deterministic)

        return x


def get_sin_cos_pos_embed(position_ids: torch.Tensor, embed_size: int) -> torch.Tensor:
    assert embed_size % 2 == 0
    pos_embed = position_ids.unsqueeze(-1) * torch.exp(
        2 * torch.arange(embed_size // 2, dtype=torch.float32, device=position_ids.device) *
        -math.log(10000.) / embed_size
    )
    pos_embed = torch.cat([torch.sin(pos_embed), torch.cos(pos_embed)], dim=-1)
    return pos_embed


class AudioEncoder(nn.Module):
    def __init__(self, config: AudioTransformerConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.patch_size, config.hidden_size)
        self.freq_positional_embedding = nn.Parameter(
            torch.randn(config.num_freq_patches, config.hidden_size) * 0.02
        )
        self.layers = nn.ModuleList([
            AudioEncoderLayer(config) for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        time_inds: torch.Tensor,
        freq_inds: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = True
    ) -> torch.Tensor:
        x = self.input_proj(x)

        time_pos_emb = get_sin_cos_pos_embed(time_inds, x.shape[-1])
        freq_pos_emb = torch.gather(
            self.freq_positional_embedding.unsqueeze(0).expand(x.shape[0], -1, -1),
            1,
            freq_inds.long().unsqueeze(-1).expand(-1, -1, x.shape[-1])
        )
        x = x + time_pos_emb
        x = x + freq_pos_emb

        for layer in self.layers:
            x = layer(x, mask, deterministic=deterministic)

        x = self.norm(x)
        return x


class AudioDecoder(nn.Module):
    def __init__(self, config: AudioTransformerConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.freq_positional_embedding = nn.Parameter(
            torch.randn(config.num_freq_patches, config.hidden_size) * 0.02
        )
        self.restore_patch = nn.Parameter(torch.randn(config.hidden_size) * 0.02)
        self.layers = nn.ModuleList([
            AudioEncoderLayer(config) for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.patch_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_inds: torch.Tensor,
        freq_inds: torch.Tensor,
        restore_time_inds: torch.Tensor,
        restore_freq_inds: torch.Tensor,
        restore_mask: torch.Tensor,
        deterministic: bool = True
    ) -> torch.Tensor:
        x = self.input_proj(x)

        time_pos_emb = get_sin_cos_pos_embed(time_inds, x.shape[-1])
        freq_pos_emb = torch.gather(
            self.freq_positional_embedding.unsqueeze(0).expand(x.shape[0], -1, -1),
            1,
            freq_inds.long().unsqueeze(-1).expand(-1, -1, x.shape[-1])
        )
        x = x + time_pos_emb
        x = x + freq_pos_emb

        x_restore = self.restore_patch.unsqueeze(0).unsqueeze(0)
        restore_time_pos_emb = get_sin_cos_pos_embed(restore_time_inds, x.shape[-1])
        x_restore = x_restore + restore_time_pos_emb
        restore_freq_pos_emb = torch.gather(
            self.freq_positional_embedding.unsqueeze(0).expand(x.shape[0], -1, -1),
            1,
            restore_freq_inds.long().unsqueeze(-1).expand(-1, -1, x.shape[-1])
        )
        x_restore = x_restore + restore_freq_pos_emb

        x = torch.cat([x, x_restore], dim=1)
        mask = torch.cat([mask, restore_mask], dim=1)

        for layer in self.layers:
            x = layer(x, mask, deterministic=deterministic)

        x = self.norm(x)
        x = self.output_proj(x)

        return x


class AudioMAE(nn.Module):
    def __init__(self, config: AudioMAEConfig):
        super().__init__()
        self.config = config
        self.encoder = AudioEncoder(config.encoder_config)
        self.decoder = AudioDecoder(config.decoder_config)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_inds: torch.Tensor,
        freq_inds: torch.Tensor,
        restore_time_inds: torch.Tensor,
        restore_freq_inds: torch.Tensor,
        restore_mask: torch.Tensor,
        deterministic: bool = True
    ) -> torch.Tensor:
        x = self.encoder(
            x,
            time_inds=time_inds,
            freq_inds=freq_inds,
            mask=mask,
            deterministic=deterministic
        )

        x = self.decoder(
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
