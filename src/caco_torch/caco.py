"""
PyTorch CACO (Contrastive Audio-Caption Optimization) model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from .audio_models.mae import AudioEncoder, AudioTransformerConfig
from .text_models.roberta import RobertaModel, RobertaDecoder, RobertaConfig


NORM_EPS = 1e-10


@dataclass
class CACOConfig:
    projection_size: int = 768
    num_attention_pool_heads: int = 2
    logit_scale_init_value: float = 2.6592


class AudioAttentionPooler(nn.Module):
    """
    Attention pooler for audio embeddings.
    Matches JAX implementation in caco.py.
    """
    def __init__(self, hidden_size: int, num_heads: int, projection_size: Optional[int] = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Dense_0: projects to 2 * hidden_size for k, v
        self.kv_proj = nn.Linear(hidden_size, 2 * hidden_size)
        # Dense_1: output projection
        self.out_proj = nn.Linear(hidden_size, projection_size or hidden_size)
        # Query parameter
        self.query = nn.Parameter(torch.randn(hidden_size) * 0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # hidden_states: (batch, seq, hidden)
        batch_size = hidden_states.shape[0]

        # Project to k, v
        kv = self.kv_proj(hidden_states)
        k, v = kv.chunk(2, dim=-1)

        # Reshape q, k, v for multi-head attention
        # q: (hidden,) -> (num_heads, head_dim)
        q = self.query.view(self.num_heads, self.head_dim)
        # k, v: (batch, seq, hidden) -> (batch, seq, num_heads, head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)

        # Attention: q @ k.T / sqrt(d)
        # attn_weights: (batch, num_heads, seq)
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_weights = torch.einsum('hd,bjhd->bhj', q * scale, k)

        if mask is not None:
            # mask: (batch, seq) where 1 = keep, 0 = ignore
            attn_weights = attn_weights.masked_fill(mask[:, None, :] == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        # output: (batch, num_heads, head_dim)
        output = torch.einsum('bhj,bjhd->bhd', attn_weights, v)

        # Reshape and project
        output = output.view(batch_size, -1)  # (batch, hidden)
        output = self.out_proj(output)

        return output


class CACO(nn.Module):
    """
    CACO model combining audio encoder, text encoder, and optional decoder.
    For inference only.
    """
    def __init__(
        self,
        audio_config: AudioTransformerConfig,
        text_config: RobertaConfig,
        caco_config: CACOConfig,
        decoder_config: Optional[RobertaConfig] = None
    ):
        super().__init__()
        self.audio_config = audio_config
        self.text_config = text_config
        self.caco_config = caco_config

        # Audio encoder
        self.audio_module = AudioEncoder(audio_config)

        # Audio attention pooler
        self.audio_attention_pool = AudioAttentionPooler(
            hidden_size=audio_config.hidden_size,
            num_heads=caco_config.num_attention_pool_heads,
            projection_size=caco_config.projection_size
        )

        # Text encoder
        self.text_module = RobertaModel(text_config)

        # Text projection (if projection_size is set)
        self.text_proj = nn.Linear(text_config.hidden_size, caco_config.projection_size)

        # Logit scale
        self.logit_scale = nn.Parameter(torch.tensor(caco_config.logit_scale_init_value))

        # Optional decoder
        self.decoder_module = None
        if decoder_config is not None:
            self.decoder_module = RobertaDecoder(decoder_config)

    def get_audio_embedding(
        self,
        audio_patches: torch.Tensor,
        audio_time_inds: torch.Tensor,
        audio_freq_inds: torch.Tensor,
        audio_mask: torch.Tensor,
        deterministic: bool = True,
        return_hidden_state: bool = True,
        normalize: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get audio embedding from patches."""
        audio_hidden_state = self.audio_module(
            x=audio_patches,
            time_inds=audio_time_inds,
            freq_inds=audio_freq_inds,
            mask=audio_mask,
            deterministic=deterministic
        )

        audio_embedding = self.audio_attention_pool(audio_hidden_state, audio_mask)

        if normalize:
            # Match JAX: add NORM_EPS before computing norm
            audio_embedding = audio_embedding / torch.norm(audio_embedding + NORM_EPS, dim=-1, keepdim=True)

        if return_hidden_state:
            return audio_embedding, audio_hidden_state
        return audio_embedding

    def get_text_embedding(
        self,
        text_input_ids: torch.Tensor,
        text_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        return_hidden_state: bool = True,
        normalize: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get text embedding from input IDs."""
        text_embedding, text_hidden_state = self.text_module(
            input_ids=text_input_ids,
            attention_mask=text_mask,
            position_ids=position_ids,
            deterministic=deterministic
        )

        text_embedding = self.text_proj(text_embedding)

        if normalize:
            # Match JAX: add NORM_EPS before computing norm
            text_embedding = text_embedding / torch.norm(text_embedding + NORM_EPS, dim=-1, keepdim=True)

        if return_hidden_state:
            return text_embedding, text_hidden_state
        return text_embedding

    def get_contrastive_logits(
        self,
        audio_patches: torch.Tensor,
        audio_time_inds: torch.Tensor,
        audio_freq_inds: torch.Tensor,
        audio_mask: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_mask: torch.Tensor,
        deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute audio-text and text-audio contrastive logits."""
        audio_embedding = self.get_audio_embedding(
            audio_patches=audio_patches,
            audio_time_inds=audio_time_inds,
            audio_freq_inds=audio_freq_inds,
            audio_mask=audio_mask,
            deterministic=deterministic,
            return_hidden_state=False,
            normalize=True
        )

        text_embedding = self.get_text_embedding(
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            deterministic=deterministic,
            return_hidden_state=False,
            normalize=True
        )

        scale = torch.exp(self.logit_scale)
        at_logits = scale * audio_embedding @ text_embedding.T
        ta_logits = scale * text_embedding @ audio_embedding.T

        return at_logits, ta_logits

    def get_decoder_logits(
        self,
        audio_hidden_state: torch.Tensor,
        audio_mask: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_mask: torch.Tensor,
        deterministic: bool = True
    ) -> torch.Tensor:
        """Get decoder logits for captioning."""
        if self.decoder_module is None:
            raise ValueError("Decoder module not initialized")

        _, text_hidden_state = self.get_text_embedding(
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            deterministic=deterministic
        )

        logits = self.decoder_module(
            text_hidden_state=text_hidden_state,
            attention_mask=text_mask,
            audio_hidden_state=audio_hidden_state,
            audio_mask=audio_mask,
            deterministic=deterministic
        )

        return logits

    def forward(
        self,
        audio_patches: torch.Tensor,
        audio_time_inds: torch.Tensor,
        audio_freq_inds: torch.Tensor,
        audio_mask: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_mask: torch.Tensor,
        deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning contrastive logits."""
        return self.get_contrastive_logits(
            audio_patches=audio_patches,
            audio_time_inds=audio_time_inds,
            audio_freq_inds=audio_freq_inds,
            audio_mask=audio_mask,
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            deterministic=deterministic
        )


def create_caco_model() -> CACO:
    """Create a CACO model with default configuration matching the checkpoint."""
    audio_config = AudioTransformerConfig(
        hidden_size=768,
        num_layers=12,
        num_heads=8,
        intermediate_size=3072,
        patch_size=256,
        max_time_ind=512,
        num_freq_patches=8,
        dropout_rate=0.0,
        drop_path_rate=0.0
    )

    text_config = RobertaConfig(
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        type_vocab_size=1,
        layer_norm_eps=1e-5,
        pad_token_id=1
    )

    caco_config = CACOConfig(
        projection_size=768,
        num_attention_pool_heads=2,
        logit_scale_init_value=2.6592
    )

    decoder_config = RobertaConfig(
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=4,  # Decoder has 4 layers
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        type_vocab_size=1,
        layer_norm_eps=1e-5,
        pad_token_id=1
    )

    return CACO(
        audio_config=audio_config,
        text_config=text_config,
        caco_config=caco_config,
        decoder_config=decoder_config
    )
