"""
PyTorch RoBERTa text model matching the JAX CACO checkpoint structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class RobertaConfig:
    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 1


class RobertaEmbeddings(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        deterministic: bool = True
    ) -> torch.Tensor:
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        if not deterministic:
            embeddings = self.dropout(embeddings)
        return embeddings


class RobertaSelfAttention(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        deterministic: bool = True
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        is_cross_attention = key_value_states is not None

        q = self.query(hidden_states)
        if is_cross_attention:
            k = self.key(key_value_states)
            v = self.value(key_value_states)
        else:
            k = self.key(hidden_states)
            v = self.value(hidden_states)

        # Reshape to (batch, heads, seq, head_dim)
        q = q.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        if not deterministic:
            attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        return attn_output


class RobertaSelfOutput(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
        deterministic: bool = True
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        if not deterministic:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RobertaAttention(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.self = RobertaSelfAttention(config)
        self.output = RobertaSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        deterministic: bool = True
    ) -> torch.Tensor:
        attn_output = self.self(
            hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            deterministic=deterministic
        )
        output = self.output(attn_output, hidden_states, deterministic=deterministic)
        return output


class RobertaIntermediate(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        return hidden_states


class RobertaOutput(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_output: torch.Tensor,
        deterministic: bool = True
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        if not deterministic:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        return hidden_states


class RobertaLayer(nn.Module):
    def __init__(self, config: RobertaConfig, has_cross_attention: bool = False):
        super().__init__()
        self.attention = RobertaAttention(config)
        self.has_cross_attention = has_cross_attention
        if has_cross_attention:
            self.crossattention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True
    ) -> torch.Tensor:
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic
        )

        if self.has_cross_attention and encoder_hidden_states is not None:
            attention_output = self.crossattention(
                attention_output,
                attention_mask=encoder_attention_mask,
                key_value_states=encoder_hidden_states,
                deterministic=deterministic
            )

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, deterministic=deterministic)
        return layer_output


class RobertaEncoder(nn.Module):
    def __init__(self, config: RobertaConfig, has_cross_attention: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            RobertaLayer(config, has_cross_attention=has_cross_attention)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                deterministic=deterministic
            )
        return hidden_states


class AttentionPooler(nn.Module):
    """Attention pooler for text embeddings."""
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.attention_pool_query = nn.Parameter(torch.randn(1, config.hidden_size) * 0.02)
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # hidden_states: (batch, seq, hidden)
        key = self.key_proj(hidden_states) / (self.attention_pool_query.shape[-1] ** 0.5)
        value = self.value_proj(hidden_states)

        # attn_weights: (batch, 1, seq)
        attn_weights = torch.einsum('mh,bnh->bmn', self.attention_pool_query, key)

        if attention_mask is not None:
            # attention_mask: (batch, seq) where 1 = keep, 0 = ignore
            attn_weights = attn_weights.masked_fill(attention_mask[:, None, :] == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.einsum('bmn,bnh->bmh', attn_weights, value)[:, 0]  # (batch, hidden)
        return output


class RobertaModel(nn.Module):
    """RoBERTa model with causal attention and attention pooling."""
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config, has_cross_attention=False)
        self.pooler = AttentionPooler(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Create causal attention mask as boolean (True = can attend)
        # Shape: (seq, seq)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool))

        # Combine with padding mask
        # attention_mask: (batch, seq) where 1 = keep, 0 = ignore
        # padding_mask: (batch, 1, 1, seq)
        padding_mask = attention_mask[:, None, None, :].bool()

        # Combined mask: (batch, 1, seq, seq)
        combined_mask = causal_mask.unsqueeze(0).unsqueeze(0) & padding_mask

        # Convert to additive mask for attention scores
        # Where mask is False (can't attend), set to -inf
        attention_bias = torch.zeros_like(combined_mask, dtype=torch.float32)
        attention_bias = attention_bias.masked_fill(~combined_mask, float('-inf'))

        hidden_states = self.embeddings(
            input_ids,
            position_ids,
            deterministic=deterministic
        )

        hidden_states = self.encoder(
            hidden_states,
            attention_mask=attention_bias,
            deterministic=deterministic
        )

        pooled_output = self.pooler(hidden_states, attention_mask)

        return pooled_output, hidden_states


class RobertaDecoder(nn.Module):
    """RoBERTa decoder with cross-attention to audio."""
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.config = config
        self.encoder = RobertaEncoder(config, has_cross_attention=True)
        self.decoder_proj = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(
        self,
        text_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_hidden_state: torch.Tensor,
        audio_mask: torch.Tensor,
        deterministic: bool = True
    ) -> torch.Tensor:
        batch_size, seq_len, _ = text_hidden_state.shape

        # Create causal attention mask as boolean (True = can attend)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=text_hidden_state.device, dtype=torch.bool))

        # Combine with text padding mask
        padding_mask = attention_mask[:, None, None, :].bool()
        combined_mask = causal_mask.unsqueeze(0).unsqueeze(0) & padding_mask

        # Convert to additive mask
        self_attn_mask = torch.zeros_like(combined_mask, dtype=torch.float32)
        self_attn_mask = self_attn_mask.masked_fill(~combined_mask, float('-inf'))

        # Cross-attention mask for audio
        # audio_mask: (batch, audio_seq) where 1 = keep, 0 = ignore
        audio_mask_bool = audio_mask[:, None, None, :].bool()
        cross_attn_mask = torch.zeros_like(audio_mask_bool, dtype=torch.float32)
        cross_attn_mask = cross_attn_mask.masked_fill(~audio_mask_bool, float('-inf'))

        hidden_states = self.encoder(
            text_hidden_state,
            attention_mask=self_attn_mask,
            encoder_hidden_states=audio_hidden_state,
            encoder_attention_mask=cross_attn_mask,
            deterministic=deterministic
        )

        logits = self.decoder_proj(hidden_states)
        return logits
