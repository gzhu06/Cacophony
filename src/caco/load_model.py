import jax.numpy as jnp
from flax.training import checkpoints
from .text_models.roberta_text_model import RobertaConfig, RobertaModel, RobertaDecoder
from .audio_models.mae import AudioEncoder, AudioTransformerConfig
from transformers import RobertaTokenizerFast
import jax
from .caco import CACO, CACOConfig

def count_params(params) -> int:
    return jax.tree_util.tree_reduce(lambda x, y: x + y.size, params, initializer=0)

def load_caco(ckpt_path, use_decoder=True):
    
    # load caco state dict
    caco_state_dict = checkpoints.restore_checkpoint(ckpt_path, target=None)
    caco_params = caco_state_dict['0']['params']

    print(count_params(caco_params['audio_module'])/1e6, 
          count_params(caco_params['decoder_module'])/1e6, 
          count_params(caco_params['text_module'])/1e6)

    # text model configs
    text_module = RobertaModel(RobertaConfig())
    decoder_module = RobertaDecoder(RobertaConfig(num_hidden_layers=4))
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    # audio model configs
    encoder_config = AudioTransformerConfig(
        hidden_size=768,
        num_layers=12,
        num_heads=8,
        intermediate_size=3072,
        patch_size=(16 * 16), # (width * height)
        max_time_ind=10000,
        num_freq_patches=8,
        dropout_rate=0.0,
        drop_path_rate=0.0,
        dtype=jnp.float32,
    )
    audio_module = AudioEncoder(encoder_config)

    # CACO config
    caco_config = CACOConfig(
        dtype=jnp.float32,
        logit_scale_init_value=2.,        
        num_attention_pool_heads=8, 
        use_decoder=use_decoder,
        projection_size=768,
    )

    # caco model config
    caco_model = CACO(
        caco_config=caco_config,
        audio_module=audio_module,
        text_module=text_module,
        decoder_module=decoder_module,
    )

    caco_model_dict = {'tokenizer':tokenizer, 
                       'caco_model':caco_model, 
                       'caco_params':caco_params}

    return caco_model_dict

def load_audiomae(ckpt_path):
    
    # load audiomae state dict
    audiomae_state_dict = checkpoints.restore_checkpoint(ckpt_path, target=None)
    audiomae_params = audiomae_state_dict['0']['params']['AudioEncoder_0']
    
    encoder_config = AudioTransformerConfig(
        hidden_size=768,
        num_layers=12,
        num_heads=8,
        intermediate_size=3072,
        patch_size=(16 * 16), # (width * height)
        max_time_ind=1000,
        num_freq_patches=8,
        dropout_rate=0.0,
        drop_path_rate=0.0,
        dtype=jnp.float32,
    )
    
    audiomae_model = AudioEncoder(encoder_config)
    
    audiomae_model_dict = {'audiomae_model':audiomae_model, 
                           'audiomae_params':audiomae_params}
    
    return audiomae_model_dict
