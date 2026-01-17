#!/usr/bin/env python3
"""
Compute the embeddings for every task and store to disk.

One benefit of this approach is that since all embeddings are cached
as numpy arrays, the final training code can be pytorch-only,
regardless of whether the embedding model is tensorflow based.

"""

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import jax
import flax
import jax.numpy as jnp
from einops import rearrange
from src.caco.load_model import load_caco
from src.caco.caco_eval_utils import compute_mel_spec_audiomae
from src.caco.dataset import Batch, AudioMAEDatasetConfig, DatasetConfig, _dataset_process_map, _tokenize_and_numpy

PyTreeDef = type(jax.tree_util.tree_structure(None))
def get_train_input(
    batch: Batch
) -> PyTreeDef:
    batch = dict(
        audio_patches=batch.audio_patches,
        audio_time_inds=batch.audio_time_inds,
        audio_freq_inds=batch.audio_freq_inds,
        audio_mask=batch.audio_mask,
        text_input_ids=batch.text_input_ids,
        text_mask=batch.text_mask,
    )
    batch = jax.tree_util.tree_map(
        lambda x: rearrange(jnp.asarray(x), '(d b) ... -> d b ...', d=jax.local_device_count()),
        batch
    )
    return batch

class Embedding:
    """
    A wrapper class to help with loading embedding models and computing embeddings

    Args:
        module_name: the import name for the embedding module
        model_path: location to load the model from
    """

    def __init__(
        self,
        model_path: str = None,
        audio_max_len: float = 10,
        batch_size: int = 1,
        sample_rate: int = 16000):

        self.model_path = model_path
        self.sample_rate = sample_rate
        self.batch_size = batch_size

        # load model
        
        caco_model_dict = load_caco(model_path)
        dataconfig = AudioMAEDatasetConfig(audio_segment_len=int(audio_max_len*sample_rate))

        self.dataconfig = dataconfig
        self.caco_params = flax.jax_utils.replicate(caco_model_dict['caco_params'], 
                                                    devices=jax.local_devices())
        self.caco_model = caco_model_dict['caco_model']
        self.tokenizer = caco_model_dict['tokenizer']
        self.audio_max_len = audio_max_len
        
        # maximum usable patches
        max_patches = (dataconfig.audio_segment_len // dataconfig.spec_hop_length // dataconfig.time_patch_size) * (dataconfig.spec_num_mels // dataconfig.freq_patch_size)
        
        self.CACOdataConfig = DatasetConfig(batch_size=batch_size,
                                            patches_seq_len=max_patches,
                                            time_patch_size=16,
                                            freq_patch_size=16,
                                            max_text_len=77,
                                            synthetic_prob=0.8)

        def compute_audio_embedding(audio_batch, model_params):
            return self.caco_model.apply(
                {'params': model_params},
                audio_patches=audio_batch['audio_patches'],
                audio_time_inds=audio_batch['audio_time_inds'],
                audio_freq_inds=audio_batch['audio_freq_inds'],
                audio_mask=audio_batch['audio_mask'],
                deterministic=True,
                return_hidden_state=True,
                normalize=True,
                method=self.caco_model.get_audio_embedding,
            )

        self.a_apply = jax.pmap(compute_audio_embedding, axis_name='dp')

    def get_embedding_as_numpy(self, audiofile, embedding_type=None) -> np.ndarray:

        data_dict = {}
        data_dict['filename'] = audiofile

        audiowav, _ = tf.audio.decode_wav(tf.io.read_file(audiofile[0]))
        if 'passt' in self.model_path:
            audiowav = tfio.audio.resample(audiowav, rate_in=48000, 
                                           rate_out=32000)
            
        audio = audiowav[:, 0]
        data_dict['spectrogram'] = compute_mel_spec_audiomae(audio, hop_length=self.dataconfig.spec_hop_length, 
                                                             window_length=self.dataconfig.spec_window_length, 
                                                             sample_rate=self.sample_rate)

        data_dict['text'] = tf.convert_to_tensor(['description']) # dummy text
        data_dict['synthetic_text'] = tf.reshape(tf.convert_to_tensor(()), (0, 1))

        d_ = _dataset_process_map(data_dict, [0, 1], self.CACOdataConfig)
        d = {}
        for d_item in d_:
            d[d_item] = tf.expand_dims(d_[d_item], axis=0)
        d = _tokenize_and_numpy(d, self.CACOdataConfig, self.tokenizer)
        
        batch = get_train_input(d)
        audio_embedding = self.a_apply(batch, self.caco_params) # tuple: (audio embedding, hidden state)

        if embedding_type == 'event':
            audio_embedding_avg = tf.nn.avg_pool(audio_embedding[-1][0], ksize=8, 
                                                 strides=8, padding='VALID')

            timestamps = np.linspace(0, self.audio_max_len*1000, audio_embedding_avg.shape[-2])
            return audio_embedding_avg, [timestamps]
        else:
            return audio_embedding[0][0]