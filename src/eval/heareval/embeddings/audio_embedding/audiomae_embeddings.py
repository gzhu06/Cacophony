#!/usr/bin/env python3

import random

import tensorflow as tf

import numpy as np
import jax
import jax.numpy as jnp
import flax
from einops import rearrange
from src.caco.load_model import load_audiomae
from src.caco.caco_eval_utils import compute_mel_spec_audiomae
from src.caco.dataset import AudioMAEDatasetConfig, Batch

PyTreeDef = type(jax.tree_util.tree_structure(None))
def get_train_input(
    batch: Batch
) -> PyTreeDef:
    batch = dict(
        audio_patches=batch['audio_patches'],
        audio_time_inds=batch['audio_time_inds'],
        audio_freq_inds=batch['audio_freq_inds'],
        audio_mask=batch['audio_mask'],
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
        sample_rate: int = 16000, 
        batch_size: int = 1,
        audio_max_len: int = 160000,
        ):
        CommondataConfig = AudioMAEDatasetConfig(audio_segment_len=audio_max_len)
        dataconfig=CommondataConfig
        self.sample_rate = sample_rate

        # load audiomae globally for test
        audiomae_model_dict = load_audiomae(model_path)
        self.audiomae_params = flax.jax_utils.replicate(audiomae_model_dict['audiomae_params'], 
                                                        devices=jax.local_devices())
        self.audiomae_model = audiomae_model_dict['audiomae_model']
        self.dataconfig = dataconfig
        self.batch_size = batch_size
        self.audio_max_len = audio_max_len
        
        # maximum usable patches
        self.max_patches = int(dataconfig.audio_segment_len *sample_rate // dataconfig.spec_hop_length // dataconfig.time_patch_size) * (dataconfig.spec_num_mels // dataconfig.freq_patch_size)
        
        def compute_audio_embedding(audio_batch, model_params):
            return self.audiomae_model.apply(
                {'params': model_params},
                x=audio_batch['audio_patches'],
                time_inds=audio_batch['audio_time_inds'],
                freq_inds=audio_batch['audio_freq_inds'],
                mask=audio_batch['audio_mask'],
                method=self.audiomae_model.__call__,
            )
        
        self.a_apply = jax.pmap(compute_audio_embedding, axis_name='dp')
    
    def get_embedding_from_wav(self, audiowav):
        
        audio = audiowav[:, 0]

        # spectrogram feature coputing
        spectrogram = compute_mel_spec_audiomae(audio, hop_length=self.dataconfig.spec_hop_length,
                                                window_length=self.dataconfig.spec_window_length,
                                                num_mels=self.dataconfig.spec_num_mels,
                                                scale=self.dataconfig.spec_scale,
                                                bias=self.dataconfig.spec_bias)

        # remove residual patches
        spectrogram = spectrogram[:int(tf.shape(spectrogram)[0]//self.dataconfig.time_patch_size*self.dataconfig.time_patch_size)]
        
        # actual used patches
        num_time_patches, num_freq_patches = tf.shape(spectrogram)[0]//self.dataconfig.time_patch_size, tf.shape(spectrogram)[1]//self.dataconfig.freq_patch_size
        total_patches = num_time_patches * num_freq_patches
        
        x = tf.reshape(spectrogram, [num_time_patches, self.dataconfig.time_patch_size, 
                                     num_freq_patches, self.dataconfig.freq_patch_size])
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [num_time_patches, num_freq_patches, 
                           self.dataconfig.time_patch_size*self.dataconfig.freq_patch_size])
        x = rearrange(x, 't1 f1 h -> (t1 f1) h')
        
        # random sample if sequence is longer
        if total_patches > self.max_patches:
            keep_inds = list(range(total_patches))
            random.shuffle(keep_inds)
            keep_inds = keep_inds[:self.max_patches]
            keep_inds = tf.sort(keep_inds)

            x = tf.gather(x, indices=keep_inds)
            audio_mask = tf.ones(self.max_patches, dtype=tf.int32)
            time_inds = keep_inds // num_freq_patches
            freq_inds = keep_inds % num_freq_patches
        else:
            audio_mask = tf.cast(tf.range(self.max_patches) < total_patches, tf.int32)
            time_inds = (audio_mask * tf.range(self.max_patches)) // num_freq_patches
            freq_inds = (audio_mask * tf.range(self.max_patches)) % num_freq_patches
            x = tf.pad(x, [[0, self.max_patches - total_patches], [0, 0]], 
                       mode='CONSTANT', constant_values = 0)

#         if tf.shape(audio)[0] > self.dataconfig.audio_segment_len:
#             audio_start_ind = random.randint(0, tf.shape(audio)[0]-self.dataconfig.audio_segment_len+1)
#             audio = audio[audio_start_ind:(audio_start_ind + self.dataconfig.audio_segment_len)]
#         original_time_dim = tf.shape(spectrogram)[0]

        return x, time_inds, freq_inds, audio_mask

    def get_embedding_as_numpy(self, audiofiles, embedding_type=None) -> np.ndarray:

        assert len(audiofiles) == 1, 'batch size must be 1'
        
        audio_tensor_list = []
        time_inds_list = []
        freq_inds_list = []
        audiomask_list = []
        for audiofile in audiofiles:
            audio, _ = tf.audio.decode_wav(tf.io.read_file(audiofile))
            
            x, time_inds, freq_inds, audio_mask = self.get_embedding_from_wav(audio)
            audio_tensor_list.append(x)
            time_inds_list.append(time_inds)
            freq_inds_list.append(freq_inds)
            audiomask_list.append(audio_mask)
            
        audio_tensors = tf.stack(audio_tensor_list, axis=0)
        freq_inds_tensors = tf.stack(freq_inds_list, axis=0)
        time_inds_tensors = tf.stack(time_inds_list, axis=0)
        audiomask_tensors = tf.stack(audiomask_list, axis=0)
        
        data_dict =  {'audio_patches': audio_tensors,
                      'audio_time_inds': time_inds_tensors,
                      'audio_freq_inds' : freq_inds_tensors,
                      'audio_mask':audiomask_tensors}
        
        batch = get_train_input(data_dict)
        audio_embeddings = self.a_apply(batch, self.audiomae_params)
        audio_embeddings = jnp.squeeze(audio_embeddings, axis=0)

        if embedding_type == 'event':
            audio_embedding_avg = tf.nn.avg_pool(audio_embeddings, ksize=8, 
                                                 strides=8, padding='VALID')
            timestamps = np.linspace(0, self.audio_max_len*1000, audio_embedding_avg.shape[-2])
            return audio_embedding_avg, [timestamps]
        else:
            audio_embeddings = jnp.mean(audio_embeddings, axis=1)
            return audio_embeddings
