import numpy as np
import tensorflow as tf
from flax import struct
tf.config.set_visible_devices([], device_type='GPU')
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Mapping
from einops import rearrange
import random

@dataclass
class DatasetLoader(ABC):

    @abstractmethod
    def get_dataset(
        self,
        process_index: int,
        num_parallel_reads: int
    ) -> tf.data.Dataset:
        pass

@struct.dataclass
class DatasetConfig:
    batch_size: int
    patches_seq_len: int 
    time_patch_size: int
    freq_patch_size: int
    max_text_len: int
    synthetic_prob: float
        
@struct.dataclass
class AudioMAEDatasetConfig:
    batch_size: int =1
    audio_segment_len: int = 160000
    time_patch_size: int = 16
    freq_patch_size: int = 16

    spec_hop_length: int = 160
    spec_window_length: int = 400
    spec_fft_size: int = 512
    spec_num_mels: int = 128
    spec_scale: float = 0.2
    spec_bias: float = 0.9

@struct.dataclass
class Batch:
    audio_patches: np.ndarray
    audio_time_inds: np.ndarray
    audio_freq_inds: np.ndarray
    audio_mask: np.ndarray
    text: np.ndarray
    text_input_ids: np.ndarray
    text_mask: np.ndarray

def _dataset_process_map(
    batch: Mapping[str, tf.Tensor], 
    seed: List[int], 
    config: DatasetConfig
) -> Mapping[str, tf.Tensor]:
    # convert a batch of spectrogram and text data

    spectrogram = batch['spectrogram']

    # remove residual patches
    spectrogram = spectrogram[:int(tf.shape(spectrogram)[0]//config.time_patch_size*config.time_patch_size)]
    num_time_patches, num_freq_patches = tf.shape(spectrogram)[0]//config.time_patch_size, tf.shape(spectrogram)[1]//config.freq_patch_size
    full_patch_size = num_time_patches * num_freq_patches
    
    x = tf.reshape(spectrogram, [num_time_patches, config.time_patch_size, 
                                 num_freq_patches, config.freq_patch_size])

    x = tf.transpose(x, perm=[0, 2, 1, 3])
    x = tf.reshape(x, [num_time_patches, num_freq_patches, 
                       config.time_patch_size*config.freq_patch_size])
    x = rearrange(x, 't1 f1 h -> (t1 f1) h')

    # random sample if sequence is longer
    if full_patch_size > config.patches_seq_len:
        keep_inds = list(range(full_patch_size))
        random.shuffle(keep_inds)
        keep_inds = keep_inds[:config.patches_seq_len]
        keep_inds = tf.sort(keep_inds)

        x = tf.gather(x, indices=keep_inds)
        audio_mask = tf.ones(config.patches_seq_len, dtype=tf.int32)
        time_inds = keep_inds // num_freq_patches
        freq_inds = keep_inds % num_freq_patches
    else:
        audio_mask = tf.cast(tf.range(config.patches_seq_len) < full_patch_size, tf.int32)
        time_inds = (audio_mask * tf.range(config.patches_seq_len)) // num_freq_patches
        freq_inds = (audio_mask * tf.range(config.patches_seq_len)) % num_freq_patches
        x = tf.pad(x, [[0, config.patches_seq_len - full_patch_size], [0, 0]], 
                   mode='CONSTANT', constant_values = 0)

    text_index = tf.random.stateless_categorical(
        tf.ones((1, tf.shape(batch['text'])[0]), dtype=tf.float32), 1, 
        seed=tf.random.experimental.stateless_fold_in(seed, 1)
    )[0, 0]

    text = batch['text'][text_index]
    
    if tf.size(batch['synthetic_text']) > 0:

        if tf.random.stateless_categorical(tf.math.log([[1-config.synthetic_prob, config.synthetic_prob]]), 1, 
                seed=tf.random.experimental.stateless_fold_in(seed, 2)) > 0:
        
            synthetic_text_index = tf.random.stateless_categorical(
                tf.ones((1, tf.shape(batch['synthetic_text'])[0]), dtype=tf.float32), 1, 
                seed=tf.random.experimental.stateless_fold_in(seed, 1)
            )[0, 0]

            text = batch['synthetic_text'][synthetic_text_index]

    d =  {
        'audio_patches': x,
        'audio_time_inds': time_inds,
        'audio_freq_inds' : freq_inds,
        'audio_mask': audio_mask,
        'text': text,
    }

    if 'data_mask' in batch.keys():
        d['data_mask'] = batch['data_mask']

    return d


def _tokenize_and_numpy(batch, config, tokenize_fn):
    text = [s.decode('utf-8') for s in batch['text']._numpy()]
    tokenize_output = tokenize_fn(text, padding='max_length', truncation=True, max_length=config.max_text_len, return_tensors='np')
    text_input_ids = tokenize_output['input_ids']
    text_mask = tokenize_output['attention_mask']
    return Batch(
        audio_patches=batch['audio_patches']._numpy(),
        audio_time_inds=batch['audio_time_inds']._numpy(),
        audio_freq_inds=batch['audio_freq_inds']._numpy(),
        audio_mask=batch['audio_mask']._numpy(),
        text=text,
        text_input_ids=text_input_ids,
        text_mask=text_mask,
    )