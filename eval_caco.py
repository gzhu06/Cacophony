import jax, flax
import jax.numpy as jnp
import tensorflow as tf
from einops import rearrange
import scipy
import argparse, os
from functools import partial
from typing import Any
from tqdm import tqdm
import soundfile as sf
import numpy as np

from src.caco.load_model import load_caco
from src.caco.caco import CACO, decode
from src.caco.dataset import Batch, DatasetConfig, _dataset_process_map, _tokenize_and_numpy
from src.caco.caco_eval_utils import load_from_list

from retrieval_utils import compute_retrieval_metric
from dataset_processors import *

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, default='./ckpt.pt', help='model ckpt path')
parser.add_argument('--task', type=str, default='zs', help='evaluation task name')
args = parser.parse_args()

# load blap globally for test
ckpt_path = args.ckpt_path
caco_model_dict = load_caco(ckpt_path, use_decoder=True)
caco_model = caco_model_dict['caco_model']

def compute_audio_embedding(audio_batch, model_params):
    return caco_model.apply(
        {'params': model_params},
        audio_patches=audio_batch['audio_patches'],
        audio_time_inds=audio_batch['audio_time_inds'],
        audio_freq_inds=audio_batch['audio_freq_inds'],
        audio_mask=audio_batch['audio_mask'],
        deterministic=True,
        return_hidden_state=False,
        normalize=True,
        method=caco_model.get_audio_embedding,
    )

def compute_text_embedding(text_batch, model_params):
    return caco_model.apply(
        {'params': model_params},
        text_input_ids=text_batch['text_input_ids'], 
        text_mask=text_batch['text_mask'],
        deterministic=True,
        return_hidden_state=False,
        normalize=True,
        method=caco_model.get_text_embedding,
    )

a_apply = jax.pmap(compute_audio_embedding, axis_name='dp')
t_apply = jax.pmap(compute_text_embedding, axis_name='dp')
caco_params = flax.jax_utils.replicate(caco_model_dict['caco_params'], devices=jax.local_devices())
tokenizer = caco_model_dict['tokenizer']

PyTreeDef = type(jax.tree_util.tree_structure(None))

def _tree_map_batch_devices(x: PyTreeDef) -> PyTreeDef:
    return jax.tree_util.tree_map(
        lambda x: rearrange(jnp.asarray(x), '(d b) ... -> d b ...', d=jax.local_device_count()),
        x
    )

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

@tf.function
def load_audio(audio_path, dataset_sampling_rate):
    audiowav, _ = sf.read(audio_path)
    audiowav = audiowav.astype(np.float32)
    if len(audiowav.shape) > 1:
        audiowav = np.mean(audiowav, axis=-1)

    if dataset_sampling_rate != 16000:
        new_num_samples = round(audiowav.shape[-1]*float(16000)/dataset_sampling_rate)
        audiowav = scipy.signal.resample(audiowav, new_num_samples)

    return audiowav

def prepare_audio_batch(audiowav, audio_description, datasetconfig):

    data_dict = load_from_list(audiowav, audio_description)
    d_ = _dataset_process_map(data_dict, [0, 1], datasetconfig)
    d = {}
    for d_item in d_:
        d[d_item] = tf.expand_dims(d_[d_item], axis=0)
    d = _tokenize_and_numpy(d, datasetconfig, tokenizer)
    batch = get_train_input(d)

    return batch, data_dict

@partial(jax.pmap, axis_name='dp', static_broadcasted_argnums=(0, 1, 2, 3))
def _decode_helper(
    model: CACO,
    tokenizer: Any,
    max_decode_length: int,
    temperature: float,
    audio_batch: PyTreeDef,
    params: PyTreeDef,
    rng: jax.random.KeyArray
) -> jnp.ndarray:

    decoded_tokens = decode(
        model,
        params=params,
        audio_batch=audio_batch,
        max_length=max_decode_length,
        temperature=temperature,
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id,
        rng=rng
    )
    return decoded_tokens

def compute_all_class_embedding(class_list, max_text_len=77, prefix=''):

    all_text_embeddings = []
    for class_text in tqdm(class_list):
        tokenized = tokenizer([prefix + class_text], 
                              padding='max_length', 
                              truncation=True,
                              max_length=max_text_len, 
                              return_tensors='np')
        text_input_ids, text_mask = tokenized['input_ids'], tokenized['attention_mask']
        text_batch = dict(text_input_ids=text_input_ids,
                          text_mask=text_mask)
        text_batch = jax.tree_util.tree_map(
            lambda x: rearrange(jnp.asarray(x), '(d b) ... -> d b ...', d=jax.local_device_count()),
            text_batch
        )
        text_embedding = t_apply(text_batch, caco_params)
        all_text_embeddings.append(text_embedding)
    all_text_embeddings = jnp.concatenate(all_text_embeddings, axis=0)
    all_text_embeddings = jnp.squeeze(all_text_embeddings, axis=1)

    return all_text_embeddings

def zs_classification(dataprocessor, datasetconfig, subdir_name='', text_prefix='This is a sound of '):

    filepaths, descriptions, _ = dataprocessor.get_filepaths_and_descriptions(current_split=subdir_name)
    class_labels = [descriptions[audioid]['description'][0] for audioid in descriptions]
    class_labels = list(set(class_labels))

    class_to_index_map = {v: i for i, v in enumerate(class_labels)}
    all_text_embeddings = compute_all_class_embedding(class_labels, prefix=text_prefix)

    dataset_len = len(filepaths)

    ks = [1]
    total_correct = {str(k): 0 for k in ks}
    for file_idx in tqdm(range(dataset_len)):

        audio_name = filepaths[file_idx].split('/')[-1].split('.wav')[0]
        audio_description = descriptions[audio_name]['description'][0]

        # load audio
        audiowav = load_audio(filepaths[file_idx], dataprocessor.config.sampling_rate)
        batch, data_dict = prepare_audio_batch(audiowav, audio_description, datasetconfig)

        audio_embedding = a_apply(batch, caco_params)

        target_idx = class_to_index_map[bytes.decode(data_dict['text'][0].numpy())]
        targets = _tree_map_batch_devices(jnp.array([target_idx]))
        audio_embedding = jnp.squeeze(audio_embedding, axis=1)
        logits = jnp.exp(caco_params['logit_scale']) * audio_embedding @ all_text_embeddings.T
        indices = jnp.argsort(-logits, axis=-1)
        
        for k in ks:
            n_correct = jnp.sum(jnp.any(targets[..., None] == indices[:, :k], axis=-1))
            total_correct[str(k)] += n_correct

    for k in ks:
        print('top '+str(k)+' accuracy:', total_correct[str(k)]/dataset_len)

    return total_correct[str(ks[0])]/dataset_len

def audio_retrieval(dataprocessor, datasetconfig, eval_split='test'):
    filepaths, descriptions, _ = dataprocessor.get_filepaths_and_descriptions(current_split=eval_split)

    dataset_len = len(filepaths)
    
    all_text = []
    all_text_embeddings = []
    all_audio = []
    all_audio_embeddings = []
    gt_audio_text = {}
    gt_text_audio = {}
    
    for file_idx in tqdm(range(dataset_len)):
        audio_name = filepaths[file_idx].split('/')[-1].split('.wav')[0]
        gt_audio_text[audio_name] = []

        # get text embeddings
        audio_descriptions = descriptions[audio_name]['description']
        for audio_description in audio_descriptions:
            
            audiowav = load_audio(filepaths[file_idx], dataprocessor.config.sampling_rate)
            batch, data_dict = prepare_audio_batch(audiowav, audio_description, datasetconfig)

            text_embedding = t_apply(batch, caco_params)

            # prepare for text embedding
            text_str = bytes.decode(data_dict['text'][0].numpy())
            gt_audio_text[audio_name].append(text_str) 
            gt_text_audio[text_str] = audio_name
            all_text.append(text_str)
            
            all_text_embeddings.append(text_embedding)

        # get audio embedding
        audio_embedding = a_apply(batch, caco_params)
        all_audio_embeddings.append(audio_embedding)
        all_audio.append(audio_name)
        
    all_text_embeddings = jnp.concatenate(all_text_embeddings, axis=0)
    all_audio_embeddings = jnp.concatenate(all_audio_embeddings, axis=0)
    logits_ar=jnp.squeeze(all_text_embeddings, axis=1) @ jnp.squeeze(all_audio_embeddings.T, axis=1)
    
    # evaluation: audio to text
    print('audio to text retrieval:')
    at_indices = jnp.argsort(jnp.transpose(-logits_ar), axis=-1)
    compute_retrieval_metric(at_indices, all_audio, all_text, gt_audio_text)

    # evaluation: text to audio
    print('text to audio retrieval:')
    ta_indices = jnp.argsort(-logits_ar, axis=-1)
    compute_retrieval_metric(ta_indices, all_text, all_audio, gt_text_audio, 'ta')

def audio_captioning(dataprocessor, datasetconfig, eval_split='test'):
    filepaths, descriptions, _ = dataprocessor.get_filepaths_and_descriptions(current_split=eval_split)

    dataset_len = len(filepaths)
    gt_audio_text = {}

    p_gather_tokens = lambda y: flax.jax_utils.unreplicate(jax.pmap(lambda x: jax.lax.all_gather(x, axis_name='dp'), axis_name='dp')(y))
    flatten_to_host = lambda x: jax.device_put(rearrange(x, 'd b ... -> (d b) ...'), 
                                               jax.local_devices(backend='cpu')[0])
    p_decode = _decode_helper
    
    audio_filename_list = []
    predicted_caption_list =[]
    gt_caption_list = []
    step = 0
    for file_idx in tqdm(range(dataset_len)):
        audio_name = filepaths[file_idx].split('/')[-1].split('.wav')[0]
        gt_audio_text[audio_name] = []

        audio_filename_list.append(audio_name)

        # get text embeddings
        audio_descriptions = descriptions[audio_name]['description']

        rng = jax.random.PRNGKey(42)
        step_rng = jax.random.split(jax.random.fold_in(rng, step), jax.local_device_count()) 
        
        # load audio and prepare for inference
        audiowav = load_audio(filepaths[file_idx], dataprocessor.config.sampling_rate)
        batch, _ = prepare_audio_batch(audiowav, audio_descriptions[0], datasetconfig)
        batch.pop('text_input_ids')
        batch.pop('text_mask')

        # decode and caption
        decoded_tokens = p_decode(caco_model, tokenizer, 100, 0.1, batch, caco_params, step_rng)
        data_mask = np.array([True]*1)
        data_mask = _tree_map_batch_devices(data_mask)

        data_mask = p_gather_tokens(data_mask)
        data_mask = flatten_to_host(data_mask)

        decoded_tokens = p_gather_tokens(decoded_tokens)
        decoded_tokens = flatten_to_host(decoded_tokens)

        all_decoded_tokens = jnp.concatenate([decoded_tokens[data_mask]], axis=0)
        text_preds = tokenizer.batch_decode(all_decoded_tokens, skip_special_tokens=True)
        step += 1

        predicted_caption_list.append(text_preds[0].strip())

        gt_caption_item = []
        for audio_description in audio_descriptions:
            gt_caption_item.append(audio_description.replace(',', ''))

        gt_caption_list.append(gt_caption_item)

    assert len(predicted_caption_list) == len(gt_caption_list)

    # write predictions into csv file with heading 
    with open(os.path.join(os.path.split(args.ckpt_path)[0], 'predictions.csv'), 'w') as fp:
        with open(os.path.join(os.path.split(args.ckpt_path)[0], 'gt.csv'), 'w') as fg:
            for i, audioname in enumerate(audio_filename_list):
                if i == 0:
                    fp.write('file_name,caption_predicted\n')
                    fg.write('file_name,caption_reference_01,caption_reference_02,caption_reference_03,caption_reference_04,caption_reference_05\n')
                else:
                    fp.write(str(i) + ',' + predicted_caption_list[i] + '\n')
                    gt_caps = ','.join(gt_caption_list[i])

                    fg.write(str(i) + ',' + gt_caps + '\n')
                
        
if __name__ == "__main__":
    
    if args.task == 'zs':
        # eval 1: ZS classification on VGGSound
        #######################################
        # In classification task: 
        # 1) compute all text embedding
        # 2) rank the top text embeddings on the given audio embedding
        #######################################

        eval_data_processors = ['ESC50Processor', 'TUTAS2017Processor', 'US8KProcessor', 'VGGSoundProcessor']
        CommondataConfig = DatasetConfig(batch_size=1,
                                         patches_seq_len=(100 * 10 * 8// 16) , # 10 second
                                         time_patch_size=16,
                                         freq_patch_size=16,
                                         max_text_len=100,
                                         synthetic_prob=0.8)
        
        ## dataset config definition
        total_acc = {}
        for data_processor_name in tqdm(eval_data_processors[-1:]):
            print('Processing:', data_processor_name, '........')
            data_processor = globals()[data_processor_name]()
            
            text_prefix = 'This is a sound on ' if data_processor_name == 'TUTAS2017Processor' else 'This is a sound of '

            acc1 = zs_classification(data_processor, CommondataConfig, text_prefix=text_prefix)
            total_acc[data_processor_name] = acc1
        
    elif args.task == 'ar':
    
        # eval 2: (ZS) text to audio retrieval on audiocaps test
        #######################################
        # In retrieval task: 
        # 1) compute all text embedding
        # 2) compute all audio embedding
        # 3a) in text to audio: rank the top audio embeddings on the given text embedding
        # 3b) in audio to text: rank the top text embeddings on the given audio embedding
        #######################################

        audio_seg_time = 30
        total_samples = 16000 * audio_seg_time
        max_patches = (total_samples * 8 // 160 // 16) 
        CommondataConfig = DatasetConfig(batch_size=1,
                                         patches_seq_len=max_patches,
                                         time_patch_size=16,
                                         freq_patch_size=16,
                                         max_text_len=100,
                                         synthetic_prob=0.8)

        clothov2processor = Clotho16kProcessor()
        audio_retrieval(clothov2processor, CommondataConfig, 'evaluation')

        audio_seg_time = 10
        total_samples = 16000 * audio_seg_time
        max_patches = (total_samples * 8 // 160 // 16) 
        ACdataConfig = DatasetConfig(batch_size=1,
                                     patches_seq_len=max_patches,
                                     time_patch_size=16,
                                     freq_patch_size=16,
                                     max_text_len=77,
                                     synthetic_prob=0.8)
        audiocapsprocessor = AudioCaps16kProcessor()
        audio_retrieval(audiocapsprocessor, ACdataConfig, 'test')

    elif args.task == 'caption':

        audio_seg_time = 10
        total_samples = 16000 * audio_seg_time
        max_patches = (total_samples // 160 // 16) * 8
        ACdataConfig = DatasetConfig(batch_size=1,
                                     patches_seq_len=max_patches,
                                     time_patch_size=16,
                                     freq_patch_size=16,
                                     max_text_len=100,
                                     synthetic_prob=0.8)
        audiocapsprocessor = AudioCaps16kProcessor()
        audio_captioning(audiocapsprocessor, ACdataConfig, 'test')

        audio_seg_time = 30
        total_samples = 16000 * audio_seg_time
        max_patches = (total_samples * 8 // 160 // 16) 
        CommondataConfig = DatasetConfig(batch_size=1,
                                         patches_seq_len=max_patches,
                                         time_patch_size=16,
                                         freq_patch_size=16,
                                         max_text_len=100,
                                         synthetic_prob=0.8)

        clothov2processor = Clotho16kProcessor()
        audio_captioning(clothov2processor, CommondataConfig, 'evaluation')
