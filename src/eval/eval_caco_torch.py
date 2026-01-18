"""
Evaluate PyTorch CACO model on audio-text retrieval and classification tasks.
"""
import torch
import torchaudio
import numpy as np
import csv
import argparse
import os
from pathlib import Path
from transformers import RobertaTokenizerFast
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from tqdm import tqdm
import soundfile as sf
import scipy.signal

from src.caco_torch.caco import create_caco_model, CACO

from .retrieval_utils import compute_retrieval_metric
from .dataset_processors import (
    ESC50Processor,
    US8KProcessor,
    VGGSoundProcessor,
    TUTAS2017Processor,
    Clotho16kProcessor,
    AudioCaps16kProcessor,
)


@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""
    batch_size: int = 1
    patches_seq_len: int = 512
    time_patch_size: int = 16
    freq_patch_size: int = 16
    max_text_len: int = 100
    synthetic_prob: float = 0.8


def compute_mel_spectrogram(
    audio: torch.Tensor,
    sr: int = 16000,
    hop_length: int = 160,
    win_length: int = 400,
    n_fft: int = 512,
    n_mels: int = 128,
    scale: float = 0.2,
    bias: float = 0.9
) -> np.ndarray:
    """Compute mel spectrogram using torchaudio, matching TF/JAX implementation."""
    # Use STFT with center=False to match TF behavior
    stft_result = torch.stft(
        audio.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        return_complex=True,
        center=False,
    )
    # Magnitude spectrogram (power=1.0 to match TF)
    spec = torch.abs(stft_result).T  # (time, freq)

    # Mel filterbank (no normalization to match TF)
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=0,
        f_max=sr / 2,
        n_mels=n_mels,
        sample_rate=sr,
        norm=None,
    )

    mel_spec = spec @ mel_fb  # (time, n_mels)
    mel_spec = torch.log(mel_spec + 1e-5) * scale + bias
    return mel_spec.numpy()


def spectrogram_to_patches(
    spectrogram: np.ndarray,
    time_patch_size: int = 16,
    freq_patch_size: int = 16,
    max_patches: int = 512
) -> Dict[str, np.ndarray]:
    """Convert spectrogram to patches for the model."""
    # Truncate to fit patch size
    num_time_frames = spectrogram.shape[0] // time_patch_size * time_patch_size
    spectrogram = spectrogram[:num_time_frames]

    num_time_patches = num_time_frames // time_patch_size
    num_freq_patches = spectrogram.shape[1] // freq_patch_size
    full_patch_size = num_time_patches * num_freq_patches

    # Reshape to patches
    x = spectrogram.reshape(num_time_patches, time_patch_size,
                            num_freq_patches, freq_patch_size)
    x = x.transpose(0, 2, 1, 3)  # (time_patches, freq_patches, time_size, freq_size)
    x = x.reshape(num_time_patches, num_freq_patches,
                  time_patch_size * freq_patch_size)
    x = x.reshape(-1, time_patch_size * freq_patch_size)  # (num_patches, patch_size)

    # Handle sequence length
    if full_patch_size > max_patches:
        # For eval, we use first max_patches to be deterministic
        keep_inds = np.arange(max_patches)
        x = x[keep_inds]
        mask = np.ones(max_patches, dtype=np.float32)
        time_inds = keep_inds // num_freq_patches
        freq_inds = keep_inds % num_freq_patches
    else:
        mask = (np.arange(max_patches) < full_patch_size).astype(np.float32)
        time_inds = (mask * np.arange(max_patches)).astype(np.int64) // num_freq_patches
        freq_inds = (mask * np.arange(max_patches)).astype(np.int64) % num_freq_patches
        # Pad
        x = np.pad(x, [[0, max_patches - full_patch_size], [0, 0]], mode='constant')

    return {
        'audio_patches': x.astype(np.float32),
        'audio_time_inds': time_inds.astype(np.float32),
        'audio_freq_inds': freq_inds.astype(np.float32),
        'audio_mask': mask.astype(np.float32)
    }


def load_audio(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load audio file and resample to target sample rate, matching JAX implementation."""
    # Use soundfile to match JAX implementation
    audio, sr = sf.read(audio_path)
    audio = audio.astype(np.float32)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=-1)

    # Resample using scipy to match JAX
    if sr != target_sr:
        new_num_samples = round(len(audio) * float(target_sr) / sr)
        audio = scipy.signal.resample(audio, new_num_samples).astype(np.float32)

    return torch.from_numpy(audio).unsqueeze(0)


def load_caco_torch(ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    """Load PyTorch CACO model from checkpoint."""
    # Create model
    model = create_caco_model()

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    return {
        'model': model,
        'tokenizer': tokenizer,
        'device': device
    }


def prepare_audio_batch(
    audio: torch.Tensor,
    datasetconfig: DatasetConfig,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Prepare audio batch for model inference."""
    # Compute mel spectrogram
    mel_spec = compute_mel_spectrogram(audio)

    # Convert to patches
    patches = spectrogram_to_patches(
        mel_spec,
        time_patch_size=datasetconfig.time_patch_size,
        freq_patch_size=datasetconfig.freq_patch_size,
        max_patches=datasetconfig.patches_seq_len
    )

    # Convert to tensors and add batch dimension
    batch = {
        'audio_patches': torch.from_numpy(patches['audio_patches']).unsqueeze(0).to(device),
        'audio_time_inds': torch.from_numpy(patches['audio_time_inds']).unsqueeze(0).to(device),
        'audio_freq_inds': torch.from_numpy(patches['audio_freq_inds']).unsqueeze(0).to(device),
        'audio_mask': torch.from_numpy(patches['audio_mask']).unsqueeze(0).to(device),
    }

    return batch


def prepare_text_batch(
    text: str,
    tokenizer: RobertaTokenizerFast,
    max_text_len: int,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Prepare text batch for model inference."""
    tokenized = tokenizer(
        [text],
        padding='max_length',
        truncation=True,
        max_length=max_text_len,
        return_tensors='pt'
    )

    return {
        'text_input_ids': tokenized['input_ids'].to(device),
        'text_mask': tokenized['attention_mask'].to(device),
    }


@torch.no_grad()
def compute_audio_embedding(
    model: CACO,
    audio_batch: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Compute audio embedding from batch."""
    embedding = model.get_audio_embedding(
        audio_patches=audio_batch['audio_patches'],
        audio_time_inds=audio_batch['audio_time_inds'],
        audio_freq_inds=audio_batch['audio_freq_inds'],
        audio_mask=audio_batch['audio_mask'],
        deterministic=True,
        return_hidden_state=False,
        normalize=True,
    )
    return embedding


@torch.no_grad()
def compute_text_embedding(
    model: CACO,
    text_batch: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Compute text embedding from batch."""
    embedding = model.get_text_embedding(
        text_input_ids=text_batch['text_input_ids'],
        text_mask=text_batch['text_mask'],
        deterministic=True,
        return_hidden_state=False,
        normalize=True,
    )
    return embedding


@torch.no_grad()
def compute_all_class_embeddings(
    model: CACO,
    tokenizer: RobertaTokenizerFast,
    class_list: List[str],
    max_text_len: int,
    device: torch.device,
    prefix: str = ''
) -> torch.Tensor:
    """Compute embeddings for all classes."""
    all_embeddings = []

    for class_text in tqdm(class_list, desc="Computing class embeddings"):
        text_batch = prepare_text_batch(
            prefix + class_text,
            tokenizer,
            max_text_len,
            device
        )
        embedding = compute_text_embedding(model, text_batch)
        all_embeddings.append(embedding)

    return torch.cat(all_embeddings, dim=0)


def zs_classification(
    model: CACO,
    tokenizer: RobertaTokenizerFast,
    dataprocessor,
    datasetconfig: DatasetConfig,
    device: torch.device,
    subdir_name: str = '',
    text_prefix: str = 'This is a sound of '
) -> float:
    """Zero-shot classification evaluation."""
    filepaths, descriptions, _ = dataprocessor.get_filepaths_and_descriptions(current_split=subdir_name)

    # Get unique class labels
    class_labels = [descriptions[audioid]['description'][0] for audioid in descriptions]
    class_labels = list(set(class_labels))
    class_to_index_map = {v: i for i, v in enumerate(class_labels)}

    # Compute all class embeddings
    all_text_embeddings = compute_all_class_embeddings(
        model, tokenizer, class_labels,
        datasetconfig.max_text_len, device, prefix=text_prefix
    )

    dataset_len = len(filepaths)
    ks = [1]
    total_correct = {str(k): 0 for k in ks}

    for file_idx in tqdm(range(dataset_len), desc="Evaluating"):
        audio_name = filepaths[file_idx].split('/')[-1].split('.wav')[0]
        audio_description = descriptions[audio_name]['description'][0]

        # Load and process audio
        audiowav = load_audio(filepaths[file_idx], dataprocessor.config.sampling_rate)
        audio_batch = prepare_audio_batch(audiowav, datasetconfig, device)

        # Get audio embedding
        audio_embedding = compute_audio_embedding(model, audio_batch)

        # Compute logits
        target_idx = class_to_index_map[audio_description]
        logits = torch.exp(model.logit_scale) * audio_embedding @ all_text_embeddings.T
        indices = torch.argsort(-logits, dim=-1)

        for k in ks:
            if target_idx in indices[0, :k].cpu().numpy():
                total_correct[str(k)] += 1

    for k in ks:
        print(f'top {k} accuracy: {total_correct[str(k)]/dataset_len:.4f}')

    return total_correct[str(ks[0])] / dataset_len


def audio_retrieval(
    model: CACO,
    tokenizer: RobertaTokenizerFast,
    dataprocessor,
    datasetconfig: DatasetConfig,
    device: torch.device,
    eval_split: str = 'test'
):
    """Audio-text retrieval evaluation."""
    filepaths, descriptions, _ = dataprocessor.get_filepaths_and_descriptions(current_split=eval_split)

    dataset_len = len(filepaths)

    all_text = []
    all_text_embeddings = []
    all_audio = []
    all_audio_embeddings = []
    gt_audio_text = {}
    gt_text_audio = {}

    for file_idx in tqdm(range(dataset_len), desc="Processing audio files"):
        audio_name = filepaths[file_idx].split('/')[-1].split('.wav')[0]
        gt_audio_text[audio_name] = []

        # Load audio once per file
        audiowav = load_audio(filepaths[file_idx], dataprocessor.config.sampling_rate)
        audio_batch = prepare_audio_batch(audiowav, datasetconfig, device)

        # Get text embeddings for all captions
        audio_descriptions = descriptions[audio_name]['description']
        for audio_description in audio_descriptions:
            text_batch = prepare_text_batch(
                audio_description,
                tokenizer,
                datasetconfig.max_text_len,
                device
            )
            text_embedding = compute_text_embedding(model, text_batch)

            gt_audio_text[audio_name].append(audio_description)
            gt_text_audio[audio_description] = audio_name
            all_text.append(audio_description)
            all_text_embeddings.append(text_embedding)

        # Get audio embedding
        audio_embedding = compute_audio_embedding(model, audio_batch)
        all_audio_embeddings.append(audio_embedding)
        all_audio.append(audio_name)

    # Concatenate embeddings
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    all_audio_embeddings = torch.cat(all_audio_embeddings, dim=0)

    # Compute similarity matrix
    logits_ar = all_text_embeddings @ all_audio_embeddings.T

    # Evaluation: audio to text
    print('audio to text retrieval:')
    at_indices = torch.argsort(-logits_ar.T, dim=-1).cpu().numpy()
    compute_retrieval_metric(at_indices, all_audio, all_text, gt_audio_text)

    # Evaluation: text to audio
    print('text to audio retrieval:')
    ta_indices = torch.argsort(-logits_ar, dim=-1).cpu().numpy()
    compute_retrieval_metric(ta_indices, all_text, all_audio, gt_text_audio, 'ta')


@torch.no_grad()
def decode_caption(
    model: CACO,
    tokenizer: RobertaTokenizerFast,
    audio_batch: Dict[str, torch.Tensor],
    max_decode_length: int = 100,
    temperature: float = 0.1
) -> str:
    """Decode caption from audio using the decoder."""
    if model.decoder_module is None:
        raise ValueError("Model does not have a decoder module. Load with use_decoder=True.")

    device = audio_batch['audio_patches'].device
    batch_size = audio_batch['audio_patches'].shape[0]

    # Get audio hidden states for cross-attention
    _, audio_hidden = model.get_audio_embedding(
        audio_patches=audio_batch['audio_patches'],
        audio_time_inds=audio_batch['audio_time_inds'],
        audio_freq_inds=audio_batch['audio_freq_inds'],
        audio_mask=audio_batch['audio_mask'],
        deterministic=True,
        return_hidden_state=True,
        normalize=False,
    )

    # Initialize with BOS token
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    # Start with BOS token
    generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)

    for _ in range(max_decode_length):
        # Create attention mask
        text_mask = torch.ones(generated.shape, dtype=torch.float32, device=device)

        # Forward through decoder
        logits = model.decoder_module(
            text_input_ids=generated,
            text_mask=text_mask,
            audio_hidden=audio_hidden,
            audio_mask=audio_batch['audio_mask'],
            deterministic=True,
        )

        # Get next token (greedy or with temperature)
        next_token_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        generated = torch.cat([generated, next_token], dim=1)

        # Check for EOS
        if (next_token == eos_id).all():
            break

    # Decode tokens to text
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return decoded[0].strip()


def audio_captioning(
    model: CACO,
    tokenizer: RobertaTokenizerFast,
    dataprocessor,
    datasetconfig: DatasetConfig,
    device: torch.device,
    eval_split: str = 'test',
    output_dir: str = './'
):
    """Audio captioning evaluation."""
    filepaths, descriptions, _ = dataprocessor.get_filepaths_and_descriptions(current_split=eval_split)

    dataset_len = len(filepaths)

    audio_filename_list = []
    predicted_caption_list = []
    gt_caption_list = []

    for file_idx in tqdm(range(dataset_len), desc="Generating captions"):
        audio_name = filepaths[file_idx].split('/')[-1].split('.wav')[0]
        audio_filename_list.append(audio_name)

        # Get ground truth captions
        audio_descriptions = descriptions[audio_name]['description']

        # Load and process audio
        audiowav = load_audio(filepaths[file_idx], dataprocessor.config.sampling_rate)
        audio_batch = prepare_audio_batch(audiowav, datasetconfig, device)

        # Generate caption
        predicted_caption = decode_caption(
            model, tokenizer, audio_batch,
            max_decode_length=100,
            temperature=0.1
        )
        predicted_caption_list.append(predicted_caption)

        # Store ground truth
        gt_caption_item = [desc.replace(',', '') for desc in audio_descriptions]
        gt_caption_list.append(gt_caption_item)

    assert len(predicted_caption_list) == len(gt_caption_list)

    # Write predictions to CSV
    pred_path = os.path.join(output_dir, 'predictions.csv')
    gt_path = os.path.join(output_dir, 'gt.csv')

    with open(pred_path, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['file_name', 'caption_predicted'])
        for i, audioname in enumerate(audio_filename_list):
            writer.writerow([audioname, predicted_caption_list[i]])

    with open(gt_path, 'w', newline='') as fg:
        writer = csv.writer(fg)
        writer.writerow(['file_name', 'caption_reference_01', 'caption_reference_02',
                        'caption_reference_03', 'caption_reference_04', 'caption_reference_05'])
        for i, audioname in enumerate(audio_filename_list):
            row = [audioname] + gt_caption_list[i]
            # Pad if less than 5 captions
            while len(row) < 6:
                row.append('')
            writer.writerow(row)

    print(f"Predictions saved to {pred_path}")
    print(f"Ground truth saved to {gt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='./ckpt.pt', help='model checkpoint path')
    parser.add_argument('--task', type=str, default='zs', choices=['zs', 'ar', 'caption'],
                        help='evaluation task: zs (zero-shot classification), ar (audio retrieval), caption (audio captioning)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device to run on')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.ckpt_path}...")
    model_dict = load_caco_torch(args.ckpt_path, device)
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']

    if args.task == 'zs':
        # Eval 1: Zero-shot classification
        #######################################
        # In classification task:
        # 1) compute all text embeddings
        # 2) rank the top text embeddings on the given audio embedding
        #######################################

        eval_data_processors = ['ESC50Processor']
        CommondataConfig = DatasetConfig(
            batch_size=1,
            patches_seq_len=(100 * 10 * 8 // 16),  # 10 seconds
            time_patch_size=16,
            freq_patch_size=16,
            max_text_len=100,
            synthetic_prob=0.8
        )

        total_acc = {}
        for data_processor_name in tqdm(eval_data_processors[-1:]):
            print(f'Processing: {data_processor_name} ........')
            data_processor = globals()[data_processor_name]()

            text_prefix = 'This is a sound on ' if data_processor_name == 'TUTAS2017Processor' else 'This is a sound of '

            acc1 = zs_classification(
                model, tokenizer, data_processor, CommondataConfig,
                device, text_prefix=text_prefix
            )
            total_acc[data_processor_name] = acc1

        print("\nFinal Results:")
        for name, acc in total_acc.items():
            print(f"  {name}: {acc:.4f}")

    elif args.task == 'ar':
        # Eval 2: (ZS) text to audio retrieval
        #######################################
        # In retrieval task:
        # 1) compute all text embeddings
        # 2) compute all audio embeddings
        # 3a) in text to audio: rank top audio embeddings on given text embedding
        # 3b) in audio to text: rank top text embeddings on given audio embedding
        #######################################

        audio_seg_time = 30
        total_samples = 16000 * audio_seg_time
        max_patches = (total_samples * 8 // 160 // 16)
        CommondataConfig = DatasetConfig(
            batch_size=1,
            patches_seq_len=max_patches,
            time_patch_size=16,
            freq_patch_size=16,
            max_text_len=100,
            synthetic_prob=0.8
        )

        clothov2processor = Clotho16kProcessor()
        audio_retrieval(model, tokenizer, clothov2processor, CommondataConfig, device, 'evaluation')

    elif args.task == 'caption':
        # Eval 3: Audio captioning
        #######################################

        audio_seg_time = 30
        total_samples = 16000 * audio_seg_time
        max_patches = (total_samples * 8 // 160 // 16)
        CommondataConfig = DatasetConfig(
            batch_size=1,
            patches_seq_len=max_patches,
            time_patch_size=16,
            freq_patch_size=16,
            max_text_len=100,
            synthetic_prob=0.8
        )

        output_dir = os.path.dirname(args.ckpt_path) if args.ckpt_path else './'
        clothov2processor = Clotho16kProcessor()
        audio_captioning(
            model, tokenizer, clothov2processor, CommondataConfig,
            device, 'evaluation', output_dir
        )
