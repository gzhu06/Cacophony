"""
Evaluate PyTorch CACO model on Clotho audio-text retrieval task.
"""
import torch
import torchaudio
import numpy as np
import csv
from pathlib import Path
from transformers import RobertaTokenizerFast
from typing import Dict, List

# Try to import tqdm, fallback to simple progress
from tqdm import tqdm

from src.caco_torch.caco import create_caco_model


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
        # Random sample (for eval, we use first max_patches to be deterministic)
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
    import soundfile as sf
    import scipy.signal

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


def compute_retrieval_metrics(
    audio_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute retrieval metrics.

    Args:
        audio_embeddings: (num_audio, embed_dim)
        text_embeddings: (num_text, embed_dim) - for Clotho, num_text = 5 * num_audio
        k_values: list of k values for R@k

    Returns:
        Dictionary with retrieval metrics
    """
    # Normalize embeddings
    audio_embeddings = audio_embeddings / np.linalg.norm(audio_embeddings, axis=1, keepdims=True)
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    # Compute similarity matrix: (num_audio, num_text)
    similarity = audio_embeddings @ text_embeddings.T

    num_audio = audio_embeddings.shape[0]
    num_captions_per_audio = text_embeddings.shape[0] // num_audio

    metrics = {}

    # Audio-to-Text retrieval (A2T)
    # For each audio, find the correct captions (indices i*5 to i*5+4)
    a2t_recalls = {k: 0 for k in k_values}
    for i in range(num_audio):
        # Ground truth caption indices for this audio
        gt_indices = set(range(i * num_captions_per_audio, (i + 1) * num_captions_per_audio))
        # Get top-k retrieved indices
        sorted_indices = np.argsort(-similarity[i])
        for k in k_values:
            top_k = set(sorted_indices[:k].tolist())
            if len(gt_indices & top_k) > 0:
                a2t_recalls[k] += 1

    for k in k_values:
        metrics[f'A2T_R@{k}'] = a2t_recalls[k] / num_audio * 100

    # Text-to-Audio retrieval (T2A)
    # For each caption, find the correct audio
    similarity_t2a = similarity.T  # (num_text, num_audio)
    t2a_recalls = {k: 0 for k in k_values}
    for i in range(text_embeddings.shape[0]):
        # Ground truth audio index for this caption
        gt_audio_idx = i // num_captions_per_audio
        # Get top-k retrieved indices
        sorted_indices = np.argsort(-similarity_t2a[i])
        for k in k_values:
            if gt_audio_idx in sorted_indices[:k]:
                t2a_recalls[k] += 1

    for k in k_values:
        metrics[f'T2A_R@{k}'] = t2a_recalls[k] / text_embeddings.shape[0] * 100

    return metrics


def evaluate_clotho(
    model_path: str,
    audio_dir: str,
    captions_csv: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size: int = 32,
    max_patches: int = 2400,  # For Clotho 30s audio: (16000 * 30 * 8 // 160 // 16)
    max_text_len: int = 100   # Match JAX config
) -> Dict[str, float]:
    """
    Evaluate CACO model on Clotho dataset.

    Args:
        model_path: Path to PyTorch checkpoint
        audio_dir: Directory containing audio files
        captions_csv: Path to captions CSV file
        device: Device to run on
        batch_size: Batch size for inference
        max_patches: Maximum number of patches per audio (2400 for 30s Clotho)
        max_text_len: Maximum text length

    Returns:
        Dictionary with retrieval metrics
    """
    print(f"Loading model from {model_path}...")
    model = create_caco_model(use_decoder=False)
    state_dict = torch.load(model_path, map_location=device)
    # Filter out decoder weights if present
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('decoder_module')}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print("Loading tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    print(f"Loading dataset from {captions_csv}...")
    # Load CSV using standard library
    with open(captions_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    audio_dir = Path(audio_dir)

    # Collect all audio embeddings
    print("Extracting audio embeddings...")
    audio_embeddings = []
    valid_rows = []

    for row in tqdm(rows, desc="Processing audio"):
        audio_path = audio_dir / row['file_name']
        if not audio_path.exists():
            print(f"Warning: {audio_path} not found, skipping...")
            continue

        valid_rows.append(row)

        # Load and process audio
        audio = load_audio(str(audio_path))
        mel_spec = compute_mel_spectrogram(audio)
        patches = spectrogram_to_patches(mel_spec, max_patches=max_patches)

        # Convert to tensors and add batch dimension
        audio_patches = torch.from_numpy(patches['audio_patches']).unsqueeze(0).to(device)
        audio_time_inds = torch.from_numpy(patches['audio_time_inds']).unsqueeze(0).to(device)
        audio_freq_inds = torch.from_numpy(patches['audio_freq_inds']).unsqueeze(0).to(device)
        audio_mask = torch.from_numpy(patches['audio_mask']).unsqueeze(0).to(device)

        with torch.no_grad():
            audio_emb = model.get_audio_embedding(
                audio_patches=audio_patches,
                audio_time_inds=audio_time_inds,
                audio_freq_inds=audio_freq_inds,
                audio_mask=audio_mask,
                deterministic=True,
                return_hidden_state=False,
                normalize=True  # Match JAX: normalize embeddings
            )

        audio_embeddings.append(audio_emb.cpu().numpy())

    rows = valid_rows  # Use only valid rows from now on

    audio_embeddings = np.concatenate(audio_embeddings, axis=0)
    print(f"Audio embeddings shape: {audio_embeddings.shape}")

    # Collect all text embeddings (5 captions per audio)
    print("Extracting text embeddings...")
    text_embeddings = []

    caption_columns = ['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']

    all_captions = []
    for row in rows:
        for cap_col in caption_columns:
            all_captions.append(row[cap_col])

    # Process text in batches
    for i in tqdm(range(0, len(all_captions), batch_size), desc="Processing text"):
        batch_captions = all_captions[i:i+batch_size]

        # Tokenize
        tokens = tokenizer(
            batch_captions,
            padding='max_length',
            truncation=True,
            max_length=max_text_len,
            return_tensors='pt'
        )

        text_input_ids = tokens['input_ids'].to(device)
        text_mask = tokens['attention_mask'].to(device)

        with torch.no_grad():
            text_emb = model.get_text_embedding(
                text_input_ids=text_input_ids,
                text_mask=text_mask,
                deterministic=True,
                return_hidden_state=False,
                normalize=True  # Match JAX: normalize embeddings
            )

        text_embeddings.append(text_emb.cpu().numpy())

    text_embeddings = np.concatenate(text_embeddings, axis=0)
    print(f"Text embeddings shape: {text_embeddings.shape}")

    # Compute retrieval metrics
    print("Computing retrieval metrics...")
    metrics = compute_retrieval_metrics(audio_embeddings, text_embeddings)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate CACO on Clotho")
    parser.add_argument("--model", type=str, required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--audio-dir", type=str, required=True, help="Path to audio directory")
    parser.add_argument("--captions", type=str, required=True, help="Path to captions CSV")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    metrics = evaluate_clotho(
        model_path=args.model,
        audio_dir=args.audio_dir,
        captions_csv=args.captions,
        device=args.device,
        batch_size=args.batch_size
    )

    print("\n=== Retrieval Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}%")
