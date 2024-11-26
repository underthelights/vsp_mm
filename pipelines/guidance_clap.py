# CLAP_Guidance.py
from laion_clap import CLAP_Module
import torchaudio
import torch

# CLAP 모듈 초기화
CLAP_MODEL = CLAP_Module()

def generate_clap_embedding(audio_path: str, model_device: str = 'cuda') -> torch.Tensor:
    """
    Generates CLAP embedding from an audio file.
    Args:
        audio_path: Path to the audio file.
        model_device: Device to load CLAP model and perform inference.

    Returns:
        audio_embedding: Torch tensor containing the audio embedding.
    """
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample to 48kHz
    if sample_rate != 48000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=48000)
        waveform = resampler(waveform)

    # Convert stereo to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Extract embedding
    audio_embedding = torch.tensor(CLAP_MODEL.get_audio_embedding_from_data(x=waveform.numpy()))
    return audio_embedding.to(model_device)
