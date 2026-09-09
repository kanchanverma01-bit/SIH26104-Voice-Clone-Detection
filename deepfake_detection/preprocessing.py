import numpy as np
import soundfile as sf
import torch
import torchaudio

from .config import SAMPLE_RATE, TARGET_SAMPLES


def load_audio(audio_path):
    audio, sample_rate = sf.read(
        str(audio_path),
        dtype="float32"
    )

    # Stereo → Mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample → 16 kHz
    if sample_rate != SAMPLE_RATE:
        audio_tensor = torch.from_numpy(audio)

        audio_tensor = torchaudio.functional.resample(
            audio_tensor,
            sample_rate,
            SAMPLE_RATE
        )

        audio = audio_tensor.numpy()

    return audio.astype(np.float32)


def prepare_audio(audio):
    if len(audio) == 0:
        raise ValueError("Audio is empty.")

    # Repeat short audio
    if len(audio) < TARGET_SAMPLES:
        repeats = TARGET_SAMPLES // len(audio) + 1
        audio = np.tile(audio, repeats)

    # Keep exactly 64600 samples
    audio = audio[:TARGET_SAMPLES]

    tensor = torch.from_numpy(audio)

    return tensor.unsqueeze(0)
