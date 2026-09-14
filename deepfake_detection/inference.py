import numpy as np
import soundfile as sf
import torch

from scipy.signal import resample_poly

from .AASIST import AASISTDetector


TARGET_SR = 16000
TARGET_SAMPLES = 64600


# Load model only once
_detector = AASISTDetector()


def load_audio(audio_path):

    audio, sr = sf.read(
        audio_path,
        dtype="float32"
    )

    # Stereo → Mono
    if audio.ndim > 1:
        audio = np.mean(
            audio,
            axis=1
        )

    # Resample to 16 kHz
    if sr != TARGET_SR:

        audio = resample_poly(
            audio,
            TARGET_SR,
            sr
        ).astype(np.float32)

    if len(audio) == 0:
        raise ValueError(
            "Audio file is empty."
        )

    # Remove invalid values
    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Normalize audio
    peak = np.max(
        np.abs(audio)
    )

    if peak > 0:
        audio = audio / peak

    # Pad short audio
    # Zero-pad short audio
    if len(audio) < TARGET_SAMPLES:
        audio = np.pad(
        audio,
        (0, TARGET_SAMPLES - len(audio)),
        mode="constant"
    )

    # Exactly 64600 samples
    audio = audio[:TARGET_SAMPLES]

    return torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0)


def predict_audio(audio_path):

    audio = load_audio(
        audio_path
    )

    result = _detector.predict(
        audio
    )

    return result
