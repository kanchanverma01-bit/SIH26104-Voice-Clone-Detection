import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from .aasist import AASISTDetector


TARGET_SR = 16000
TARGET_SAMPLES = 64600


def load_audio(audio_path):
    audio, sr = sf.read(audio_path, dtype="float32")

    # Stereo → Mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample → 16 kHz
    if sr != TARGET_SR:
        audio = resample_poly(
            audio,
            TARGET_SR,
            sr
        ).astype(np.float32)

    # Empty audio check
    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

    # Pad short audio
    if len(audio) < TARGET_SAMPLES:
        repeats = (TARGET_SAMPLES // len(audio)) + 1
        audio = np.tile(audio, repeats)

    # Take exactly first 64600 samples
    audio = audio[:TARGET_SAMPLES]

    return torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0)


def predict_audio(audio_path):
    audio = load_audio(audio_path)

    detector = AASISTDetector()

    result = detector.predict(audio)

    return result
