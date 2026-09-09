import torch

from .inference import load_model
from .preprocessing import load_audio, prepare_audio


_model = None


def detect_voice(audio_path):
    global _model

    if _model is None:
        _model = load_model()

    audio = load_audio(audio_path)
    audio_tensor = prepare_audio(audio)

    audio_tensor = audio_tensor.to(
        next(_model.parameters()).device
    )

    with torch.no_grad():
        output = _model(audio_tensor)

    return output
