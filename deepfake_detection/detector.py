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
        _, output = _model(audio_tensor)

        probabilities = torch.softmax(output, dim=1)

        synthetic_probability = float(
            probabilities[0, 0].item()
        )

        authentic_probability = float(
            probabilities[0, 1].item()
        )

        model_confidence = max(
            synthetic_probability,
            authentic_probability
        )

    return {
        "synthetic_probability": synthetic_probability,
        "authentic_probability": authentic_probability,
        "model_confidence": model_confidence
    }
