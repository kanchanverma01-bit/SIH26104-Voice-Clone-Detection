from .inference import predict_audio


def detect_voice(audio_path):
    """
    Detect whether an audio file is real or AI-generated.
    """

    result = predict_audio(audio_path)

    return {
        "synthetic_probability": round(
            result["synthetic_probability"], 4
        ),
        "authentic_probability": round(
            result["authentic_probability"], 4
        ),
        "model_confidence": round(
            result["model_confidence"], 4
        )
    }
