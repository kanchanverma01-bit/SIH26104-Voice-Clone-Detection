from .inference import predict_audio


def detect_voice(audio_path):

    result = predict_audio(audio_path)

    synthetic = float(
        result["synthetic_probability"]
    )

    authentic = float(
        result["authentic_probability"]
    )

    confidence = float(
        result["model_confidence"]
    )

    return {
        "synthetic_probability": round(
            synthetic, 4
        ),

        "authentic_probability": round(
            authentic, 4
        ),

        "model_confidence": round(
            confidence, 4
        ),

        "class_0_probability": round(
            float(result.get("class_0_probability", 0.0)),
            4
        ),

        "class_1_probability": round(
            float(result.get("class_1_probability", 0.0)),
            4
        ),
    }