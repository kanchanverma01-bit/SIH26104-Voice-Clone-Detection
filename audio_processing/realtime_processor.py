import subprocess
import sys
import time

from deepfake_detection.inference import predict_audio


CHUNK_DURATION = 5


print("====================================")
print(" REAL-TIME VOICE ANALYSIS STARTED ")
print("====================================")
print("Press Ctrl+C to stop.\n")


for _ in range(1):

    print("Listening... Speak now!")

    # Run complete audio processing pipeline
    subprocess.run(
        [sys.executable, "-m", "audio_processing.audio_processor"],
        check=True
    )

    print("\nProcessing audio with AASIST...")

    try:
        result = predict_audio(
            r"audio_processing\aasist_input.wav"
        )

        print("\n========== PREDICTION ==========")

        print(
            "Synthetic probability:",
            round(result["synthetic_probability"] * 100, 2),
            "%"
        )

        print(
            "Authentic probability:",
            round(result["authentic_probability"] * 100, 2),
            "%"
        )

        print(
            "Confidence:",
            round(result["model_confidence"] * 100, 2),
            "%"
        )

        print("================================")

    except Exception as e:
        print("Prediction error:", e)

    print("\nAudio analysis completed!")