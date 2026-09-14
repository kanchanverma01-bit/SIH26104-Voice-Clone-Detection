import torch
from pathlib import Path

from .aasist_model import Model


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_SAMPLES = 64600

# Model label mapping
#
# CURRENT CHECKPOINT ASSUMPTION:
# Class 0 = SYNTHETIC / AI-GENERATED
# Class 1 = AUTHENTIC / REAL HUMAN
#
# If your trained model was trained with the opposite labels,
# this must be changed after verifying the training labels.
CLASS_0_IS_SYNTHETIC = True


class AASISTDetector:

    def __init__(self):

        # ====================================================
        # DEVICE
        # ====================================================

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ====================================================
        # AASIST MODEL CONFIGURATION
        # ====================================================

        config = {
            "architecture": "AASIST",

            "nb_samp": TARGET_SAMPLES,

            "first_conv": 128,

            "filts": [
                70,
                [1, 32],
                [32, 32],
                [32, 64],
                [64, 64]
            ],

            "gat_dims": [
                64,
                32
            ],

            "pool_ratios": [
                0.5,
                0.7,
                0.5,
                0.5
            ],

            "temperatures": [
                2.0,
                2.0,
                100.0,
                100.0
            ],
        }

        # ====================================================
        # CREATE MODEL
        # ====================================================

        self.model = Model(
            config
        ).to(
            self.device
        )

        # ====================================================
        # MODEL CHECKPOINT PATH
        # ====================================================

        model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "deepfake"
            / "AASIST.pth"
        )

        # ====================================================
        # CHECK MODEL EXISTS
        # ====================================================

        if not model_path.exists():

            raise FileNotFoundError(
                "AASIST model checkpoint not found.\n\n"
                f"Expected location:\n{model_path}"
            )

        # ====================================================
        # LOAD CHECKPOINT
        # ====================================================

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        # ====================================================
        # LOAD WEIGHTS
        # ====================================================

        self.model.load_state_dict(
            checkpoint
        )

        # ====================================================
        # EVALUATION MODE
        # ====================================================

        self.model.eval()

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        audio
    ):

        # ----------------------------------------------------
        # Move audio to correct device
        # ----------------------------------------------------

        audio = audio.to(
            self.device
        )

        # ----------------------------------------------------
        # Model inference
        # ----------------------------------------------------

        with torch.no_grad():

            _, output = self.model(
                audio
            )

        # ----------------------------------------------------
        # Convert logits to probabilities
        # ----------------------------------------------------

        probabilities = torch.softmax(
            output,
            dim=1
        )

        # ----------------------------------------------------
        # Extract raw class probabilities
        # ----------------------------------------------------

        class_0_probability = float(
            probabilities[0, 0].item()
        )

        class_1_probability = float(
            probabilities[0, 1].item()
        )

        # ----------------------------------------------------
        # Map classes to semantic labels
        # ----------------------------------------------------

        if CLASS_0_IS_SYNTHETIC:

            synthetic_probability = (
                class_0_probability
            )

            authentic_probability = (
                class_1_probability
            )

        else:

            synthetic_probability = (
                class_1_probability
            )

            authentic_probability = (
                class_0_probability
            )

        # ----------------------------------------------------
        # Model confidence
        # ----------------------------------------------------

        model_confidence = max(
            synthetic_probability,
            authentic_probability
        )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        return {

            "synthetic_probability":
                synthetic_probability,

            "authentic_probability":
                authentic_probability,

            "model_confidence":
                model_confidence,

            # Debug values
            "class_0_probability":
                class_0_probability,

            "class_1_probability":
                class_1_probability,
        }