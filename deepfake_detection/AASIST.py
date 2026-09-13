import torch
from pathlib import Path

from .aasist_model import Model

class AASISTDetector:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        config = {
            "architecture": "AASIST",
            "nb_samp": 64600,
            "first_conv": 128,
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0],
        }

        self.model = Model(config).to(self.device)

        model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "deepfake"
            / "AASIST.pth"
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(self, audio):
        audio = audio.to(self.device)

        with torch.no_grad():
            _, output = self.model(audio)

        probabilities = torch.softmax(output, dim=1)

        authentic_probability = probabilities[0, 1].item()
        synthetic_probability = probabilities[0, 0].item()

        confidence = max(
            authentic_probability,
            synthetic_probability
        )

        return {
            "synthetic_probability": synthetic_probability,
            "authentic_probability": authentic_probability,
            "model_confidence": confidence
        }
