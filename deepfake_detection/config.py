from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "deepfake" / "AASIST.pth"

SAMPLE_RATE = 16000
TARGET_SAMPLES = 64600

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
