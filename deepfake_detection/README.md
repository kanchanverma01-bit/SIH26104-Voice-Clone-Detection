# Deepfake Detection

This module detects whether a voice recording is authentic or AI-generated/spoofed.

## Model

The module uses **AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)** for audio deepfake/spoof detection.

AASIST is designed for detecting spoofed speech by learning spectro-temporal characteristics from raw audio.

## Input

The detector accepts a speech audio file.

Audio is:
- Converted to mono if required
- Resampled to **16 kHz**
- Prepared as exactly **64,600 samples (~4.04 seconds)** for AASIST inference
- Shorter audio is repeated to reach the required length

## Output

The module returns the standardized team contract:

```python
{
    "synthetic_probability": 0.0,
    "authentic_probability": 0.0,
    "model_confidence": 0.0
}
