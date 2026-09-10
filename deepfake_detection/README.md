# Deepfake Voice Detection

## Overview

This module detects whether an input voice recording is likely to be **authentic (real)** or **synthetic (AI-generated / cloned)**.

It uses the **AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)** deep learning architecture for audio anti-spoofing.

The module focuses only on **voice authenticity detection**. Final fraud risk and prevention decisions are handled by the downstream Risk Engine.

---

## Detection Pipeline

```text
Input Audio
     ↓
Audio Loading
     ↓
Stereo → Mono
     ↓
Resampling → 16 kHz
     ↓
Audio Length Normalization
     ↓
AASIST Model
     ↓
Authentic / Synthetic Probabilities
     ↓
Standard JSON Output
