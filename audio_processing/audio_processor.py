import sounddevice as sd
import soundfile as sf

sample_rate = 16000
duration = 5

print("Recording started...")
print("Speak now!")

audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype="float32"
)

sd.wait()

print("Recording completed!")

sf.write("my_voice.wav", audio, sample_rate)

print("Audio saved as my_voice.wav")
import matplotlib.pyplot as plt
import numpy as np

audio = audio.flatten()

time = np.arange(len(audio)) / sample_rate

plt.figure(figsize=(10, 4))
plt.plot(time, audio)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Voice Waveform")

plt.show()
from scipy.fft import rfft, rfftfreq

N = len(audio)

frequency = rfftfreq(
    N,
    1 / sample_rate
)

magnitude = np.abs(
    rfft(audio)
)

plt.figure(figsize=(10, 4))

plt.plot(frequency, magnitude)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.title("Voice Frequency Spectrum")

plt.xlim(0, 8000)

plt.show()
from scipy.fft import rfft, rfftfreq

# Total number of samples
N = len(audio)

# Calculate frequency values
frequency = rfftfreq(N, 1 / sample_rate)

# Calculate magnitude
magnitude = np.abs(rfft(audio))

# Plot frequency spectrum
plt.figure(figsize=(10, 4))

plt.plot(frequency, magnitude)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Voice Frequency Spectrum")

# Show frequencies up to 8000 Hz
plt.xlim(0, 8000)

plt.show()
import librosa
import librosa.display

# Convert audio into time-frequency representation
D = librosa.stft(audio)

# Convert amplitude into decibels
S_db = librosa.amplitude_to_db(
    np.abs(D),
    ref=np.max
)

# Create spectrogram
plt.figure(figsize=(10, 5))

librosa.display.specshow(
    S_db,
    sr=sample_rate,
    x_axis="time",
    y_axis="hz"
)

plt.colorbar(format="%+2.0f dB")

plt.title("Voice Spectrogram")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")

plt.show()
# ==============================
# VOICE ACTIVITY DETECTION (VAD)
# ==============================

import numpy as np

# Audio ko normalize karna
audio = audio / (np.max(np.abs(audio)) + 1e-8)

# Frame duration = 30 milliseconds
frame_duration = 0.03

# Ek frame me kitne samples honge
frame_length = int(sample_rate * frame_duration)

rms_values = []

# Har frame ka RMS calculate karna
for start in range(0, len(audio), frame_length):

    frame = audio[start:start + frame_length]

    if len(frame) == 0:
        continue

    rms = np.sqrt(np.mean(frame ** 2) + 1e-10)

    rms_values.append(rms)

# Voice detection threshold
threshold = 0.1 * np.max(rms_values)

# Voice wale frames identify karna
speech_frames = np.array(rms_values) > threshold

# Check karna ki voice mili ya nahi
speech_indices = np.where(speech_frames)[0]

if len(speech_indices) > 0:

    start_frame = speech_indices[0]
    end_frame = speech_indices[-1]

    start_sample = start_frame * frame_length
    end_sample = min((end_frame + 1) * frame_length, len(audio))

    speech_audio = audio[start_sample:end_sample]

    print("Voice detected!")
    print("Speech duration:",
          round(len(speech_audio) / sample_rate, 2),
          "seconds")

else:

    speech_audio = np.array([])

    print("No voice detected.")
    # ==============================
# SAVE SPEECH ONLY
# ==============================

import soundfile as sf

if len(speech_audio) > 0:

    sf.write(
        "speech_only.wav",
        speech_audio,
        sample_rate
    )

    print("Speech audio saved as speech_only.wav")

else:
    print("No speech audio to save.")
# ==============================
# NOISE REDUCTION
# ==============================

import noisereduce as nr
import soundfile as sf

# Noise reduce karna
clean_audio = nr.reduce_noise(
    y=speech_audio,
    sr=sample_rate
)

# Clean audio save karna
sf.write(
    "clean_speech.wav",
    clean_audio,
    sample_rate
)

print("Noise reduction completed!")
print("Clean audio saved as clean_speech.wav")
# ==============================
# AUDIO PREPROCESSING
# ==============================

import librosa
import soundfile as sf
import numpy as np

# Clean audio load karo
processed_audio, _ = librosa.load(
    "clean_speech.wav",
    sr=16000,
    mono=True
)

# Normalize audio
processed_audio = processed_audio / (
    np.max(np.abs(processed_audio)) + 1e-8
)

# Final ML-ready audio save karo
sf.write(
    "processed_audio.wav",
    processed_audio,
    16000
)

print("Audio preprocessing completed!")
print("ML-ready audio saved as processed_audio.wav")
# ==============================
# AASIST INPUT PREPARATION
# ==============================

import librosa
import soundfile as sf
import numpy as np

TARGET_SAMPLES = 64600

audio, sr = librosa.load(
    "processed_audio.wav",
    sr=16000,
    mono=True
)

if len(audio) < TARGET_SAMPLES:
    repeat_count = int(np.ceil(TARGET_SAMPLES / len(audio)))
    audio = np.tile(audio, repeat_count)

audio = audio[:TARGET_SAMPLES]

sf.write(
    "aasist_input.wav",
    audio,
    16000
)

print("AASIST input preparation completed!")
print("Samples:", len(audio))
print("Duration:", round(len(audio) / 16000, 2), "seconds")
print("Saved as: aasist_input.wav")