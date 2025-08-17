# Make sure to install the required libraries:
# pip install soundfile numpy matplotlib scipy librosa
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, sosfilt, hilbert
from scipy.optimize import curve_fit

# Load MP3 file using librosa
file_path = r'C:\Users\keena\Downloads\dre-muay-psy-kick-89602.mp3'
signal, fs = librosa.load(file_path, sr=None, mono=True)  # sr=None preserves original sampling rate

# Convert to mono if stereo
if signal.ndim > 1:
    signal = signal.mean(axis=1)

# Select a short segment (first 0.25s)
duration = len(signal) / fs
samples = int(fs * duration)
segment = signal[:samples]

# Perform FFT
fft_result = np.abs(fft(segment))
freqs = fftfreq(len(segment), 1/fs)

# Use only positive frequencies
pos_mask = freqs > 0
freqs = freqs[pos_mask]
fft_result = fft_result[pos_mask]

# Main peaks (higher threshold)
main_threshold = np.max(fft_result) * 0.2
main_indices = np.where(fft_result > main_threshold)[0]
sorted_main_peaks = main_indices[np.argsort(fft_result[main_indices])[::-1]]

# Subpeaks (lower threshold)
sub_threshold = np.max(fft_result) * 0.05
sub_indices = np.where((fft_result > sub_threshold) & (fft_result <= main_threshold))[0]
sorted_sub_peaks = sub_indices[np.argsort(fft_result[sub_indices])[::-1]]

# Normalize to base pitch (first main peak as root note)
base_freq = freqs[sorted_main_peaks[0]]

print("Base frequency (root note):", base_freq)
print("\nHarmonic wavefunctions for instrument:")

# Define exponential decay function
def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)

# Bandpass filter design
def bandpass_filter(signal, fs, center_freq, bandwidth=10):
    low = (center_freq - bandwidth/2) / (fs / 2)
    high = (center_freq + bandwidth/2) / (fs / 2)
    sos = butter(4, [low, high], btype='bandpass', output='sos')
    return sosfilt(sos, signal)

# Estimate decay time in seconds
def estimate_decay_time(signal, fs, freq):
    filtered = bandpass_filter(signal, fs, freq, bandwidth=10)
    envelope = np.abs(hilbert(filtered))
    time = np.linspace(0, len(envelope)/fs, len(envelope))
    
    try:
        popt, _ = curve_fit(exp_decay, time, envelope, p0=(np.max(envelope), 0.1), maxfev=10000)
        A, tau = popt
        decay_time = -tau * np.log(0.05)  # time to decay to 5%
        return round(decay_time, 3)
    except:
        return None

harmonics = []
for i, idx in enumerate(sorted_main_peaks[:10]):
    harmonic_freq = freqs[idx]
    amplitude = fft_result[idx] / np.max(fft_result)
    ratio = harmonic_freq / base_freq
    decay_time = estimate_decay_time(segment, fs, harmonic_freq)
    harmonics.append({
        'type': 'damped_sine',
        'ratio': round(ratio, 3),
        'amp': round(amplitude, 3),
        'decay': decay_time if decay_time is not None else 'unknown'
    })

subpeaks = []
for i, idx in enumerate(sorted_sub_peaks[:10]):
    harmonic_freq = freqs[idx]
    amplitude = fft_result[idx] / np.max(fft_result)
    ratio = harmonic_freq / base_freq
    decay_time = estimate_decay_time(segment, fs, harmonic_freq)
    subpeaks.append({
        'type': 'damped_sine',
        'ratio': round(ratio, 3),
        'amp': round(amplitude, 3),
        'decay': decay_time if decay_time is not None else 'unknown'
    })

print("\nmain_peaks = [")
for h in harmonics:
    print(f"    {h},")
print("\n")

print("# subpeaks")
for h in subpeaks:
    print(f"    {h},")
print("]")

# Optional: plot the spectrum
plt.figure(figsize=(10, 4))
plt.plot(freqs, fft_result)
plt.axhline(main_threshold, color='r', linestyle='--', label='Main Threshold')
plt.axhline(sub_threshold, color='g', linestyle='--', label='Sub Threshold')
plt.title('Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
