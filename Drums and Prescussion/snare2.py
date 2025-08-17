import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import square, sawtooth

# Sampling rate
fs = 44100

# Duration for a single snare hit
single_snare_duration = 0.45

# --- Waveform Functions ---
def damped_sine(t, freq, amp, decay, phase=0):
    return amp * np.sin(2 * np.pi * freq * t + phase) * np.exp(-decay * t)

def damped_triangle(t, freq, amp, decay, phase=0):
    return amp * (2 * np.abs(2 * ((t * freq + phase/(2*np.pi)) % 1) - 1) - 1) * np.exp(-decay * t)

def damped_square(t, freq, amp, decay, phase=0):
    return amp * square(2 * np.pi * freq * t + phase) * np.exp(-decay * t)

def damped_sawtooth(t, freq, amp, decay, phase=0):
    return amp * sawtooth(2 * np.pi * freq * t + phase) * np.exp(-decay * t)

def white_noise(t, freq, amp=1.0, decay=8, phase=0):
    signal = amp * np.random.uniform(-1, 1, len(t))
    return signal * np.exp(-decay * t)

# Dictionary of waveform functions
wave_funcs = {
    'damped_sine': damped_sine,
    'damped_triangle': damped_triangle,
    'damped_square': damped_square,
    'damped_sawtooth': damped_sawtooth,
    'white_noise': white_noise
}

# Instrument class
class Instrument:
    def __init__(self, harmonic_intervals, duration):
        self.harmonic_intervals = harmonic_intervals
        self.duration = duration

    def synth(self, fs, root_freq):
        t = np.linspace(0, self.duration, int(fs * self.duration), endpoint=False)
        signal = np.zeros_like(t)
        for h in self.harmonic_intervals:
            func = wave_funcs[h['type']]
            freq = root_freq * h['ratio']
            amp = h['amp']
            decay = h['decay']
            phase = h.get('phase', 0)
            signal += func(t, freq, amp, decay, phase)
        return signal, t

# Harmonic intervals for snare2
snare2_intervals = [
    {'type': 'damped_sine', 'ratio': 1.0, 'amp': 0.7, 'decay': 6},
    {'type': 'white_noise', 'ratio': 1.0, 'amp': 0.5, 'decay': 8},    
    {'type': 'damped_triangle', 'ratio': 1.25, 'amp': 0.4, 'decay': 8},
    {'type': 'damped_sawtooth', 'ratio': 2.5, 'amp': 0.3, 'decay': 10},
    {'type': 'damped_sine', 'ratio': 3.1, 'amp': 0.2, 'decay': 12}
]

# Create snare instrument
snare2 = Instrument(harmonic_intervals=snare2_intervals, duration=single_snare_duration)

# Root frequency
root_note_freq = 432.5

# Generate one snare
single_snare_signal, _ = snare2.synth(fs, root_note_freq)

# Normalize
if np.max(np.abs(single_snare_signal)) > 0:
    single_snare_signal /= np.max(np.abs(single_snare_signal))

# Repeat snare with silence between hits
silence = np.zeros(int(fs * 0.1))
signal = np.concatenate([np.concatenate([single_snare_signal, silence]) for _ in range(5)])

# Time axis
t = np.linspace(0, len(signal)/fs, len(signal), endpoint=False)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(t[:int(fs * 1.0)], signal[:int(fs * 1.0)])
plt.title("Synthesized Snare - First Hit")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

print("Playing sound...")
sd.play(signal, fs)
plt.show()
sd.wait()
print("Playback finished.")
