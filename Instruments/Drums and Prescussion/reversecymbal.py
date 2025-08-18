import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import square, sawtooth

# Sampling rate
fs = 44100

# Duration for a single reversed cymbal-like hit
reverse_cymbal_duration = 0.6

# --- Waveform Functions ---
def white_noise(t, freq=None, amp=1.0, decay=1.0, phase=0):
    signal = amp * np.random.uniform(-1, 1, len(t))
    return signal

def pink_noise(t, freq=None, amp=1.0, decay=1.0, phase=0):
    n = len(t)
    num_sources = 16
    array = np.zeros(n)
    white = np.random.randn(num_sources, n)
    running_sum = np.zeros(n)
    for i in range(num_sources):
        step = 2 ** i
        running_sum += np.repeat(np.add.reduceat(white[i], np.arange(0, n, step)) / step, step)[:n]
    signal = running_sum / np.max(np.abs(running_sum))
    return signal * amp

# Dictionary of waveform functions
wave_funcs = {
    'white_noise': white_noise,
    'pink_noise': pink_noise
}

# Instrument class
class Instrument:
    def __init__(self, harmonic_intervals, duration):
        self.harmonic_intervals = harmonic_intervals
        self.duration = duration

    def synth(self, fs, root_freq=None):
        t = np.linspace(0, self.duration, int(fs * self.duration), endpoint=False)
        signal = np.zeros_like(t)
        for h in self.harmonic_intervals:
            func = wave_funcs[h['type']]
            amp = h['amp']
            decay = h.get('decay', 1.0)
            s = func(t, amp=amp, decay=decay)
            signal += s
        # Apply reverse envelope (starts quiet, ends abruptly)
        envelope = np.sqrt(np.linspace(0, 1, len(t)))  # soft attack
        signal *= envelope
        return signal, t

# Harmonic set for reverse cymbal effect
reverse_cymbal_intervals = [
    {'type': 'white_noise', 'amp': 1.0},
    {'type': 'pink_noise', 'amp': 0.7}
]

# Create instrument
reverse_cymbal = Instrument(harmonic_intervals=reverse_cymbal_intervals, duration=reverse_cymbal_duration)

# Generate sound
signal, t = reverse_cymbal.synth(fs)

# Normalize
signal /= np.max(np.abs(signal))

# Plot
plt.figure(figsize=(10, 4))
plt.plot(t, signal)
plt.title('Reversed Cymbal-Like Hit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Play sound
print("Playing sound...")
sd.play(signal, fs)
sd.wait()
print("Playback finished.")
