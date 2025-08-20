import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square, sawtooth
import sounddevice as sd

def sine_wave(t, freq, amp, phase=0):
    return amp * np.sin(2 * np.pi * freq * t + phase)

def triangle_wave(t, freq, amp, phase=0):
    return amp * (2 * np.abs(2 * ((t * freq + phase/(2*np.pi)) % 1) - 1) - 1)

def square_wave(t, freq, amp, phase=0):
    return amp * square(2 * np.pi * freq * t + phase)

def sawtooth_wave(t, freq, amp, phase=0):
    return amp * sawtooth(2 * np.pi * freq * t + phase)

wave_funcs = {
    'sine': sine_wave,
    'triangle': triangle_wave,
    'square': square_wave,
    'sawtooth': sawtooth_wave
}

class Instrument:
    def __init__(self, harmonics, duration=4.0):
        self.harmonics = harmonics
        self.duration = duration

    def synth(self, root_freq, fs):
        t = np.linspace(0, self.duration, int(fs * self.duration), endpoint=False)
        signal = np.zeros_like(t)
        for h in self.harmonics:
            func = wave_funcs[h['type']]
            freq = root_freq * h['interval'] * h['mult']
            amp = h['amp']
            phase = h.get('phase', 0)
            signal += func(t, freq, amp, phase)
        return signal, t

# Define the intervals for a major 7th chord: root (1), fifth (3/2), major seventh (15/8)
maj7_intervals = [1, 3/2, 15/8]

# Define the harmonics for each chord tone (relative to the root)
maj7_harmonics = [
    # Root
    {'type': 'sine', 'interval': maj7_intervals[0], 'mult': 1, 'amp': 10.5},
    {'type': 'sine', 'interval': maj7_intervals[0], 'mult': 2, 'amp': 0.3},
    {'type': 'sine', 'interval': maj7_intervals[0], 'mult': 3, 'amp': 0.9},
    {'type': 'sine', 'interval': maj7_intervals[0], 'mult': 4, 'amp': 0.3},
    {'type': 'sine', 'interval': maj7_intervals[0], 'mult': 5, 'amp': 0.15},
    {'type': 'sine', 'interval': maj7_intervals[0], 'mult': 6, 'amp': 0.12},
    {'type': 'sine', 'interval': maj7_intervals[0], 'mult': 7, 'amp': 0.12},
    {'type': 'sine', 'interval': maj7_intervals[0], 'mult': 8, 'amp': 0.015},
    {'type': 'sine', 'interval': maj7_intervals[0], 'mult': 9, 'amp': 0.0075},
    {'type': 'sine', 'interval': maj7_intervals[0], 'mult': 10, 'amp': 0.003},
    # Fifth
    {'type': 'triangle', 'interval': maj7_intervals[1], 'mult': 1, 'amp': 0.08},
    {'type': 'triangle', 'interval': maj7_intervals[1], 'mult': 2, 'amp': 0.05},
    {'type': 'triangle', 'interval': maj7_intervals[1], 'mult': 4, 'amp': 0.02},
    {'type': 'sine', 'interval': maj7_intervals[1], 'mult': 8, 'amp': 0.01},
    {'type': 'sine', 'interval': maj7_intervals[1], 'mult': 16, 'amp': 0.005},
    {'type': 'sine', 'interval': maj7_intervals[1], 'mult': 32, 'amp': 0.0025},
    {'type': 'sine', 'interval': maj7_intervals[1], 'mult': 64, 'amp': 0.001},
    # Major seventh
    {'type': 'square', 'interval': maj7_intervals[2], 'mult': 1, 'amp': 0.05},
    {'type': 'sawtooth', 'interval': maj7_intervals[2], 'mult': 2, 'amp': 0.07},
    {'type': 'sine', 'interval': maj7_intervals[2], 'mult': 4, 'amp': 0.03}
]

# Create the instrument
maj7synth = Instrument(harmonics=maj7_harmonics, duration=8.0)

# Example usage: play a major 7th chord with root 55 Hz (A1)
fs = 44100
root_freq = 432  # A1
signal, t = maj7synth.synth(root_freq, fs)
signal /= np.max(np.abs(signal))

plt.figure(figsize=(10, 4))
plt.plot(t[:1000], signal[:1000])
plt.title("maj7synth with root 55 Hz (A1)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)


sd.play(signal, fs)
plt.show()
sd.wait()