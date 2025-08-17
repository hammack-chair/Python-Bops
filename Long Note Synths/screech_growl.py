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
def damped_sine(t, freq, amp, decay, phase=0):
    return amp * np.sin(2 * np.pi * freq * t + phase) * np.exp(-decay * t)
def damped_triangle(t, freq, amp, decay, phase=0):
    return amp * (2 * np.abs(2 * ((t * freq + phase/(2*np.pi)) % 1) - 1) - 1) * np.exp(-decay * t)
def damped_square(t, freq, amp, decay, phase=0):
    return amp * square(2 * np.pi * freq * t + phase) * np.exp(-decay * t)

def damped_sawtooth(t, freq, amp, decay, phase=0):
    return amp * sawtooth(2 * np.pi * freq * t + phase) * np.exp(-decay * t)


wave_funcs = {
    'sine': sine_wave,
    'triangle': triangle_wave,
    'square': square_wave,
    'sawtooth': sawtooth_wave,
    'damped_sine': damped_sine,
    'damped_triangle': damped_triangle,
    'damped_square': damped_square,
    'damped_sawtooth': damped_sawtooth
}

class Instrument:
    def __init__(self, harmonics, duration=4.0):
        self.harmonics = harmonics
        self.duration = duration

    def synth(self, root_freq, fs, duration=None):
        if duration is None:
            duration = self.duration
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        signal = np.zeros_like(t)
        for h in self.harmonics:
            func = wave_funcs[h['type']]
            freq = root_freq * h['interval'] * h['mult']
            amp = h['amp']
            phase = h.get('phase', 0)
            decay = h.get('decay', None)
            if decay is not None:
                signal += func(t, freq, amp, decay, phase)
            else:
                signal += func(t, freq, amp, phase)
        return signal, t





chromatic_intervals = [2 ** (0 / 12), 2 ** (1 / 12), 2 ** (2 / 12), 2 ** (3 / 12), 2 ** (4 / 12),
                       2 ** (5 / 12), 2 ** (6 / 12), 2 ** (7 / 12), 2 ** (8 / 12), 2 ** (9 / 12), 2 ** (10 / 12), 2 ** (11 / 12)]

# Define the harmonics for each chord tone (relative to the root)
screech_growl_harmonics = [
    {'type': 'damped_sawtooth', 'interval': 1, 'mult': 0.5, 'amp': 0.1, 'decay': 0.2},
    {'type': 'damped_sawtooth', 'interval': 1, 'mult': 1, 'amp': 0.9, 'decay': 0.2},
    {'type': 'triangle', 'interval': 2 ** (12/12), 'mult': 2, 'amp': 0.09},
    {'type': 'square', 'interval': 2 ** (7/12), 'mult': 3, 'amp': 0.08},
    {'type': 'damped_sine', 'interval': 2 ** (5/12), 'mult': 1, 'amp': 0.5, 'decay': 1.0},
    {'type': 'sine', 'interval': 2 ** (19/12), 'mult': 1, 'amp': 0.2}
]


# Create the instrument
screech_growl = Instrument(harmonics=screech_growl_harmonics, duration=8.0)

# Example usage: play with root 110 Hz (A1)
fs = 44100
root_freq = 110  # A1
signal, t = screech_growl.synth(root_freq, fs)
signal /= np.max(np.abs(signal))

plt.figure(figsize=(10, 4))
plt.plot(t[:1000], signal[:1000])
plt.title("chromaticsynth with root 110 Hz (A1)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

sd.play(signal, fs)
plt.show()
sd.wait()