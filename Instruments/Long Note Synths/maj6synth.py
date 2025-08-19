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
chromatic_intervals = [2 ** (0 / 12), 2 ** (1 / 12), 2 ** (2 / 12), 2 ** (3 / 12), 2 ** (4 / 12),
                       2 ** (5 / 12), 2 ** (6 / 12), 2 ** (7 / 12), 2 ** (8 / 12), 2 ** (9 / 12), 2 ** (10 / 12), 2 ** (11 / 12)]

# Define the harmonics for each chord tone (relative to the root)
maj6synth_harmonics = [
    # Root
{'type': 'sine', 'interval': chromatic_intervals[0], 'mult': 1, 'amp': 30.5},
{'type': 'sawtooth', 'interval': chromatic_intervals[0], 'mult': 0.125, 'amp': 4.5},
{'type': 'sine', 'interval': chromatic_intervals[0], 'mult': 2, 'amp': 0.3456},
{'type': 'sine', 'interval': chromatic_intervals[0], 'mult': 3, 'amp': 1.0368},
{'type': 'sine', 'interval': chromatic_intervals[0], 'mult': 4, 'amp': 0.3456},
{'type': 'sine', 'interval': chromatic_intervals[0], 'mult': 5, 'amp': 0.1728},
{'type': 'sine', 'interval': chromatic_intervals[0], 'mult': 6, 'amp': 0.13824},
{'type': 'sine', 'interval': chromatic_intervals[0], 'mult': 7, 'amp': 0.13824},
{'type': 'sine', 'interval': chromatic_intervals[0], 'mult': 8, 'amp': 0.01728},
{'type': 'sine', 'interval': chromatic_intervals[0], 'mult': 9, 'amp': 0.00864},
{'type': 'sine', 'interval': chromatic_intervals[0], 'mult': 10, 'amp': 0.003456},
# third
{'type': 'triangle', 'interval': chromatic_intervals[4], 'mult': 1, 'amp': 0.0},
{'type': 'triangle', 'interval': chromatic_intervals[4], 'mult': 2, 'amp': 0.12},
{'type': 'triangle', 'interval': chromatic_intervals[4], 'mult': 4, 'amp': 0.048},
{'type': 'triangle', 'interval': chromatic_intervals[4], 'mult': 8, 'amp': 0.024},
{'type': 'triangle', 'interval': chromatic_intervals[4], 'mult': 16, 'amp': 0.012},
{'type': 'triangle', 'interval': chromatic_intervals[4], 'mult': 32, 'amp': 0.006},
{'type': 'triangle', 'interval': chromatic_intervals[4], 'mult': 64, 'amp': 0.0024},
# fifth
{'type': 'sine', 'interval': chromatic_intervals[7], 'mult': 1, 'amp': 0.56},
{'type': 'sine', 'interval': chromatic_intervals[7], 'mult': 2, 'amp': 0.048},
{'type': 'sine', 'interval': chromatic_intervals[7], 'mult': 3, 'amp': 0.144},
{'type': 'sine', 'interval': chromatic_intervals[7], 'mult': 4, 'amp': 0.048},
{'type': 'sine', 'interval': chromatic_intervals[7], 'mult': 5, 'amp': 0.024},
{'type': 'sine', 'interval': chromatic_intervals[7], 'mult': 6, 'amp': 0.0192},
{'type': 'sine', 'interval': chromatic_intervals[7], 'mult': 7, 'amp': 0.0192},
{'type': 'sine', 'interval': chromatic_intervals[7], 'mult': 8, 'amp': 0.0024},
{'type': 'sine', 'interval': chromatic_intervals[7], 'mult': 9, 'amp': 0.0012},
{'type': 'sine', 'interval': chromatic_intervals[7], 'mult': 10, 'amp': 0.00048},
# sixth
{'type': 'sine', 'interval': chromatic_intervals[9], 'mult': 1, 'amp': 0.0},
{'type': 'sine', 'interval': chromatic_intervals[9], 'mult': 2, 'amp': 0.048},
{'type': 'sine', 'interval': chromatic_intervals[9], 'mult': 3, 'amp': 0.144},
{'type': 'sine', 'interval': chromatic_intervals[9], 'mult': 4, 'amp': 0.048},
]

# Create the instrument
maj6synth = Instrument(harmonics=maj6synth_harmonics, duration=4.0)

# Example usage: play with root 110 Hz (A1)
fs = 44100
root_freq = 48.109  # A1
signal, t = maj6synth.synth(root_freq, fs)
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