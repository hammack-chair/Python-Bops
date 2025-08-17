import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square, sawtooth
import sounddevice as sd

def sine_wave(t, freq, amp, phase=0):
    return amp * np.sin(2 * np.pi * freq * t + phase)

def triangle_wave(t, freq, amp, phase=0):
    return amp * (2 * np.abs(2 * ((t * freq + phase/(2*np.pi)) % 1) - 1) - 1)

def square_wave(t, freq, amp, phase=0):
    # Lazy import to avoid scipy hard dependency if you swap funcs
    from scipy.signal import square
    return amp * square(2 * np.pi * freq * t + phase)

def sawtooth_wave(t, freq, amp, phase=0):
    from scipy.signal import sawtooth
    return amp * sawtooth(2 * np.pi * freq * t + phase)

def damped_sine(t, freq, amp, decay, phase=0):
    return amp * np.sin(2 * np.pi * freq * t + phase) * np.exp(-decay * t)

def damped_triangle(t, freq, amp, decay, phase=0):
    return amp * (2 * np.abs(2 * ((t * freq + phase/(2*np.pi)) % 1) - 1) - 1) * np.exp(-decay * t)

def damped_square(t, freq, amp, decay, phase=0):
    from scipy.signal import square
    return amp * square(2 * np.pi * freq * t + phase) * np.exp(-decay * t)

def damped_sawtooth(t, freq, amp, decay, phase=0):
    from scipy.signal import sawtooth
    return amp * sawtooth(2 * np.pi * freq * t + phase) * np.exp(-decay * t)

def white_noise(t, freq, amp=1.0, decay=8, phase=0):
    signal = amp * np.random.uniform(-1, 1, len(t))
    return signal * np.exp(-decay * t)

def pink_noise(t, freq=None, amp=1.0, decay=8, phase=0):
    # Voss-McCartney pink noise approximation
    n = len(t)
    num_sources = 16
    array = np.zeros(n)
    white = np.random.randn(num_sources, n)
    running_sum = np.zeros(n)
    for i in range(num_sources):
        step = 2 ** i
        running_sum += np.repeat(np.add.reduceat(white[i], np.arange(0, n, step)) / step, step)[:n]
    signal = amp * running_sum / np.max(np.abs(running_sum))
    return signal * np.exp(-decay * t)

# Dictionary of waveform functions
wave_funcs = {
    'sine': sine_wave,
    'triangle': triangle_wave,
    'square': square_wave,
    'sawtooth': sawtooth_wave,
    'damped_sine': damped_sine,
    'damped_triangle': damped_triangle,
    'damped_square': damped_square,
    'damped_sawtooth': damped_sawtooth,
    'white_noise': white_noise,
    'pink_noise': pink_noise
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
            freq = root_freq * h['ratio']
            amp = h['amp']
            phase = h.get('phase', 0)
            signal += func(t, freq, amp, phase)
        return signal, t

# Define the intervals for a major 7th chord: root (1), fifth (3/2), major seventh (15/8)
maj7_intervals = [1, 3/2, 15/8]

# Define the harmonics for each chord tone (relative to the root)
laser_harmonic_intervals = [
    {'type': 'damped_sine', 'ratio': 1.000, 'amp': 1.000, 'decay': 2},
    {'type': 'damped_sine', 'ratio': 4.030, 'amp': 0.370, 'decay': 3},
    {'type': 'damped_sine', 'ratio': 5.940, 'amp': 0.290, 'decay': 5},
    {'type': 'damped_sine', 'ratio': 7.030, 'amp': 0.250, 'decay': 5},
    {'type': 'damped_sine', 'ratio': 8.060, 'amp': 0.210, 'decay': 2.1},
    {'type': 'damped_sine', 'ratio': 9.960, 'amp': 0.180, 'decay': 2.1},
    {'type': 'damped_sine', 'ratio': 11.000, 'amp': 0.150, 'decay': 4.2},
    {'type': 'damped_sine', 'ratio': 12.000, 'amp': 0.120, 'decay': 8},
    {'type': 'damped_sine', 'ratio': 13.050, 'amp': 0.100, 'decay': 7.6},
    {'type': 'damped_sine', 'ratio': 14.100, 'amp': 0.080, 'decay': 7},
    {'type': 'damped_sine', 'ratio': 1.700, 'amp': 0.120, 'decay': 8.5},
    {'type': 'damped_sine', 'ratio': 2.400, 'amp': 0.110, 'decay': 8},
    {'type': 'damped_sine', 'ratio': 3.200, 'amp': 0.090, 'decay': 8},
    {'type': 'damped_sine', 'ratio': 4.800, 'amp': 0.080, 'decay': 8},
    {'type': 'damped_sine', 'ratio': 6.500, 'amp': 0.070, 'decay': 8},
    {'type': 'damped_sine', 'ratio': 7.800, 'amp': 0.060, 'decay': 8},
    {'type': 'damped_sine', 'ratio': 9.200, 'amp': 0.050, 'decay': 8},
    {'type': 'damped_sine', 'ratio': 10.500, 'amp': 0.040, 'decay': 8},
    {'type': 'damped_sine', 'ratio': 11.800, 'amp': 0.030, 'decay': 8},
    {'type': 'damped_sine', 'ratio': 13.200, 'amp': 0.025, 'decay': 8}
]


# Create the instrument
laser = Instrument(harmonics=laser_harmonic_intervals, duration=2.0)

# Example usage: play a major 7th chord with root 55 Hz (A1)
fs = 44100
root_freq = 1728  # A1
signal, t = laser.synth(root_freq, fs)
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