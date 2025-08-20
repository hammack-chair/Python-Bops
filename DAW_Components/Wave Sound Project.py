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
    def __init__(self, base_freq, harmonics, duration=4.0):
        self.base_freq = base_freq
        self.harmonics = harmonics
        self.duration = duration

    def synth(self, fs):
        t = np.linspace(0, self.duration, int(fs * self.duration), endpoint=False)
        signal = np.zeros_like(t)
        for h in self.harmonics:
            func = wave_funcs[h['type']]
            freq = self.base_freq * h['mult']
            amp = h['amp']
            phase = h.get('phase', 0)
            signal += func(t, freq, amp, phase)
        return signal, t

# Example instruments with custom durations
piano = Instrument(
    base_freq=27,
    harmonics=[
        {'type': 'sine', 'mult': 1, 'amp': 3.5},
        {'type': 'sine', 'mult': 2, 'amp': 0.1},
        {'type': 'sine', 'mult': 3, 'amp': 0.3},
        {'type': 'sine', 'mult': 4, 'amp': 0.1},
        {'type': 'sine', 'mult': 5, 'amp': 0.05},
        {'type': 'sine', 'mult': 6, 'amp': 0.04},
        {'type': 'sine', 'mult': 7, 'amp': 0.04},
        {'type': 'sine', 'mult': 8, 'amp': 0.005},
        {'type': 'sine', 'mult': 9, 'amp': 0.0025},
        {'type': 'sine', 'mult': 10, 'amp': 0.001}
    ],
    duration=8.0
)

organ = Instrument(
    base_freq=40.454,
    harmonics=[
        {'type': 'square', 'mult': 1, 'amp': 0.05},
        {'type': 'sawtooth', 'mult': 2, 'amp': 0.07},
        {'type': 'sine', 'mult': 4, 'amp': 0.3}
    ],
    duration=6.0
)

borgan = Instrument(
    base_freq=203.877,
    harmonics=[
        {'type': 'triangle', 'mult': 1, 'amp': 0.08},
        {'type': 'triangle', 'mult': 2, 'amp': 0.05},
        {'type': 'triangle', 'mult': 4, 'amp': 0.02},
        {'type': 'sine', 'mult': 8, 'amp': 0.01},
        {'type': 'sine', 'mult': 16, 'amp': 0.005},
        {'type': 'sine', 'mult': 32, 'amp': 0.0025},
        {'type': 'sine', 'mult': 64, 'amp': 0.001}
    ],
    duration=4.0
)



fs = 44100

# Synthesize each instrument
piano_signal, piano_t = piano.synth(fs)
organ_signal, organ_t = organ.synth(fs)
borgan_signal, borgan_t = borgan.synth(fs)

# Mix signals: pad shorter signal with zeros
max_len = max(len(piano_signal), len(organ_signal), len(borgan_signal))
piano_signal = np.pad(piano_signal, (0, max_len - len(piano_signal)))
organ_signal = np.pad(organ_signal, (0, max_len - len(organ_signal)))
borgan_signal = np.pad(borgan_signal, (0, max_len - len(borgan_signal)))
signal = piano_signal + organ_signal + borgan_signal
signal /= np.max(np.abs(signal))  # Normalize

# Use the longest time array for plotting
t = np.linspace(0, max(piano.duration, organ.duration, borgan.duration), max_len, endpoint=False)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(t[:1000], signal[:1000])
plt.title("Custom Instruments Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)


# Play sound
sd.play(signal, fs)
plt.show()
sd.wait()