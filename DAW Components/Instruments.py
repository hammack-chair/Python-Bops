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
    def __init__(self, base_freq, harmonics):
        """
        harmonics: list of dicts, each with keys:
            'type': 'sine'|'triangle'|'square'|'sawtooth'
            'mult': frequency multiplier (1 for fundamental, 2 for 2nd harmonic, etc.)
            'amp': amplitude for this harmonic
            'phase': phase offset (optional)
        """
        self.base_freq = base_freq
        self.harmonics = harmonics

    def synth(self, t):
        signal = np.zeros_like(t)
        for h in self.harmonics:
            func = wave_funcs[h['type']]
            freq = self.base_freq * h['mult']
            amp = h['amp']
            phase = h.get('phase', 0)
            signal += func(t, freq, amp, phase)
        return signal