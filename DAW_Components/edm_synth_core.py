# Core library for instruments, waveforms, and timeline scheduling.

import numpy as np
from scipy.signal import square, sawtooth

# ------------------------------
# WAVEFORMS
# ------------------------------

def sine_wave(t, freq, amp, phase=0):
    return amp * np.sin(2 * np.pi * freq * t + phase)

def triangle_wave(t, freq, amp, phase=0):
    return amp * (2 * np.abs(2 * ((t * freq + phase/(2*np.pi)) % 1) - 1) - 1)

def square_wave(t, freq, amp, phase=0):
    return amp * square(2 * np.pi * freq * t + phase)

def sawtooth_wave(t, freq, amp, phase=0):
    return amp * sawtooth(2 * np.pi * freq * t + phase)

def damped(func):
    # Decorator to add exponential decay to a waveform
    def wrapper(t, freq, amp, decay=1.0, phase=0):
        return func(t, freq, amp, phase) * np.exp(-decay * t)
    return wrapper

# Damped variants
damped_sine = damped(sine_wave)
damped_triangle = damped(triangle_wave)
damped_square = damped(square_wave)
damped_sawtooth = damped(sawtooth_wave)

# Noise generators

def white_noise(t, amp=1.0, decay=8):
    return amp * np.random.uniform(-1, 1, len(t)) * np.exp(-decay * t)

def pink_noise(t, amp=1.0, decay=8, num_sources=16):
    n = len(t)
    array = np.zeros(n)
    white = np.random.randn(num_sources, n)
    running_sum = np.zeros(n)
    for i in range(num_sources):
        step = 2 ** i
        running_sum += np.repeat(np.add.reduceat(white[i], np.arange(0, n, step)) / step, step)[:n]
    signal = amp * running_sum / np.max(np.abs(running_sum))
    return signal * np.exp(-decay * t)

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

# ------------------------------
# INSTRUMENT
# ------------------------------

class Instrument:
    def __init__(self, harmonics, duration=4.0):
        self.harmonics = harmonics
        self.duration = duration

    def synth(self, root_freq, fs, duration=None):
        dur = float(self.duration if duration is None else duration)
        t = np.linspace(0, dur, int(fs * dur), endpoint=False)
        if t.size == 0:
            return np.zeros(0, dtype=float)

        # Precompute function references for speed
        funcs = [(wave_funcs[h['type']], h) for h in self.harmonics]
        out = np.zeros_like(t)

        for func, h in funcs:
            freq = root_freq * h.get('interval', h.get('ratio',1.0)) * h.get('mult', 1.0)
            amp = h.get('amp', 1.0)
            decay = h.get('decay', 0)
            phase = h.get('phase', 0)

            # Only pass decay if function accepts it (damped waveforms)
            if 'decay' in func.__code__.co_varnames:
                out += func(t, freq, amp, decay, phase)
            else:
                out += func(t, freq, amp, phase)

        peak = np.max(np.abs(out))
        if peak > 1e-9:
            out = 0.95 * out / peak
        return out

# ------------------------------
# TIMELINE
# ------------------------------

class Timeline:
    def __init__(self, bpm, time_signature=(4,4), measures=4, fs=44100):
        self.bpm = float(bpm)
        self.beats_per_measure = int(time_signature[0])
        self.note_value = int(time_signature[1])
        self.measures = int(measures)
        self.fs = int(fs)
        self.events = []

    def seconds_per_beat(self):
        return (60.0 / self.bpm) * (4.0 / self.note_value)

    def beats_to_seconds(self, beats):
        return beats * self.seconds_per_beat()

    def add_note(self, instrument, root_freq, start_beat, end_beat, volume=1.0):
        duration_s = self.beats_to_seconds(end_beat - start_beat)
        start_time_s = self.beats_to_seconds(start_beat)
        note = instrument.synth(root_freq, self.fs, duration=duration_s) * volume
        self.events.append({'t0': start_time_s, 'wave': note})

    def render(self):
        total_s = max(self.beats_to_seconds(self.beats_per_measure * self.measures),
                      max((e['t0'] + len(e['wave'])/self.fs for e in self.events), default=0))
        out = np.zeros(int(self.fs * total_s), dtype=float)
        for e in self.events:
            start = int(e['t0'] * self.fs)
            end = start + len(e['wave'])
            out[start:end] += e['wave']

        peak = np.max(np.abs(out))
        if peak > 1e-9:
            out = 0.98 * out / peak
        return out

    def play(self):
        import sounddevice as sd
        audio = self.render()
        if audio.size:
            sd.play(audio, self.fs)
            sd.wait()
