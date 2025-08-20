import numpy as np
import sounddevice as sd
import inspect

# ------------------------------
# WAVEFORMS (import-safe: no plotting or playback here)
# ------------------------------

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


# ------------------------------
# INSTRUMENT (per-note overrideable; **import-safe**)
# ------------------------------
class Instrument:
    def __init__(self, harmonics, duration=4.0):
        """
        harmonics: list of dicts like
          { 'type': 'damped_sine', 'interval': 2**(7/12), 'mult': 1, 'amp': 0.5, 'decay': 2.0, 'phase': 0 }
        duration: default seconds if not overridden by timeline
        """
        self.harmonics = harmonics
        self.duration = duration

    def synth(self, root_freq, fs, *, duration=None):
        """Synthesize a note.
        - root_freq: base frequency for this note (overwrites any prior value)
        - fs: sample rate
        - duration: optional seconds (overwrites self.duration if provided)
        Returns: np.array signal
        """
        dur = float(self.duration if duration is None else duration)
        t = np.linspace(0, dur, int(fs * dur), endpoint=False)
        if t.size == 0:
            return np.zeros(0, dtype=float)

        out = np.zeros_like(t)

        for h in self.harmonics:
            func = wave_funcs[h['type']]
            sig = inspect.signature(func)

            freq = root_freq * h.get('interval', 1.0) * h.get('mult', 1.0)
            amp = h.get('amp', 1.0)
            phase = h.get('phase', 0)

            # Build kwargs only for params that the function actually accepts
            kwargs = {'t': t, 'freq': freq, 'amp': amp}
            if 'decay' in sig.parameters and 'decay' in h:
                kwargs['decay'] = h['decay']
            if 'phase' in sig.parameters:
                kwargs['phase'] = phase

            out += func(**kwargs)

        # Prevent clipping on a per-note basis (safe headroom)
        peak = np.max(np.abs(out))
        if peak > 1e-9:
            out = 0.95 * out / peak
        return out
