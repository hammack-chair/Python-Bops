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

# ------------------------------
# TIMELINE (beats scheduling; per-note duration & root override)
# ------------------------------
class Timeline:
    def __init__(self, bpm, time_signature=(4,4), measures=4, fs=44100):
        self.bpm = float(bpm)
        self.time_signature = time_signature  # <--- add this line
        self.beats_per_measure = int(time_signature[0])
        self.note_value = int(time_signature[1])  # denominator (4=quarter, 8=eighth)
        self.measures = int(measures)
        self.fs = int(fs)
        self.events = []  # list of dicts

    def seconds_per_beat(self):
        # e.g., in 6/8, a "beat" is an eighth note: (60/bpm) * (4/8)
        return (60.0 / self.bpm) * (4.0 / self.note_value)

    def beats_to_seconds(self, beats):
        return beats * self.seconds_per_beat()

    def add_note(self, instrument, root_freq, start_beat, end_beat, volume=1.0):
        """Schedule a note:
        - instrument: Instrument instance
        - root_freq: per-note root frequency (overwrites any prior value)
        - start_beat, end_beat: in beats (end exclusive). Duration is end-start.
        - volume: linear gain applied to this note (optional)
        """
        start_beat = float(start_beat)
        end_beat = float(end_beat)
        assert end_beat > start_beat, "end_beat must be > start_beat"

        duration_s = self.beats_to_seconds(end_beat - start_beat)
        start_time_s = self.beats_to_seconds(start_beat)

        # Synthesize WITHOUT mutating the instrument's default duration
        note = instrument.synth(root_freq, self.fs, duration=duration_s) * float(volume)

        self.events.append({
            't0': start_time_s,
            'wave': note,
        })

    def _total_seconds(self):
        timeline_len_s = self.beats_to_seconds(self.beats_per_measure * self.measures)
        if not self.events:
            return timeline_len_s
        last_event_end = max(e['t0'] + len(e['wave']) / self.fs for e in self.events)
        return max(timeline_len_s, last_event_end)

    def render(self):
        total_s = self._total_seconds()
        out = np.zeros(int(self.fs * total_s), dtype=float)
        for e in self.events:
            t0 = e['t0']
            start = int(t0 * self.fs)
            wave = e['wave']
            end = min(start + len(wave), len(out))
            out[start:end] += wave[:end-start]

        # Final normalization to avoid clipping from overlaps
        peak = np.max(np.abs(out))
        if peak > 1e-9:
            out = 0.98 * out / peak
        return out

    def play(self):
        audio = self.render()
        if audio.size:
            sd.play(audio, self.fs)
            sd.wait()


minorkick_intervals = [
    {'type': 'damped_sine', 'ratio': 1.0, 'amp': 1.0, 'decay': 8},
    {'type': 'damped_sine', 'ratio': 1.095, 'amp': 0.6, 'decay': 10},
    {'type': 'damped_sine', 'ratio': 1.19, 'amp': 0.4, 'decay': 12},
    {'type': 'damped_sine', 'ratio': 1.238, 'amp': 0.3, 'decay': 14},
    {'type': 'damped_sine', 'ratio': 1.287, 'amp': 0.2, 'decay': 16},
    {'type': 'damped_sine', 'ratio': 1.414, 'amp': 0.1, 'decay': 18}  # Optional higher overtone
]

# Create the instrument with intervals
minorkick = Instrument(harmonics=minorkick_intervals, duration=0.5)


tl = Timeline(bpm=170, time_signature=(4,4), measures=2, fs=44100)

for beat in range(0, 4 * tl.beats_per_measure * tl.measures):
# kick drum loop    
    if beat % 4 == 0:
        tl.add_note(minorkick, root_freq=54, start_beat=beat, end_beat=beat+0.5)
        tl.add_note(minorkick, root_freq=54, start_beat=beat+2, end_beat=beat+2.5)
        tl.add_note(minorkick, root_freq=54, start_beat=beat, end_beat=beat+0.5)
        tl.add_note(minorkick, root_freq=54, start_beat=beat+2, end_beat=beat+2.5)
        tl.add_note(minorkick, root_freq=54, start_beat=beat, end_beat=beat+0.5)
        tl.add_note(minorkick, root_freq=54, start_beat=beat+2, end_beat=beat+2.5)
        tl.add_note(minorkick, root_freq=54, start_beat=beat, end_beat=beat+0.5)
        tl.add_note(minorkick, root_freq=54, start_beat=beat+2, end_beat=beat+2.5)
         

tl.play()