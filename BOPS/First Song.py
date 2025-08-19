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



#---------------INSTRUMENTS --------------
if __name__ == "__main__":
    # Define the harmonics from your example
    neon_pluck_harmonics = [
        {'type': 'damped_sine', 'interval': 1, 'mult': 1, 'amp': 1.0, 'decay': 2.5},
        {'type': 'damped_sawtooth', 'interval': 1, 'mult': 2, 'amp': 0.6, 'decay': 2.0},
        {'type': 'sine', 'interval': 1, 'mult': 3, 'amp': 0.3},
        {'type': 'triangle', 'interval': 2 ** (7/12), 'mult': 1, 'amp': 0.4},
        {'type': 'damped_triangle', 'interval': 2 ** (12/12), 'mult': 2, 'amp': 0.3, 'decay': 1.5},
        {'type': 'square', 'interval': 2 ** (3/12), 'mult': 1, 'amp': 0.2},
    ]

    neon_pluck = Instrument(harmonics=neon_pluck_harmonics, duration=2.0)

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

acid_bass_harmonics = [
    {'type': 'sawtooth', 'interval': 1, 'mult': 1, 'amp': 1.0},
    {'type': 'square', 'interval': 1, 'mult': 2, 'amp': 0.5},
    {'type': 'damped_sawtooth', 'interval': 2 ** (7/12), 'mult': 1, 'amp': 0.8, 'decay': 3.0},
    {'type': 'damped_square', 'interval': 2 ** (12/12), 'mult': 2, 'amp': 0.4, 'decay': 2.5},
    {'type': 'sine', 'interval': 2 ** (3/12), 'mult': 3, 'amp': 0.2}
]

# Create the instrument
acid_bass = Instrument(harmonics=acid_bass_harmonics, duration=8.0)

robofap_intervals = [
    # Sub-bass body
    {'type': 'damped_square', 'ratio': 1.00,  'amp': 5.00, 'decay': 5},    # ~29.5 Hz
    {'type': 'damped_sine', 'ratio': 2.00,  'amp': 1.75, 'decay': 6},    # ~59 Hz

    # Mid bass harmonics
    {'type': 'damped_sine', 'ratio': 3.25,  'amp': 0.45, 'decay': 7},    # ~95.9 Hz
    {'type': 'damped_sine', 'ratio': 5.00,  'amp': 0.35, 'decay': 8},   # ~147.5 Hz
    {'type': 'damped_sine', 'ratio': 6.70,  'amp': 0.25, 'decay': 9},   # ~197.6 Hz

    # Higher transient/click content
    {'type': 'damped_triangle', 'ratio': 10.0,  'amp': 0.15, 'decay': 13},   # ~295 Hz
    {'type': 'damped_triangle', 'ratio': 17.0,  'amp': 0.08, 'decay': 14},   # ~501.5 Hz
    {'type': 'damped_triangle', 'ratio': 23.5,  'amp': 0.05, 'decay': 15},   # ~693 Hz
]


# Create the instrument with intervals
robofap = Instrument(harmonics=robofap_intervals, duration=0.5)

mid_growl_harmonics = [
    {'type': 'sawtooth', 'interval': 1, 'mult': 1, 'amp': 1.0},
    {'type': 'square', 'interval': 1, 'mult': 2, 'amp': 0.7},
    {'type': 'damped_sawtooth', 'interval': 2 ** (3/12), 'mult': 1, 'amp': 0.6, 'decay': 2.5},
    {'type': 'damped_square', 'interval': 2 ** (7/12), 'mult': 2, 'amp': 0.4, 'decay': 2.0},
    {'type': 'damped_triangle', 'interval': 2 ** (-12/12), 'mult': 1, 'amp': 0.3, 'decay': 1.8}
]

chromatic_intervals = [2 ** (0 / 12), 2 ** (1 / 12), 2 ** (2 / 12), 2 ** (3 / 12), 2 ** (4 / 12),
                       2 ** (5 / 12), 2 ** (6 / 12), 2 ** (7 / 12), 2 ** (8 / 12), 2 ** (9 / 12), 2 ** (10 / 12), 2 ** (11 / 12)]

# Create the instrument
mid_growl = Instrument(harmonics=mid_growl_harmonics, duration=8.0)

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
maj6synth = Instrument(harmonics=maj6synth_harmonics, duration=8.0)

hardstylekick_intervals = [
    {'type': 'damped_sine', 'ratio': np.float64(1.0), 'amp': np.float32(5.0), 'decay': np.float64(24.4241)},
    {'type': 'damped_square', 'ratio': np.float64(1.0), 'amp': np.float32(0.25), 'decay': np.float64(0.5241)},
    {'type': 'damped_sine', 'ratio': np.float64(1.083), 'amp': np.float32(0.701), 'decay': np.float64(22.4262)},
    {'type': 'damped_sine', 'ratio': np.float64(0.917), 'amp': np.float32(0.474), 'decay': np.float64(7.9986)},
    {'type': 'damped_sine', 'ratio': np.float64(1.167), 'amp': np.float32(0.43), 'decay': np.float64(29.7087)},
    {'type': 'damped_sine', 'ratio': np.float64(1.25), 'amp': np.float32(0.426), 'decay': np.float64(21.653)},
    {'type': 'damped_sine', 'ratio': np.float64(1.333), 'amp': np.float32(0.362), 'decay': np.float64(29.033)},
    {'type': 'damped_sine', 'ratio': np.float64(1.5), 'amp': np.float32(0.345), 'decay': np.float64(10.1983)},
    {'type': 'damped_sine', 'ratio': np.float64(1.583), 'amp': np.float32(0.283), 'decay': np.float64(41.864)},
    {'type': 'damped_sine', 'ratio': np.float64(1.417), 'amp': np.float32(0.281), 'decay': np.float64(40.9591)},
    {'type': 'damped_sine', 'ratio': np.float64(1.667), 'amp': np.float32(0.275), 'decay': np.float64(38.6245)},

    {'type': 'damped_sawtooth', 'ratio': np.float64(0.125), 'amp': np.float32(0.536), 'decay': np.float64(1.0982)},
    {'type': 'damped_square', 'ratio': np.float64(0.25), 'amp': np.float32(1.036), 'decay': np.float64(2.0982)},
    {'type': 'damped_square', 'ratio': np.float64(2), 'amp': np.float32(0.036), 'decay': np.float64(1.0982)},
    {'type': 'damped_sawtooth', 'ratio': np.float64(5.25), 'amp': np.float32(0.036), 'decay': np.float64(3.0982)},
    {'type': 'damped_square', 'ratio': np.float64(8.64), 'amp': np.float32(0.036), 'decay': np.float64(5.0982)},
    {'type': 'damped_triangle', 'ratio': np.float64(1.5), 'amp': np.float32(0.036), 'decay': np.float64(2.0982)},
    {'type': 'damped_sawtooth', 'ratio': np.float64(4.5), 'amp': np.float32(0.036), 'decay': np.float64(19.0982)},

    {'type': 'damped_sawtooth', 'ratio': np.float64(10), 'amp': np.float32(0.036), 'decay': np.float64(1.0982)},
    {'type': 'damped_sawtooth', 'ratio': np.float64(11), 'amp': np.float32(0.036), 'decay': np.float64(1.0982)},
    {'type': 'damped_sawtooth', 'ratio': np.float64(15), 'amp': np.float32(0.036), 'decay': np.float64(1.0982)},

    {'type': 'damped_sine', 'ratio': np.float64(2.5), 'amp': np.float32(0.199), 'decay': np.float64(39.6052)},
    {'type': 'damped_sine', 'ratio': np.float64(2.583), 'amp': np.float32(0.189), 'decay': np.float64(41.2948)},
    {'type': 'damped_sine', 'ratio': np.float64(2.417), 'amp': np.float32(0.188), 'decay': np.float64(42.8798)},
    {'type': 'damped_sine', 'ratio': np.float64(3.333), 'amp': np.float32(0.177), 'decay': np.float64(38.0787)},
    {'type': 'damped_sine', 'ratio': np.float64(2.833), 'amp': np.float32(0.176), 'decay': np.float64(40.9083)},
    {'type': 'damped_sine', 'ratio': np.float64(3.583), 'amp': np.float32(0.17), 'decay': np.float64(41.9336)},
    {'type': 'damped_sine', 'ratio': np.float64(3.25), 'amp': np.float32(0.168), 'decay': np.float64(41.1906)},
    {'type': 'damped_sine', 'ratio': np.float64(2.333), 'amp': np.float32(0.167), 'decay': np.float64(40.8501)},
    {'type': 'damped_sine', 'ratio': np.float64(3.5), 'amp': np.float32(0.166), 'decay': np.float64(24.1894)},
    {'type': 'damped_sine', 'ratio': np.float64(3.417), 'amp': np.float32(0.166), 'decay': np.float64(41.3995)},
]

# Create the instrument with intervals
hardstylekick = Instrument(harmonics=hardstylekick_intervals, duration=0.5)

scikick_intervals = [
    {'type': 'damped_sine', 'ratio': np.float64(1.0), 'amp': np.float32(1.0), 'decay': np.float64(1.878)},
    {'type': 'damped_sine', 'ratio': np.float64(0.989), 'amp': np.float32(0.956), 'decay': np.float64(1.881)},
    {'type': 'damped_sine', 'ratio': np.float64(1.011), 'amp': np.float32(0.937), 'decay': np.float64(1.879)},
    {'type': 'damped_sine', 'ratio': np.float64(0.978), 'amp': np.float32(0.813), 'decay': np.float64(1.891)},
    {'type': 'damped_sine', 'ratio': np.float64(1.022), 'amp': np.float32(0.793), 'decay': np.float64(1.882)},
    {'type': 'damped_sine', 'ratio': np.float64(1.033), 'amp': np.float32(0.625), 'decay': np.float64(1.891)},
    {'type': 'damped_sine', 'ratio': np.float64(0.967), 'amp': np.float32(0.608), 'decay': np.float64(1.906)},
    {'type': 'damped_sine', 'ratio': np.float64(1.044), 'amp': np.float32(0.497), 'decay': np.float64(1.91)},
    {'type': 'damped_sine', 'ratio': np.float64(1.055), 'amp': np.float32(0.444), 'decay': np.float64(1.936)},
    {'type': 'damped_sine', 'ratio': np.float64(1.066), 'amp': np.float32(0.434), 'decay': np.float64(1.962)},


# subpeaks
    {'type': 'damped_sine', 'ratio': np.float64(1.396), 'amp': np.float32(0.197), 'decay': np.float64(1.294)},
    {'type': 'damped_sine', 'ratio': np.float64(1.407), 'amp': np.float32(0.194), 'decay': np.float64(1.293)},
    {'type': 'damped_sine', 'ratio': np.float64(1.429), 'amp': np.float32(0.194), 'decay': np.float64(1.274)},
    {'type': 'damped_sine', 'ratio': np.float64(1.549), 'amp': np.float32(0.193), 'decay': np.float64(1.216)},
    {'type': 'damped_sine', 'ratio': np.float64(1.418), 'amp': np.float32(0.193), 'decay': np.float64(1.286)},
    {'type': 'damped_sine', 'ratio': np.float64(1.495), 'amp': np.float32(0.193), 'decay': np.float64(1.236)},
    {'type': 'damped_sine', 'ratio': np.float64(1.44), 'amp': np.float32(0.193), 'decay': np.float64(1.264)},
    {'type': 'damped_sine', 'ratio': np.float64(1.538), 'amp': np.float32(0.193), 'decay': np.float64(1.223)},
    {'type': 'damped_sine', 'ratio': np.float64(1.56), 'amp': np.float32(0.193), 'decay': np.float64(1.212)},
    {'type': 'damped_sine', 'ratio': np.float64(1.505), 'amp': np.float32(0.193), 'decay': np.float64(1.231)},
]

scikick = Instrument(harmonics=scikick_intervals, duration=0.5)

reverse_cymbal_intervals = [
    {'type': 'white_noise', 'amp': 1.0},
    {'type': 'pink_noise', 'amp': 0.7}
]

# Create instrument
reverse_cymbal = Instrument(harmonics=reverse_cymbal_intervals, duration=0.6)

snare3_intervals = [
    {'type': 'damped_sine', 'ratio': 1.0, 'amp': 0.8, 'decay': 7},
    {'type': 'damped_sawtooth', 'ratio': 2.0, 'amp': 0.5, 'decay': 9},
    {'type': 'damped_triangle', 'ratio': 2.5, 'amp': 0.3, 'decay': 11},
    {'type': 'damped_square', 'ratio': 3.5, 'amp': 0.2, 'decay': 13},
    {'type': 'white_noise', 'ratio': 1.0, 'amp': 0.6, 'decay': 12},
    {'type': 'pink_noise', 'ratio': 1.0, 'amp': 0.7, 'decay': 10}
]

# Create snare instrument
snare3 = Instrument(harmonics=snare3_intervals, duration=0.5)

hihat_intervals = [
    {'type': 'white_noise', 'amp': 0.9, 'decay': 15},
    {'type': 'pink_noise', 'amp': 0.6, 'decay': 12}
]

# Create hi-hat instrument
hihat = Instrument(harmonics=hihat_intervals, duration=0.15)

greenlight_intervals = [
    # Sub-bass body
    {'type': 'sine', 'ratio': 1, 'amp': 13},
    {'type': 'damped_square', 'ratio': 1.00,  'amp': 15.00, 'decay': 7},    
    {'type': 'damped_sine', 'ratio': 2.00,  'amp': 11.75, 'decay': 8},    

    # Mid bass harmonics
    {'type': 'damped_sine', 'ratio': 3.25,  'amp': 0.045, 'decay': 9},    
    {'type': 'damped_sine', 'ratio': 5.00,  'amp': 0.035, 'decay': 10},   
    {'type': 'damped_sine', 'ratio': 6.70,  'amp': 0.025, 'decay': 11},   

    # Higher transient/click content
    {'type': 'damped_triangle', 'ratio': 10.0,  'amp': 0.15, 'decay': 13},   
    {'type': 'damped_triangle', 'ratio': 17.0,  'amp': 0.08, 'decay': 14},   
    {'type': 'damped_triangle', 'ratio': 23.5,  'amp': 0.05, 'decay': 15},   
]


# Create the instrument with intervals
greenlight = Instrument(harmonics=greenlight_intervals, duration=0.5)



# Create the instrument with intervals


#---------------START OF TIMELINE---------------



tl = Timeline(bpm=120, time_signature=(4,4), measures=25, fs=44100)

# kick loop
for beat in range(1, tl.time_signature[0] * tl.measures + 1):
    tl.add_note(minorkick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    tl.add_note(scikick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    tl.add_note(minorkick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    tl.add_note(scikick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    tl.add_note(minorkick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    tl.add_note(scikick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    tl.add_note(minorkick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    tl.add_note(scikick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    tl.add_note(minorkick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    tl.add_note(scikick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    tl.add_note(minorkick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    tl.add_note(scikick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    tl.add_note(minorkick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    tl.add_note(scikick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    tl.add_note(minorkick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    tl.add_note(scikick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    tl.add_note(minorkick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    tl.add_note(scikick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    tl.add_note(minorkick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    tl.add_note(scikick, root_freq=48.109, start_beat=beat-1, end_beat=beat)
    if beat % 2 == 0:
        tl.add_note(snare3, root_freq=48.109, start_beat=beat-1, end_beat=beat-0.5)
        tl.add_note(snare3, root_freq=48.109, start_beat=beat-1, end_beat=beat-0.5)
        tl.add_note(snare3, root_freq=48.109, start_beat=beat-1, end_beat=beat-0.5)
        tl.add_note(snare3, root_freq=48.109, start_beat=beat-1, end_beat=beat-0.5)
        tl.add_note(snare3, root_freq=48.109, start_beat=beat-1, end_beat=beat-0.5)
        tl.add_note(snare3, root_freq=48.109, start_beat=beat-1, end_beat=beat-0.5)
        tl.add_note(snare3, root_freq=48.109, start_beat=beat-1, end_beat=beat-0.5)
        tl.add_note(snare3, root_freq=48.109, start_beat=beat-1, end_beat=beat-0.5)
    #tl.add_note(hihat, root_freq=48.109, start_beat=beat-0.75, end_beat=beat-0.625)
    tl.add_note(hihat, root_freq=48.109, start_beat=beat-0.5, end_beat=beat-0.375)
    #tl.add_note(hihat, root_freq=48.109, start_beat=beat-0.25, end_beat=beat-0.125)

#=======================================================
# first chord Gm
for beat in range(1, tl.time_signature[0] * 2 + 1):
    tl.add_note(acid_bass, root_freq=96.217, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(acid_bass, root_freq=114.422, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(acid_bass, root_freq=144.163, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(acid_bass, root_freq=171.439, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(acid_bass, root_freq=96.217, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(acid_bass, root_freq=114.422, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(acid_bass, root_freq=144.163, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(acid_bass, root_freq=171.439, start_beat=beat-0.25, end_beat=beat)

# second chord F
for beat in range(tl.time_signature[0] * 2 + 1, tl.time_signature[0] * 4 + 1):
    tl.add_note(acid_bass, root_freq=85.72, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(acid_bass, root_freq=108.0, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(acid_bass, root_freq=128.434, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(acid_bass, root_freq=144.163, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(acid_bass, root_freq=85.72, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(acid_bass, root_freq=108.0, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(acid_bass, root_freq=128.434, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(acid_bass, root_freq=144.163, start_beat=beat-0.25, end_beat=beat)

# third chord Cm
for beat in range(tl.time_signature[0] * 4 + 1, tl.time_signature[0] * 6 + 1):
    tl.add_note(acid_bass, root_freq=128.434, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(acid_bass, root_freq=76.368, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(acid_bass, root_freq=96.217, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(acid_bass, root_freq=114.422, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(acid_bass, root_freq=128.434, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(acid_bass, root_freq=76.368, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(acid_bass, root_freq=96.217, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(acid_bass, root_freq=114.422, start_beat=beat-0.25, end_beat=beat)

# fourth chord Dm
for beat in range(tl.time_signature[0] * 6 + 1, tl.time_signature[0] * 7 + 1):
    tl.add_note(acid_bass, root_freq=128.434, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(acid_bass, root_freq=72.081, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(acid_bass, root_freq=85.72 , start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(acid_bass, root_freq=108.0, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(acid_bass, root_freq=128.434, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(acid_bass, root_freq=72.081, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(acid_bass, root_freq=85.72 , start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(acid_bass, root_freq=108.0, start_beat=beat-0.25, end_beat=beat)

# fifth chord D
for beat in range(tl.time_signature[0] * 7 + 1, tl.time_signature[0] * 8 + 1):
    tl.add_note(acid_bass, root_freq=128.434, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(acid_bass, root_freq=72.081, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(acid_bass, root_freq=90.817 , start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(acid_bass, root_freq=108.0, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(acid_bass, root_freq=128.434, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(acid_bass, root_freq=72.081, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(acid_bass, root_freq=90.817 , start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(acid_bass, root_freq=108.0, start_beat=beat-0.25, end_beat=beat)
#first repeat of the chord progression

#first chord Gm w harmony
for beat in range(tl.time_signature[0] * 8 + 1, tl.time_signature[0] * 10 + 1):
    tl.add_note(acid_bass, root_freq=96.217, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=228.844, start_beat=beat-1, end_beat=beat-0.75)  # harmony 10th
    tl.add_note(acid_bass, root_freq=114.422, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=288.325, start_beat=beat-0.75, end_beat=beat-0.5)  # harmony 10th
    tl.add_note(acid_bass, root_freq=228.844, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=576.651, start_beat=beat-0.5, end_beat=beat-0.25)  # harmony 10th
    tl.add_note(acid_bass, root_freq=288.325, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(maj6synth, root_freq=685.757, start_beat=beat-0.25, end_beat=beat)  # harmony 10th
    tl.add_note(acid_bass, root_freq=96.217, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(acid_bass, root_freq=114.422, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(acid_bass, root_freq=228.844, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(acid_bass, root_freq=288.325, start_beat=beat-0.25, end_beat=beat)

    tl.add_note(greenlight, root_freq=384.868, start_beat=beat-1, end_beat=beat-(2/3))
    tl.add_note(greenlight, root_freq=457.688, start_beat=beat-(2/3), end_beat=beat-(1/3))
    tl.add_note(greenlight, root_freq=576.651, start_beat=beat-(1/3), end_beat=beat)  # harmony 10th
# second chord F
for beat in range(tl.time_signature[0] * 10 + 1, tl.time_signature[0] * 12 + 1):
    tl.add_note(acid_bass, root_freq=85.72, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=216.0, start_beat=beat-1, end_beat=beat-0.75)  # harmony 10th
    tl.add_note(acid_bass, root_freq=108.0, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=256.869, start_beat=beat-0.75, end_beat=beat-0.5)  # harmony 10th
    tl.add_note(acid_bass, root_freq=128.434, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=305.47, start_beat=beat-0.5, end_beat=beat-0.25)  # harmony 10th
    tl.add_note(acid_bass, root_freq=144.163, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(maj6synth, root_freq=342.879, start_beat=beat-0.25, end_beat=beat)  # harmony 10th
    tl.add_note(acid_bass, root_freq=85.72, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(acid_bass, root_freq=108.0, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(acid_bass, root_freq=128.434, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(acid_bass, root_freq=144.163, start_beat=beat-0.25, end_beat=beat)

    tl.add_note(greenlight, root_freq=342.879, start_beat=beat-1, end_beat=beat-(2/3))
    tl.add_note(greenlight, root_freq=432.0, start_beat=beat-(2/3), end_beat=beat-(1/3))
    tl.add_note(greenlight, root_freq=513.737, start_beat=beat-(1/3), end_beat=beat)  # harmony 10th

# third chord Cm
for beat in range(tl.time_signature[0] * 12 + 1, tl.time_signature[0] * 14 + 1):
    tl.add_note(acid_bass, root_freq=128.434, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=305.47, start_beat=beat-1, end_beat=beat-0.75)  # harmony 10th
    tl.add_note(acid_bass, root_freq=76.368, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=192.434, start_beat=beat-0.75, end_beat=beat-0.5)  # harmony 10th
    tl.add_note(acid_bass, root_freq=96.217, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=228.844, start_beat=beat-0.5, end_beat=beat-0.25)  # harmony 10th
    tl.add_note(acid_bass, root_freq=114.422, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(maj6synth, root_freq=288.325, start_beat=beat-0.25, end_beat=beat)  # harmony 10th
    tl.add_note(acid_bass, root_freq=128.434, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(acid_bass, root_freq=76.368, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(acid_bass, root_freq=96.217, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(acid_bass, root_freq=114.422, start_beat=beat-0.25, end_beat=beat)

    tl.add_note(greenlight, root_freq=384.868, start_beat=beat-1, end_beat=beat-(2/3))
    tl.add_note(greenlight, root_freq=513.737, start_beat=beat-(2/3), end_beat=beat-(1/3))
    tl.add_note(greenlight, root_freq=610.94, start_beat=beat-(1/3), end_beat=beat)  # harmony 10th

# fourth chord Dm
for beat in range(tl.time_signature[0] * 14 + 1, tl.time_signature[0] * 15 + 1):
    tl.add_note(acid_bass, root_freq=128.434, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=305.47, start_beat=beat-1, end_beat=beat-0.75)  # harmony 10th
    tl.add_note(acid_bass, root_freq=72.081, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=171.439, start_beat=beat-0.75, end_beat=beat-0.5)  # harmony 10th
    tl.add_note(acid_bass, root_freq=85.72 , start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=216.0, start_beat=beat-0.5, end_beat=beat-0.25)  # harmony 10th
    tl.add_note(acid_bass, root_freq=108.0, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(maj6synth, root_freq=162.0, start_beat=beat-0.25, end_beat=beat)  # harmony 10th
    tl.add_note(acid_bass, root_freq=128.434, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(acid_bass, root_freq=72.081, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(acid_bass, root_freq=85.72 , start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(acid_bass, root_freq=108.0, start_beat=beat-0.25, end_beat=beat)

    tl.add_note(greenlight, root_freq=432.0, start_beat=beat-1, end_beat=beat-(2/3))
    tl.add_note(greenlight, root_freq=576.651, start_beat=beat-(2/3), end_beat=beat-(1/3))
    tl.add_note(greenlight, root_freq=685.757, start_beat=beat-(1/3), end_beat=beat)  # harmony 10th

# fifth chord D (with D# for harmonic minor)
for beat in range(tl.time_signature[0] * 15 + 1, tl.time_signature[0] * 16 + 1):
    tl.add_note(acid_bass, root_freq=128.434, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=305.47, start_beat=beat-1, end_beat=beat-0.75)  # harmony 10th
    tl.add_note(acid_bass, root_freq=72.081, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=181.634, start_beat=beat-0.75, end_beat=beat-0.5)  # harmony 10th
    tl.add_note(acid_bass, root_freq=90.817 , start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=216.0 , start_beat=beat-0.5, end_beat=beat-0.25)  # harmony 10th
    tl.add_note(acid_bass, root_freq=108.0, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(maj6synth, root_freq=162.0, start_beat=beat-0.25, end_beat=beat)  # harmony 10th
    tl.add_note(acid_bass, root_freq=128.434, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(acid_bass, root_freq=72.081, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(acid_bass, root_freq=90.817 , start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(acid_bass, root_freq=108.0, start_beat=beat-0.25, end_beat=beat)

    tl.add_note(greenlight, root_freq=432.0, start_beat=beat-1, end_beat=beat-(2/3))
    tl.add_note(greenlight, root_freq=576.651, start_beat=beat-(2/3), end_beat=beat-(1/3))
    tl.add_note(greenlight, root_freq=610.94, start_beat=beat-(1/3), end_beat=beat)  # harmony 10th


#========================================================

#------------------
#measure 17
for beat in range(tl.time_signature[0] * 16 + 1, tl.time_signature[0] * 17 + 1):
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*18, end_beat=4*18+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*18+4, end_beat=4*18+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*18, end_beat=4*18+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*18+4, end_beat=4*18+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*18, end_beat=4*18+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*18+4, end_beat=4*18+5)
    
    tl.add_note(robofap, root_freq=48.109, start_beat=4*18+1, end_beat=4*18+1+0.25)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*18+1+0.5, end_beat=4*18+1+0.75)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*18+2, end_beat=4*18+2+0.5)
    tl.add_note(robofap, root_freq=72.081, start_beat=4*18+4, end_beat=4*18+4+0.25)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*18+4.5, end_beat=4*18+4.75)

    tl.add_note(mid_growl, root_freq=48.109, start_beat=4*18+1, end_beat=4*18+1+0.25)
    tl.add_note(mid_growl, root_freq=48.109, start_beat=4*18+1+0.5, end_beat=4*18+1+0.75)
    tl.add_note(hardstylekick, root_freq=48.109, start_beat=4*18+2, end_beat=4*18+2+0.5)
    tl.add_note(mid_growl, root_freq=72.081, start_beat=4*18+4, end_beat=4*18+4+0.25)
    tl.add_note(mid_growl, root_freq=48.109, start_beat=4*18+4.5, end_beat=4*18+4.75)


    tl.add_note(acid_bass, root_freq=576.651, start_beat=4*18+3.5, end_beat=4*18+3.75)
    tl.add_note(acid_bass, root_freq=1153.302, start_beat=4*18+3.5, end_beat=4*18+3.75)
    tl.add_note(acid_bass, root_freq=915.376, start_beat=4*18+3.5, end_beat=4*18+3.75)
    
# measure 18
for beat in range(tl.time_signature[0] * 17 + 1, tl.time_signature[0] * 18 + 1):
    tl.add_note(maj6synth, root_freq=85.72, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=108.0, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=128.434, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=144.163, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(maj6synth, root_freq=85.72, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=108.0, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=128.434, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=144.163, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(maj6synth, root_freq=85.72, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=108.0, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=128.434, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=144.163, start_beat=beat-0.25, end_beat=beat)


    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*19, end_beat=4*19+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*19+4, end_beat=4*19+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*19, end_beat=4*19+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*19+4, end_beat=4*19+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*19, end_beat=4*19+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*19+4, end_beat=4*19+5)

    tl.add_note(robofap, root_freq=48.109, start_beat=4*19, end_beat=4*19+0.25)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*19+0.5, end_beat=4*19+0.75)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*19+1, end_beat=4*19+1+0.5)
    tl.add_note(robofap, root_freq=72.081, start_beat=4*19+3, end_beat=4*19+3+0.25)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*19+3.5, end_beat=4*19+3.75)

    tl.add_note(hardstylekick, root_freq=48.109, start_beat=4*19, end_beat=4*19+0.25)
    tl.add_note(hardstylekick, root_freq=48.109, start_beat=4*19+0.5, end_beat=4*19+0.75)
    tl.add_note(hardstylekick, root_freq=48.109, start_beat=4*19+1, end_beat=4*19+1+0.5)
    tl.add_note(hardstylekick, root_freq=72.081, start_beat=4*19+3, end_beat=4*19+3+0.25)
    tl.add_note(mid_growl, root_freq=48.109, start_beat=4*19+3.5, end_beat=4*19+3.75)

#meausure 19
for beat in range(tl.time_signature[0] * 18 + 1, tl.time_signature[0] * 19 + 1):
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*20, end_beat=4*20+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*20+4, end_beat=4*20+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*20, end_beat=4*20+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*20+4, end_beat=4*20+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*20, end_beat=4*20+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*20+4, end_beat=4*20+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*20, end_beat=4*20+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*20+4, end_beat=4*20+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*20, end_beat=4*20+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*20+4, end_beat=4*20+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*20, end_beat=4*20+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*20+4, end_beat=4*20+5)
    
    tl.add_note(robofap, root_freq=48.109, start_beat=4*20+1, end_beat=4*20+1+0.25)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*20+1+0.5, end_beat=4*20+1+0.75)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*20+2, end_beat=4*20+2+0.5)
    tl.add_note(robofap, root_freq=72.081, start_beat=4*20+4, end_beat=4*20+4+0.25)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*20+4.5, end_beat=4*20+4.75)

    tl.add_note(mid_growl, root_freq=48.109, start_beat=4*20+1, end_beat=4*20+1+0.25)
    tl.add_note(mid_growl, root_freq=48.109, start_beat=4*20+1+0.5, end_beat=4*20+1+0.75)
    tl.add_note(hardstylekick, root_freq=48.109, start_beat=4*20+2, end_beat=4*20+2+0.5)
    tl.add_note(mid_growl, root_freq=72.081, start_beat=4*20+4, end_beat=4*20+4+0.25)
    tl.add_note(mid_growl, root_freq=48.109, start_beat=4*20+4.5, end_beat=4*20+4.75)


    tl.add_note(acid_bass, root_freq=576.651*4, start_beat=4*20+3.5, end_beat=4*20+3.75)
    tl.add_note(acid_bass, root_freq=1153.302*4, start_beat=4*20+3.5, end_beat=4*20+3.75)
    tl.add_note(acid_bass, root_freq=915.376*4, start_beat=4*20+3.5, end_beat=4*20+3.75)

# measure 20
for beat in range(tl.time_signature[0] * 19 + 1, tl.time_signature[0] * 20 + 1):
    tl.add_note(maj6synth, root_freq=144.163, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=128.434, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=108, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=85.72, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(maj6synth, root_freq=144.163, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=128.434, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=108, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=85.72, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(maj6synth, root_freq=144.163, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=128.434, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=108, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=85.72, start_beat=beat-0.25, end_beat=beat)


    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*21, end_beat=4*21+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*21+4, end_beat=4*21+4.75)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*21, end_beat=4*21+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*21+4, end_beat=4*21+4.75)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*21, end_beat=4*21+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*21+4, end_beat=4*21+4.75)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*21, end_beat=4*21+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*21+4, end_beat=4*21+4.75)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*21, end_beat=4*21+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*21+4, end_beat=4*21+4.75)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*21, end_beat=4*21+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*21+4, end_beat=4*21+4.75)

    tl.add_note(robofap, root_freq=48.109, start_beat=4*21, end_beat=4*21+0.25)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*21+0.5, end_beat=4*21+0.75)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*21+1, end_beat=4*21+1+0.5)
    tl.add_note(robofap, root_freq=72.081, start_beat=4*21+3, end_beat=4*21+3+0.25)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*21+3.5, end_beat=4*21+3.75)

    tl.add_note(hardstylekick, root_freq=48.109, start_beat=4*21, end_beat=4*21+0.25)
    tl.add_note(hardstylekick, root_freq=48.109, start_beat=4*21+0.5, end_beat=4*21+0.75)
    tl.add_note(hardstylekick, root_freq=48.109, start_beat=4*21+1, end_beat=4*21+1+0.5)
    tl.add_note(hardstylekick, root_freq=72.081, start_beat=4*21+3, end_beat=4*21+3+0.25)
    tl.add_note(mid_growl, root_freq=48.109, start_beat=4*21+3.5, end_beat=4*21+3.75)


    tl.add_note(acid_bass, root_freq=685.757, start_beat=4*21+3, end_beat=4*21+3.75)
    tl.add_note(acid_bass, root_freq=864.0, start_beat=4*21+3, end_beat=4*21+3.75)
    tl.add_note(acid_bass, root_freq=1371.515, start_beat=4*21+3, end_beat=4*21+3.75)

    tl.add_note(acid_bass, root_freq=576.651, start_beat=4*21+4, end_beat=4*21+4.5)
    tl.add_note(acid_bass, root_freq=1153.302, start_beat=4*21+4, end_beat=4*21+4.5)
    tl.add_note(acid_bass, root_freq=915.376, start_beat=4*21+4, end_beat=4*21+4.5)

# measure 21
for beat in range(tl.time_signature[0] * 20 + 1, tl.time_signature[0] * 21 + 1):
    tl.add_note(maj6synth, root_freq=85.72, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=108.0, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=128.434, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=144.163, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(maj6synth, root_freq=85.72, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=108.0, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=128.434, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=144.163, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(maj6synth, root_freq=85.72, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=108.0, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=128.434, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=144.163, start_beat=beat-0.25, end_beat=beat)

    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*22, end_beat=4*22+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*22+4, end_beat=4*22+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*22, end_beat=4*22+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*22+4, end_beat=4*22+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*22, end_beat=4*22+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*22+4, end_beat=4*22+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*22, end_beat=4*22+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*22+4, end_beat=4*22+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*22, end_beat=4*22+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*22+4, end_beat=4*22+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*22, end_beat=4*22+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*22+4, end_beat=4*22+5)

    tl.add_note(robofap, root_freq=48.109, start_beat=4*22+1, end_beat=4*22+1+0.25)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*22+1+0.5, end_beat=4*22+1+0.75)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*22+2, end_beat=4*22+2+0.5)
    tl.add_note(robofap, root_freq=72.081, start_beat=4*22+4, end_beat=4*22+4+0.25)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*22+4.5, end_beat=4*22+4.75)

    tl.add_note(mid_growl, root_freq=48.109, start_beat=4*22+1, end_beat=4*22+1+0.25)
    tl.add_note(mid_growl, root_freq=48.109, start_beat=4*22+1+0.5, end_beat=4*22+1+0.75)
    tl.add_note(mid_growl, root_freq=48.109, start_beat=4*22+2, end_beat=4*22+2+0.5)
    tl.add_note(mid_growl, root_freq=72.081, start_beat=4*22+4, end_beat=4*22+4+0.25)
    tl.add_note(hardstylekick, root_freq=48.109, start_beat=4*22+4.5, end_beat=4*22+4.75)


    tl.add_note(acid_bass, root_freq=576.651, start_beat=4*22+3.5, end_beat=4*22+3.75)
    tl.add_note(acid_bass, root_freq=1153.302, start_beat=4*22+3.5, end_beat=4*22+3.75)
    tl.add_note(acid_bass, root_freq=915.376, start_beat=4*22+3.5, end_beat=4*22+3.75)

# measure 22
for beat in range(tl.time_signature[0] * 21 + 1, tl.time_signature[0] * 22 + 1):
    tl.add_note(maj6synth, root_freq=85.72, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=108.0, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=128.434, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=144.163, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(maj6synth, root_freq=85.72, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=108.0, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=128.434, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=144.163, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(maj6synth, root_freq=85.72, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=108.0, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=128.434, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=144.163, start_beat=beat-0.25, end_beat=beat)

    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*23, end_beat=4*23+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*23+4, end_beat=4*23+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*23, end_beat=4*23+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*23+4, end_beat=4*23+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*23, end_beat=4*23+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*23+4, end_beat=4*23+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*23, end_beat=4*23+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*23+4, end_beat=4*23+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*23, end_beat=4*23+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*23+4, end_beat=4*23+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*23, end_beat=4*23+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*23+4, end_beat=4*23+5)

    tl.add_note(robofap, root_freq=48.109, start_beat=4*23, end_beat=4*23+0.25)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*23+0.5, end_beat=4*23+0.75)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*23+1, end_beat=4*23+1+0.5)
    tl.add_note(robofap, root_freq=72.081, start_beat=4*23+3, end_beat=4*23+3+0.25)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*23+3.5, end_beat=4*23+3.75)

    tl.add_note(hardstylekick, root_freq=48.109, start_beat=4*23, end_beat=4*23+0.25)
    tl.add_note(hardstylekick, root_freq=48.109, start_beat=4*23+0.5, end_beat=4*23+0.75)
    tl.add_note(hardstylekick, root_freq=48.109, start_beat=4*23+1, end_beat=4*23+1+0.5)
    tl.add_note(hardstylekick, root_freq=72.081, start_beat=4*23+3, end_beat=4*23+3+0.25)
    tl.add_note(mid_growl, root_freq=48.109, start_beat=4*23+3.5, end_beat=4*23+3.75)


# measure 24
for beat in range(tl.time_signature[0] * 23 + 1, tl.time_signature[0] * 24 + 1):
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*24, end_beat=4*24+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*24+4, end_beat=4*24+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*24, end_beat=4*24+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*24+4, end_beat=4*24+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*24, end_beat=4*24+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*24+4, end_beat=4*24+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*24, end_beat=4*24+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*24+4, end_beat=4*24+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*24, end_beat=4*24+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*24+4, end_beat=4*24+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*24, end_beat=4*24+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*24+4, end_beat=4*24+5)

    tl.add_note(robofap, root_freq=48.109, start_beat=4*24+1, end_beat=4*24+1+0.25)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*24+1+0.5, end_beat=4*24+1+0.75)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*24+2, end_beat=4*24+2+0.5)
    tl.add_note(robofap, root_freq=72.081, start_beat=4*24+4, end_beat=4*24+4+0.25)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*24+4.5, end_beat=4*24+4.75)

    tl.add_note(mid_growl, root_freq=48.109, start_beat=4*24+1, end_beat=4*24+1+0.25)
    tl.add_note(mid_growl, root_freq=48.109, start_beat=4*24+1+0.5, end_beat=4*24+1+0.75)
    tl.add_note(mid_growl, root_freq=48.109, start_beat=4*24+2, end_beat=4*24+2+0.5)
    tl.add_note(mid_growl, root_freq=72.081, start_beat=4*24+4, end_beat=4*24+4+0.25)
    tl.add_note(mid_growl, root_freq=48.109, start_beat=4*24+4.5, end_beat=4*24+4.75)


    tl.add_note(acid_bass, root_freq=576.651*4, start_beat=4*24+3.5, end_beat=4*24+3.75)
    tl.add_note(acid_bass, root_freq=1153.302*4, start_beat=4*24+3.5, end_beat=4*24+3.75)
    tl.add_note(acid_bass, root_freq=915.376*4, start_beat=4*24+3.5, end_beat=4*24+3.75)

# measure 25
for beat in range(tl.time_signature[0] * 24 + 1, tl.time_signature[0] * 25 + 1):
    tl.add_note(maj6synth, root_freq=144.163, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=128.434, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=108, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=85.72, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(maj6synth, root_freq=144.163, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=128.434, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=108, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=85.72, start_beat=beat-0.25, end_beat=beat)
    tl.add_note(maj6synth, root_freq=144.163, start_beat=beat-1, end_beat=beat-0.75)
    tl.add_note(maj6synth, root_freq=128.434, start_beat=beat-0.75, end_beat=beat-0.5)
    tl.add_note(maj6synth, root_freq=108, start_beat=beat-0.5, end_beat=beat-0.25)
    tl.add_note(maj6synth, root_freq=85.72, start_beat=beat-0.25, end_beat=beat)

    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*25, end_beat=4*25+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*25+4, end_beat=4*25+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*25, end_beat=4*25+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*25+4, end_beat=4*25+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*25, end_beat=4*25+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*25+4, end_beat=4*25+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*25, end_beat=4*25+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*25+4, end_beat=4*25+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*25, end_beat=4*25+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*25+4, end_beat=4*25+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*25, end_beat=4*25+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*25+4, end_beat=4*25+5)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*25, end_beat=4*25+1)
    tl.add_note(maj6synth, root_freq=48.109, start_beat=4*25+4, end_beat=4*25+5)

    tl.add_note(robofap, root_freq=48.109, start_beat=4*25, end_beat=4*25+0.25)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*25+0.5, end_beat=4*25+0.75)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*25+1, end_beat=4*25+1+0.5)
    tl.add_note(robofap, root_freq=72.081, start_beat=4*25+3, end_beat=4*25+3+0.25)
    tl.add_note(robofap, root_freq=48.109, start_beat=4*25+3.5, end_beat=4*25+3.75)

    tl.add_note(hardstylekick, root_freq=48.109, start_beat=4*25, end_beat=4*25+0.25)
    tl.add_note(hardstylekick, root_freq=48.109, start_beat=4*25+0.5, end_beat=4*25+0.75)
    tl.add_note(hardstylekick, root_freq=48.109, start_beat=4*25+1, end_beat=4*25+1+0.5)
    tl.add_note(hardstylekick, root_freq=72.081, start_beat=4*25+3, end_beat=4*25+3+0.25)
    tl.add_note(mid_growl, root_freq=48.109, start_beat=4*25+3.5, end_beat=4*25+3.75)


    tl.add_note(acid_bass, root_freq=685.757, start_beat=4*25+3, end_beat=4*25+3.75)
    tl.add_note(acid_bass, root_freq=864.0, start_beat=4*25+3, end_beat=4*25+3.75)
    tl.add_note(acid_bass, root_freq=1371.515, start_beat=4*25+3, end_beat=4*25+3.75)

    tl.add_note(acid_bass, root_freq=576.651, start_beat=4*25+4, end_beat=4*25+4.75)
    tl.add_note(acid_bass, root_freq=1153.302, start_beat=4*25+4, end_beat=4*25+4.75)
    tl.add_note(acid_bass, root_freq=915.376, start_beat=4*25+4, end_beat=4*25+4.75)



tl.play()



# for beat in range(tl.time_signature[0] * 7 + 1 + loop*8, tl.time_signature[0] * 8 + 1 + loop*8):
#       tl.add_note(acid_bass, root_freq=128.434, start_beat=beat-1, end_beat=beat-0.75)
#        tl.add_note(acid_bass, root_freq=72.081, start_beat=beat-0.75, end_beat=beat-0.5)
#        tl.add_note(acid_bass, root_freq=90.817 , start_beat=beat-0.5, end_beat=beat-0.25)
#        tl.add_note(acid_bass, root_freq=108.0, start_beat=beat-0.25, end_beat=beat)
