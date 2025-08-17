import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square, sawtooth
import sounddevice as sd

# --- Wave generators ---
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

# --- Instrument class ---
class Instrument:
    def __init__(self, harmonics, duration=4.0):
        self.harmonics = harmonics
        self.duration = duration

    def synth(self, root_freq, fs, duration=None):
        dur = duration if duration is not None else self.duration
        t = np.linspace(0, dur, int(fs * dur), endpoint=False)
        signal = np.zeros_like(t)
        for h in self.harmonics:
            func = wave_funcs[h['type']]
            freq = root_freq * h['interval'] * h['mult']
            amp = h['amp']
            phase = h.get('phase', 0)
            if 'decay' in h:
                signal += func(t, freq, amp, h['decay'], phase)
            else:
                signal += func(t, freq, amp, phase)
        return signal, t

# --- Timeline class with measure & beat positioning ---
class Timeline:
    def __init__(self, bpm, time_signature=(4,4), measures=4, fs=44100):
        self.bpm = bpm
        self.beats_per_measure = time_signature[0]
        self.note_value = time_signature[1]
        self.fs = fs
        self.seconds_per_beat = 60.0 / bpm
        self.length_seconds = measures * self.beats_per_measure * self.seconds_per_beat
        self.timeline = np.zeros(int(self.fs * self.length_seconds))

    def beat_to_seconds(self, measure, beat, beat_fraction=0.0):
        total_beats = (measure - 1) * self.beats_per_measure + (beat - 1) + beat_fraction
        return total_beats * self.seconds_per_beat

    def add_note(self, instrument, root_freq, measure, beat, beat_fraction, length_beats):
        start_time = self.beat_to_seconds(measure, beat, beat_fraction)
        dur_seconds = length_beats * self.seconds_per_beat
        start_sample = int(start_time * self.fs)
        note, _ = instrument.synth(root_freq, self.fs, duration=dur_seconds)
        end_sample = start_sample + len(note)
        if end_sample > len(self.timeline):
            self.timeline = np.pad(self.timeline, (0, end_sample - len(self.timeline)))
        self.timeline[start_sample:end_sample] += note

    def play(self):
        sd.play(self.timeline, self.fs)
        sd.wait()

# --- Example instrument ---
screech_growl_harmonics = [
    {'type': 'damped_sawtooth', 'interval': 1, 'mult': 1, 'amp': 0.9, 'decay': 2.0},
    {'type': 'triangle', 'interval': 2 ** (12/12), 'mult': 2, 'amp': 0.2},
    {'type': 'square', 'interval': 2 ** (7/12), 'mult': 3, 'amp': 0.4},
    {'type': 'damped_sine', 'interval': 2 ** (5/12), 'mult': 1, 'amp': 0.5, 'decay': 3.0},
    {'type': 'sine', 'interval': 2 ** (19/12), 'mult': 1, 'amp': 0.2}
]
screech_growl = Instrument(screech_growl_harmonics)

# --- Example: one octave major scale in 4/4 at 60 BPM starting at 55 Hz ---
major_scale_intervals = [0, 2, 4, 5, 7, 9, 11, 12]  # semitones

timeline = Timeline(bpm=60, time_signature=(4,4), measures=4)
measure = 1
beat = 1
for semis in major_scale_intervals:
    freq = 55 * (2 ** (semis/12))
    timeline.add_note(screech_growl, freq, measure, beat, 0.0, 1)
    beat += 1
    if beat > 4:
        beat = 1
        measure += 1

# --- Play and visualize ---
plt.plot(timeline.timeline[:2000])
plt.title("Timeline Waveform Example")
timeline.play()
plt.show()


