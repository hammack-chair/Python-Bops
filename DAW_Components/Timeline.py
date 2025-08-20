import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square, sawtooth
import sounddevice as sd

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


