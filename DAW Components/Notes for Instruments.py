from Notes_432_Hz import notes_dict
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from Notes_432_Hz import notes_dict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Instruments import Instrument, wave_funcs

def generate_note_waveforms(instrument, fs=44100):
    """
    Returns a dictionary mapping note labels to their synthesized waveforms for the given instrument.
    """
    note_waveforms = {}
    for note_label, freq in notes_dict.items():
        # Temporarily set the instrument's base_freq to the note's frequency
        orig_base_freq = instrument.base_freq
        instrument.base_freq = freq
        signal, _ = instrument.synth(fs)
        instrument.base_freq = orig_base_freq  # Restore original base_freq
        # Normalize
        signal = signal / np.max(np.abs(signal))
        note_waveforms[note_label] = signal
    return note_waveforms

# Example usage:
# all_notes = generate_note_waveforms(piano)
# To play a note later:
# sd.play(all_notes['C4'], fs)
# sd.wait()

def make_chord_instrument(chord_tones, harmonics_per_tone, duration=4.0):
    """
    chord_tones: list of base frequencies (e.g. [root, third, fifth, seventh])
    harmonics_per_tone: list of lists, each inner list contains dicts:
        [
            [ {'type': 'sine', 'mult': 1, 'amp': 0.5}, ... ],  # for chord_tones[0]
            [ {'type': 'triangle', 'mult': 1, 'amp': 0.3}, ... ],  # for chord_tones[1]
            ...
        ]
    duration: duration in seconds
    """
    harmonics = []
    for base, harm_list in zip(chord_tones, harmonics_per_tone):
        for h in harm_list:
            harmonics.append({**h, 'base': base})
    # Custom synth method to use each harmonic's own base
    class ChordInstrument(Instrument):
        def synth(self, t):
            signal = np.zeros_like(t)
            for h in self.harmonics:
                func = wave_funcs[h['type']]
                freq = h['base'] * h['mult']
                amp = h['amp']
                phase = h.get('phase', 0)
                signal += func(t, freq, amp, phase)
            return signal
    inst = ChordInstrument(base_freq=1, harmonics=harmonics)
    inst.duration = duration
    return inst
# Example usage:
# Cmaj7 chord: C (261.63 Hz), E (329.63 Hz), G (392.00 Hz), B (493.88 Hz)
chord_note_names = ['A1']
chord_tones = [notes_dict[n] for n in chord_note_names]

harmonics_per_tone = [
    [  # C
        {'type': 'sine', 'mult': 1, 'amp': 0.5},
        {'type': 'sine', 'mult': 2, 'amp': 0.2}
    ]
]

# Use your make_chord_instrument function
maj7synth = make_chord_instrument(chord_tones, harmonics_per_tone, duration=4.0)

# Synthesize and play
fs = 44100
t = np.linspace(0, maj7synth.duration, int(fs * maj7synth.duration), endpoint=False)
signal = maj7synth.synth(t)
signal /= np.max(np.abs(signal))
sd.play(signal, fs)
plt.plot(t[:1000], signal[:1000])
plt.title("Cmaj7 chord from note list")
plt.show()
sd.wait()