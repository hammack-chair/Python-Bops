import sys
import os
import sounddevice as sd

# Add the parent directory to Python's module search path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from DAW_Components import Instrument, Timeline



minorkick_intervals = [
    {'type': 'damped_sine', 'ratio': 1.0, 'amp': 1.0, 'decay': 8},
    {'type': 'damped_sine', 'ratio': 1.095, 'amp': 0.6, 'decay': 10},
    # ...
]

minorkick = Instrument(harmonics=minorkick_intervals, duration=0.5)
tl = Timeline(bpm=120, time_signature=(4,4), measures=2, fs=44100)

for beat in range(1, tl.time_signature[0] * tl.measures + 1):
    tl.add_note(minorkick, root_freq=48.109, start_beat=beat-1, end_beat=beat)

tl.play()
