from DAW_Components import edm_synth_core as esc


minorkick_intervals = [
    {'type': 'damped_sine', 'ratio': 1.0, 'amp': 1.0, 'decay': 8},
    {'type': 'damped_sine', 'ratio': 1.095, 'amp': 0.6, 'decay': 10},
    # ...
]

minorkick = esc.Instrument(harmonics=minorkick_intervals, duration=0.5)
tl = esc.Timeline(bpm=120, measures=4)

for beat in range(1, tl.time_signature[0] * tl.measures + 1):
    tl.add_note(minorkick, root_freq=48.109, start_beat=beat-1, end_beat=beat)

tl.play()
