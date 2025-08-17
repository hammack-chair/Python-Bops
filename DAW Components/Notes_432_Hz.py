import numpy as np

# Reference frequency for A4
f0 = 432  # Hz

# Note names in one octave (using flats and sharps as requested)
note_names = [
    "A", "Bflt", "B", "C", "Dflt", "D", "Eflt", "E", "F", "Fshp", "G", "Gshp"
]

# Piano range: A0 (lowest) to C8 (highest)
lowest_n = int(np.ceil(np.log2(25 / f0) * 12))
highest_n = int(np.floor(np.log2(4119 / f0) * 12))

notes_list = []

for n in range(lowest_n, highest_n + 1):
    freq = f0 * 2 ** (n / 12)
    note_index = n % 12
    octave = (n // 12) + 4  # A4 is n=0, so octave 4 at n=0
    note_label = f"{note_names[note_index]}{octave}"
    notes_list.append((note_label, round(freq, 3)))  # <-- round to nearest thousandth

# Create a dictionary for easy lookup
notes_dict = {label: freq for label, freq in notes_list}

# Example: print all notes
for label, freq in notes_list:
    print(f"{label}: {freq} Hz")

# You can now pull any note by label from notes_dict, e.g.:
# notes_dict['C5']  # Frequency of C in