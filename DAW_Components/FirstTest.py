from Instruments import Instrument  # Make sure Instruments.py is in the same folder or PYTHONPATH
from Notes_432_Hz import notes_dict  # Make sure Notes 432 Hz.py is renamed to Notes_432_Hz.py

def play_note_on_instrument(note_label, instrument, fs=44100):
    """
    Plays a note (by label, e.g. 'G6') on a given Instrument instance.
    """
    if note_label not in notes_dict:
        raise ValueError(f"Note '{note_label}' not found in notes_dict.")
    freq = notes_dict[note_label]
    t = np.linspace(0, instrument.duration, int(fs * instrument.duration), endpoint=False)
    # Override instrument base_freq for this note
    orig_base_freq = instrument.base_freq
    instrument.base_freq = freq
    signal = instrument.synth(t)
    instrument.base_freq = orig_base_freq  # Restore original base_freq
    signal /= np.max(np.abs(signal))  # Normalize
    sd.play(signal, fs)
    plt.figure(figsize=(10, 4))
    plt.plot(t[:1000], signal[:1000])
    plt.title(f"{note_label} played on instrument")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    sd.wait()

# Example usage:
# play_note_on_instrument('G6', piano)
# play_note_on_instrument('C5', organ)