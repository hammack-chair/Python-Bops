import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# Sampling rate
fs = 44100

# Duration for a single kick
single_kick_duration = 0.5

# Total duration (4 kicks)
duration = 4 * single_kick_duration

# Define a damped sine wave
def damped_sine(t, freq, amp, decay, phase=0):
    return amp * np.sin(2 * np.pi * freq * t + phase) * np.exp(-decay * t)

# Custom waveforms dictionary for extended compatibility
wave_funcs = {
    'damped_sine': damped_sine
}

class Instrument:
    def __init__(self, harmonic_intervals, duration):
        self.harmonic_intervals = harmonic_intervals
        self.duration = duration

    def synth(self, fs, root_freq):
        t = np.linspace(0, self.duration, int(fs * self.duration), endpoint=False)
        signal = np.zeros_like(t)
        for h in self.harmonic_intervals:
            func = wave_funcs[h['type']]
            freq = root_freq * h['ratio']
            amp = h['amp']
            decay = h['decay']
            phase = h.get('phase', 0)
            signal += func(t, freq, amp, decay, phase)
        return signal, t


# Harmonic intervals as ratios relative to the root note
minorkick_intervals = [
    {'type': 'damped_sine', 'ratio': 1.0, 'amp': 1.0, 'decay': 8},
    {'type': 'damped_sine', 'ratio': 1.095, 'amp': 0.6, 'decay': 10},
    {'type': 'damped_sine', 'ratio': 1.19, 'amp': 0.4, 'decay': 12},
    {'type': 'damped_sine', 'ratio': 1.238, 'amp': 0.3, 'decay': 14},
    {'type': 'damped_sine', 'ratio': 1.287, 'amp': 0.2, 'decay': 16},
    {'type': 'damped_sine', 'ratio': 1.414, 'amp': 0.1, 'decay': 18}  # Optional higher overtone
]

# Create the instrument with intervals
minorkick = Instrument(harmonic_intervals=minorkick_intervals, duration=single_kick_duration)

# Choose a root note frequency (e.g., 49.5 Hz)
root_note_freq = 29.5

# Generate one kick
single_kick_signal, _ = minorkick.synth(fs, root_note_freq)

# Normalize signal
if np.max(np.abs(single_kick_signal)) > 0:
    single_kick_signal /= np.max(np.abs(single_kick_signal))

# Repeat the kick 4 times with silence in between
silence = np.zeros(int(fs * 0.1))  # 100ms of silence between kicks
signal = np.concatenate([np.concatenate([single_kick_signal, silence]) for _ in range(15)])

# Create time axis for plotting
t = np.linspace(0, len(signal)/fs, len(signal), endpoint=False)

# Plot the waveform (first full second)
plt.figure(figsize=(10, 4))
plt.plot(t[:int(fs * 1.0)], signal[:int(fs * 1.0)])
plt.title("Synthesized Minor Kick - First Hit")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)


# Ensure sound is played after closing the plot
print("Playing sound...")
sd.play(signal, fs)
plt.show()
sd.wait()
print("Playback finished.")