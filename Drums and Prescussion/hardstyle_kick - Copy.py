import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import square, sawtooth

# Sampling rate
fs = 44100

# Duration for a single kick
single_kick_duration = 0.45

# Total duration (4 kicks)
duration = 4 * single_kick_duration

# Define a damped sine wave
def damped_sine(t, freq, amp, decay, phase=0):
    return amp * np.sin(2 * np.pi * freq * t + phase) * np.exp(-decay * t)
def damped_triangle(t, freq, amp, decay, phase=0):
    return amp * (2 * np.abs(2 * ((t * freq + phase/(2*np.pi)) % 1) - 1) - 1) * np.exp(-decay * t)
def damped_square(t, freq, amp, decay, phase=0):
    return amp * square(2 * np.pi * freq * t + phase) * np.exp(-decay * t)

def damped_sawtooth(t, freq, amp, decay, phase=0):
    return amp * sawtooth(2 * np.pi * freq * t + phase) * np.exp(-decay * t)

# Custom waveforms dictionary for extended compatibility
wave_funcs = {
    'damped_sine': damped_sine,
    'damped_triangle': damped_triangle,
    'damped_square': damped_square,
    'damped_sawtooth': damped_sawtooth
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
hardstylekick = Instrument(harmonic_intervals=hardstylekick_intervals, duration=single_kick_duration)

# Choose a root note frequency (e.g., 49.5 Hz)
root_note_freq = 49.5

# Generate one kick
single_kick_signal, _ = hardstylekick.synth(fs, root_note_freq)

# Normalize signal
if np.max(np.abs(single_kick_signal)) > 0:
    single_kick_signal /= np.max(np.abs(single_kick_signal))

# Repeat the kick 4 times with silence in between
silence = np.zeros(int(fs * 0.1))  # 100ms of silence between kicks
signal = np.concatenate([np.concatenate([single_kick_signal, silence]) for _ in range(5)])

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
