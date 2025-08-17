import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import square, sawtooth

# Sampling rate
fs = 44100

# Duration for a single kick
single_kick_duration = 0.5

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
robofap_intervals = [
    # Sub-bass body
    {'type': 'damped_square', 'ratio': 1.00,  'amp': 15.00, 'decay': 7},    # ~29.5 Hz
    {'type': 'damped_sine', 'ratio': 2.00,  'amp': 11.75, 'decay': 8},    # ~59 Hz

    # Mid bass harmonics
    {'type': 'damped_sine', 'ratio': 3.25,  'amp': 0.045, 'decay': 9},    # ~95.9 Hz
    {'type': 'damped_sine', 'ratio': 5.00,  'amp': 0.035, 'decay': 10},   # ~147.5 Hz
    {'type': 'damped_sine', 'ratio': 6.70,  'amp': 0.025, 'decay': 11},   # ~197.6 Hz

    # Higher transient/click content
    {'type': 'damped_triangle', 'ratio': 10.0,  'amp': 0.15, 'decay': 13},   # ~295 Hz
    {'type': 'damped_triangle', 'ratio': 17.0,  'amp': 0.08, 'decay': 14},   # ~501.5 Hz
    {'type': 'damped_triangle', 'ratio': 23.5,  'amp': 0.05, 'decay': 15},   # ~693 Hz
]


# Create the instrument with intervals
robofap = Instrument(harmonic_intervals=robofap_intervals, duration=single_kick_duration)

# Choose a root note frequency (e.g., 49.5 Hz)
root_note_freq = 55

# Generate one kick
single_kick_signal, _ = robofap.synth(fs, root_note_freq)

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
