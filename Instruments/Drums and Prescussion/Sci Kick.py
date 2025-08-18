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
scikick_intervals = [
    {'type': 'damped_sine', 'ratio': np.float64(1.0), 'amp': np.float32(1.0), 'decay': np.float64(1.878)},
    {'type': 'damped_sine', 'ratio': np.float64(0.989), 'amp': np.float32(0.956), 'decay': np.float64(1.881)},
    {'type': 'damped_sine', 'ratio': np.float64(1.011), 'amp': np.float32(0.937), 'decay': np.float64(1.879)},
    {'type': 'damped_sine', 'ratio': np.float64(0.978), 'amp': np.float32(0.813), 'decay': np.float64(1.891)},
    {'type': 'damped_sine', 'ratio': np.float64(1.022), 'amp': np.float32(0.793), 'decay': np.float64(1.882)},
    {'type': 'damped_sine', 'ratio': np.float64(1.033), 'amp': np.float32(0.625), 'decay': np.float64(1.891)},
    {'type': 'damped_sine', 'ratio': np.float64(0.967), 'amp': np.float32(0.608), 'decay': np.float64(1.906)},
    {'type': 'damped_sine', 'ratio': np.float64(1.044), 'amp': np.float32(0.497), 'decay': np.float64(1.91)},
    {'type': 'damped_sine', 'ratio': np.float64(1.055), 'amp': np.float32(0.444), 'decay': np.float64(1.936)},
    {'type': 'damped_sine', 'ratio': np.float64(1.066), 'amp': np.float32(0.434), 'decay': np.float64(1.962)},


# subpeaks
    {'type': 'damped_sine', 'ratio': np.float64(1.396), 'amp': np.float32(0.197), 'decay': np.float64(1.294)},
    {'type': 'damped_sine', 'ratio': np.float64(1.407), 'amp': np.float32(0.194), 'decay': np.float64(1.293)},
    {'type': 'damped_sine', 'ratio': np.float64(1.429), 'amp': np.float32(0.194), 'decay': np.float64(1.274)},
    {'type': 'damped_sine', 'ratio': np.float64(1.549), 'amp': np.float32(0.193), 'decay': np.float64(1.216)},
    {'type': 'damped_sine', 'ratio': np.float64(1.418), 'amp': np.float32(0.193), 'decay': np.float64(1.286)},
    {'type': 'damped_sine', 'ratio': np.float64(1.495), 'amp': np.float32(0.193), 'decay': np.float64(1.236)},
    {'type': 'damped_sine', 'ratio': np.float64(1.44), 'amp': np.float32(0.193), 'decay': np.float64(1.264)},
    {'type': 'damped_sine', 'ratio': np.float64(1.538), 'amp': np.float32(0.193), 'decay': np.float64(1.223)},
    {'type': 'damped_sine', 'ratio': np.float64(1.56), 'amp': np.float32(0.193), 'decay': np.float64(1.212)},
    {'type': 'damped_sine', 'ratio': np.float64(1.505), 'amp': np.float32(0.193), 'decay': np.float64(1.231)},
]


# Create the instrument with intervals
scikick = Instrument(harmonic_intervals=scikick_intervals, duration=single_kick_duration)

# Choose a root note frequency (e.g., 49.5 Hz)
root_note_freq = 70.5

# Generate one kick
single_kick_signal, _ = scikick.synth(fs, root_note_freq)

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
