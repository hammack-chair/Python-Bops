# Python-Bops

Makin' bippity bops on pippity python 🎶  
The goal is to have a fully functioning DAW that renders audio and makes it easy/fun to create music.  
A visualizer will be added eventually.  

Bop till you stop — anaconda don't want none?  
No python don't want none, hun :*  

---

## Installation

Make sure you have Python 3.10+ installed.

### Option 1: Install packages individually

```powershell
## Upgrade pip (good practice)
python -m pip install --upgrade pip

# For maths and plotting (sounds are maths)
pip install numpy scipy matplotlib

# Audio processing
pip install sounddevice librosa
```
---

### Option 2: Install all requirements together
```powershell
#Installs all requirements locally
pip install -r requirements.txt
```

## Usage

The program is built around three main components: Sound Synthesis, DAW Components, and the Bop Conveyor Belt.

### Sound Synthesis

This portion of the program uses scipy, numpy, and matplotlib to combine basic waveforms into overtone series, creating new instruments. You can shape an instrument’s timbre by adjusting the number, frequency, volume envelope, and amplitude of its overtones, as well as choosing different waveforms for each overtone.

### Instrument Example
```python
# Define the harmonics for each chord tone (relative to the root)
acid_bass_harmonics = [
    {'type': 'sawtooth', 'interval': 1, 'mult': 1, 'amp': 1.0},
    {'type': 'square', 'interval': 1, 'mult': 2, 'amp': 0.5},
    {'type': 'damped_sawtooth', 'interval': 2 ** (7/12), 'mult': 1, 'amp': 0.8, 'decay': 3.0},
    {'type': 'damped_square', 'interval': 2 ** (12/12), 'mult': 2, 'amp': 0.4, 'decay': 2.5},
    {'type': 'sine', 'interval': 2 ** (3/12), 'mult': 3, 'amp': 0.2}
]
# Create the instrument
acid_bass = Instrument(harmonics=acid_bass_harmonics, duration=8.0)
```

This section follows common practices of additive synthesis. To hear how an individual instrument sounds, simply run the instrument’s file in the terminal. Doing so will play the sound and plot a visualization of the waveform. As you add overtones, you can listen and see how the timbre is effected.

This section will continue to grow as new instruments are added.

### DAW Components

The DAW components are the hard-coded classes that make the creation of instruments, the timeline, and effects possible. This is where the main functionality lives for building instruments, placing notes, and applying effects. This section also contains a Fast Fourier Transform (FFT) audio reader (librosia) that passes an audio file and performs analysis (scipy) which can then be used as a set of overtones/harmonics.

### Timeline Component
```python
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
```

New effects and features will be periodically added to expand this section.

### Bop Conveyor Belt

This is where you actually create music. Using a timeline, you can write out songs by calling simple functions with natural language parameters. Be creative—there are many logical ways to build a song!

All frequencies are expressed in Hz (vibrations per second), corresponding to musical notes. Once you’ve written your song, just run the song file in the terminal to hear it.

### Drum Loop
```python
# create timeline
tl = Timeline(bpm=120, time_signature=(4,4), measures=25, fs=44100)

# drum loop
for beat in range(1, tl.time_signature[0] * tl.measures + 1): # runs loop for full timeline
    tl.add_note(minorkick, root_freq=48.109, start_beat=beat-1, end_beat=beat) # kick drum 1
    tl.add_note(scikick, root_freq=48.109, start_beat=beat-1, end_beat=beat) # kick drum 2
    if beat % 2 == 0:
        tl.add_note(snare3, root_freq=48.109, start_beat=beat-1, end_beat=beat-0.5) # snare drum
    tl.add_note(hihat, root_freq=48.109, start_beat=beat-0.75, end_beat=beat-0.625) # hihat hit 1
    tl.add_note(hihat, root_freq=48.109, start_beat=beat-0.5, end_beat=beat-0.375) # hihat hit 2
    tl.add_note(hihat, root_freq=48.109, start_beat=beat-0.25, end_beat=beat-0.125) # hihat hit 3

tl.play() # plays full timeline
```

### Future Feature: Live Coding

A future update will allow for live coding on a continuous timeline loop at a given BPM. In this interface, you’ll be able to add and remove parts (like drum loops and build-ups) on the fly. The song will keep going until you stop the program — perfect for live DJ sets or extended jam sessions with friends :* don't have friends? Create your own backing tracks hehe the world is your toaster!
