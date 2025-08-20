# DAW_Components package
"""
DAW Components - Core audio synthesis and timeline functionality

This package provides the essential components for creating digital audio:
- Instrument: For creating synthesizer instruments with harmonic series
- Timeline: For sequencing and arranging musical events
- Waveform functions: Basic and advanced audio waveforms
- Effects: Audio processing and effects

Usage:
    from DAW_Components import Instrument, Timeline, sine_wave, sawtooth_wave
"""

# Import core classes from their respective modules
from .Instrument import Instrument, wave_funcs
from .Timeline import Timeline

# Import waveform functions for convenience
from .Instrument import (
    sine_wave,
    triangle_wave, 
    square_wave,
    sawtooth_wave,
    damped_sine,
    damped_triangle,
    damped_square,
    damped_sawtooth,
    white_noise,
    pink_noise
)

# Import additional components if they exist
try:
    from .Effects import *
except ImportError:
    pass

try:
    from .edm_synth_core import *
except ImportError:
    pass

# Version info
__version__ = "1.0.0"

# Define what gets imported with "from DAW_Components import *"
__all__ = [
    'Instrument',
    'Timeline', 
    'sine_wave',
    'triangle_wave',
    'square_wave', 
    'sawtooth_wave',
    'damped_sine',
    'damped_triangle',
    'damped_square',
    'damped_sawtooth',
    'white_noise',
    'pink_noise',
    'wave_funcs'
]