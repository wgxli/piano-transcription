import numpy as np

from music21.pitch import Pitch

import dsp.cqt

LOW_PITCH = Pitch('D1', microtone = (-100/3))
HIGH_PITCH = Pitch('C9', microtone = (+100/3))

BINS_PER_OCTAVE = 36
BINS = 286
QUALITY = 96

FILTER_WIDTH = 2**16
FRAME_RATE = 20

# Utilities
FREQUENCIES = dsp.cqt.frequencies(BINS_PER_OCTAVE,
                                  LOW_PITCH.frequency,
                                  HIGH_PITCH.frequency)
CQT_TRANSFORM = dsp.cqt.get_transform(FREQUENCIES, QUALITY)

def frames_to_time(frames):
    """Converts a duration in spectrogram frames to seconds."""
    return frames / FRAME_RATE

def time_to_frames(time):
    """Converts a duration in seconds to spectrogram frames."""
    return time * FRAME_RATE

def pad(array, width):
    """Zero-pads and centers an array."""
    total_width = width - len(array)
    left_width = total_width // 2
    right_width = total_width - left_width
    return np.pad(array, (left_width, right_width), 'constant')

def clips(sound, size, skip):
    """Yields clips of the given size of the sound with the given spacing."""
    samples, sample_rate = sound
    for index in range(0, len(samples) - size + 1, skip):
            yield (samples[index:index + size], sample_rate)

def spectrogram(sound):
    samples, sample_rate = sound
    
    skip = round(sample_rate / FRAME_RATE)
    sound = (pad(samples, len(samples) + FILTER_WIDTH), sample_rate)
    spectra = [CQT_TRANSFORM(clip)
               for clip in clips(sound, FILTER_WIDTH, skip)]
    return np.array(spectra)
