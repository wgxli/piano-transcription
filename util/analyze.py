from music21.pitch import Pitch
import numpy as np
import dsp


WINDOW_WIDTH = 16
NOTES = 88
DETECTION_THRESHOLD = 0.8
SILENCE_THRESHOLD = 0.5

CHUNK_SIZE = 128

def windows(spectrogram):
    # Loop through all possible reading windows
    for i in range(-WINDOW_WIDTH//2, len(spectrogram) - WINDOW_WIDTH//2):
        if i < 0:
            # Window near beginning
            data_slice = np.zeros((WINDOW_WIDTH, dsp.BINS))
            data_slice[-i:] = spectrogram[:i + WINDOW_WIDTH]
        elif i > len(spectrogram) - WINDOW_WIDTH:
            # Window near end
            data_slice = np.zeros((WINDOW_WIDTH, dsp.BINS))
            data_slice[:len(spectrogram) - WINDOW_WIDTH - i] = spectrogram[i:]
        else:
            # Normal window in middle
            data_slice = spectrogram[i:i + WINDOW_WIDTH]

        # Ignore windows that are too quiet
        if data_slice.max() < SILENCE_THRESHOLD:
            yield 0 * data_slice
            continue

        # Return normalized window
        yield data_slice / np.max(data_slice)

def piano_roll(spectrogram, network):
    # Create piano roll
    piano_roll = []

    # Padding
    piano_roll.append(np.zeros((1, NOTES)))

    # Evaluate for each window position
    chunk = []
    for window in windows(spectrogram):
        chunk.append(window)
        if len(chunk) == CHUNK_SIZE:
            piano_roll.append(network.predict(np.array(chunk)))
            chunk = []
    if chunk:
        piano_roll.append(network.predict(np.array(chunk)))

    # Padding
    piano_roll.append(np.zeros((1, NOTES)))
    return np.vstack(piano_roll)

def onsets(spectrogram, network):
    # Compute binary piano roll
    binary_piano_roll = piano_roll(spectrogram, network) > DETECTION_THRESHOLD

    # Find onset times
    onset_times = []
    for note, row in enumerate(binary_piano_roll.T):
            onset_start = np.where(row[1:] & ~row[:-1])[0]
            onset_stop = np.where(~row[1:] & row[:-1])[0]
            note_frames = 0.5 * (onset_start + onset_stop)
            for frame in note_frames:
                onset_times.append((dsp.frames_to_time(frame - 1), note))
    return onset_times


MIDI_OFFSET = 21

def write_onsets(onsets):
    for time, note in sorted(onsets):
        frequency = Pitch(midi=note + MIDI_OFFSET).frequency
        print(f'{time}\t{frequency:.2f}')
