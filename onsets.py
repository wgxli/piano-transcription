import sys

import soundfile
import resampy

from keras.models import model_from_json

import util.analyze
import dsp

_, INPUT_FILENAME = sys.argv

# Load audio file
print('Loading audio.')
data, sample_rate = soundfile.read(INPUT_FILENAME)
data = resampy.resample(data.sum(axis=1), sample_rate, 22050)
sound = (data, 22050)

# Create spectrogram
print('Generating spectrogram.')
spectrogram = dsp.spectrogram(sound)

# Load pre-trained model
print('Loading model.')
with open('model.json', 'r') as f:
    model_json = f.read()
network = model_from_json(model_json)
network.load_weights('weights.h5')

# Predict onsets
print('Detecting onsets.')
predicted_onsets = util.analyze.onsets(spectrogram, network)

# Write results
print('Writing results.')
util.analyze.write_onsets(predicted_onsets)
