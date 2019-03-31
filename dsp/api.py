import numpy as np
import matplotlib.pyplot as plt

class Spectrogram:
    def __init__(self, spectra, transform,
                 window_size, window_skip, sample_rate):
        """Spectra should be a 3D array with first axis time samples,
           second axis frequency, and third axis channel"""
        self.data = np.array(spectra)
        self.transform = transform
        self.window_size = window_size
        self.window_skip = window_skip
        self.sample_rate = sample_rate

    def channels(self):
        return self.data.shape[2]

    def channel(self, n):
        new_spectrogram = self.copy()
        new_spectrogram.data = np.reshape(self.data[:, :, n], (-1, -1, 1))
        return new_spectrogram

    def samples(self):
        return self.data.shape[0]

    def audio_samples(self):
        return self.window_skip * (self.samples() - 1)

    def window_positions(self):
        return range(0, self.audio_samples() + 1,
                     self.window_skip)

    def windows(self, width):
        for i in range(0, self.samples() - width + 1):
            yield self.data[i:i+slice_width]

    def show(self):
        for channel, data in enumerate(self.data.transpose(2, 0, 1)):
            plt.figure()
            plt.imshow(abs(data.T),
                       cmap='viridis',
                       aspect='auto')
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.title('Channel {}'.format(channel + 1))
            plt.xlabel('Time (samples)')
            plt.ylabel('Frequency')
        plt.show()

class Transform:
    def __init__(self, base_function, inverse=None):
        """Base function should accept a 2D array and a sample rate.
           The first array dimension is samples and the second is channels."""
        self.base_function = base_function
        self.inverse = inverse

    def __call__(self, sound):
        return self.base_function(sound.data, sound.sample_rate)
