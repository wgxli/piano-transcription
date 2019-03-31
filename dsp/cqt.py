from functools import lru_cache

import numpy as np

from librosa.util import sparsify_rows

import dsp

def wavelets(frequencies, quality, sample_rate, width):
    filters = np.zeros([len(frequencies), width])
    for index, frequency in enumerate(frequencies):
        filter_width = round(quality * sample_rate / frequency)
        if filter_width > width:
            raise ValueError('Given width ({}) lower than filter width ({}) for {} Hz!'.format(width, filter_width, frequency))
        sample_points = quality * np.linspace(-np.pi, np.pi, filter_width)
        carrier = np.cos(sample_points)
        wavelet = carrier / filter_width * np.hamming(filter_width)
        filters[index] = dsp.pad(wavelet, width)
    return filters

@lru_cache(maxsize=32)
def kernel(frequencies, quality, width, sample_rate, sparsity_quantile=0.1):
    print("Generating kernel: len(f)={}, Q={}, sr={}, width={}".format(len(frequencies), quality, sample_rate, width))
    filters = wavelets(frequencies, quality, sample_rate, width)
    raw_kernel = np.fft.rfft(filters, axis=1)
    return sparsify_rows(raw_kernel, quantile=sparsity_quantile)

def get_transform(frequencies, quality):
    """Returns the CQT transform for the given frequencies with the given Q-factor."""
    def cqt_transform(sound):
        samples, sample_rate = sound
        kern = kernel(frequencies, quality, len(samples), sample_rate)
        spectrum = np.fft.rfft(samples)
        return abs(kern @ spectrum)
    return cqt_transform

def frequencies(bins_per_octave, low_freq, high_freq):
    frequency_list = [low_freq]
    spacing_factor = pow(2, 1/bins_per_octave)
    while frequency_list[-1] < high_freq:
        frequency_list.append(spacing_factor * frequency_list[-1])
    return tuple(frequency_list)
