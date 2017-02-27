#!/usr/bin/env python
from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import fftpack


PREFIX = 'music_speech/'
DENOM = 32768.0

MFCC_MEAN_HEADERS = ["@ATTRIBUTE MFCC-{}_MEAN".format(i) for i in xrange(1, 27)]
MFCC_STD_HEADERS = ["@ATTRIBUTE MFCC-{}_STD".format(i) for i in xrange(1, 27)]
FINAL_HEADERS = ['@ATTRIBUTE class {music,speech}\n',
                 '@DATA']

ARFF_HEADER = ['@RELATION music_speech'] + MFCC_MEAN_HEADERS + MFCC_STD_HEADERS + FINAL_HEADERS


# Input/Output

def parseGroundTruth(path='music_speech.mf'):
    """ Returns tuple of (filepath, category) """
    with open("music_speech.mf") as f:
        paths = f.readlines()
        return [(x[0], x[1]) for x in (x.strip().split('\t') for x in paths)]


def readWav(path):
    """ Read wav file
        Returns: sampling frequency, normalized audio
    """
    Fs, x = wavfile.read('{}{}'.format(PREFIX, path))
    return Fs, x / DENOM


# Utility functions

def mag(spectrum):
    """ Calculates magnitude of given FFT spectrum """
    return np.abs(spectrum)


def preemphasis(signal, coeff=0.95):
    """ Performs preemphasis on signal
        coeff: preemphasis coefficient
    """
    preemp = np.append(signal[0], signal[1:]-coeff*signal[:-1])
    return preemp


# Main code

def freq2mel(f):
    """ Converts frequency to Mel Scale """
    return 1127.0 * np.log(1 + f / 700.0)


def mel2freq(mel):
    """ Converts Mel scale to frequency """
    return 700 * (np.exp(mel / 1127.0) - 1)


def getMelFilterBank(sampling_freq, lowfreq=0, highfreq=None, n_filters=26,
                     buffer_size=1024):
    """ Generates Mel filter bank from audio signal """

    # Use half of the sampling frequency as the upper bound
    highfreq = highfreq if highfreq else sampling_freq / 2

    lowmel = freq2mel(lowfreq)
    highmel = freq2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, n_filters + 2)
    freqpoints = mel2freq(melpoints)
    bin = buffer_size * freqpoints / sampling_freq

    fbank = np.zeros((n_filters, int(np.floor(buffer_size / 2 + 1))))

    for m in range(1, n_filters + 1):
        f_m_minus = int(math.floor(bin[m - 1]))  # left
        f_m = int(round(bin[m]))                 # center
        f_m_plus = int(math.ceil(bin[m + 1]))    # right
        for k in range(f_m_minus, f_m_plus):
            xidx = [f_m_minus, f_m, f_m_plus]
            yidx = [0, 1, 0]
            if k < bin[m - 1] or k > bin[m + 1]:
                fbank[m - 1, k] = 0
            else:
                fbank[m - 1, k] = np.interp(k, xidx, yidx)
    return fbank


def mfcc(spectrum, fbank):
    """ Generates MFCC vector for a given spectrum
        fbank: Mel filter banks
    """
    magnitude_spectrum = mag(spectrum)

    feats = np.dot(magnitude_spectrum, fbank.T)
    feats = np.log10(feats)

    feature_vec = fftpack.dct(feats)
    return feature_vec


def generateFFT(wav, buffer_size=1024, overlap=512,
                window_func=np.hamming):
    """ Splits given signal into frames and apply preemphasis to each frame
    before applying FFT
        buffer_size: length of window
        overlap: step size
        window_func: windowing function applied to samples
        Returns: FFT frames
    """
    count = len(wav)
    splitted = np.array(
         # Generate only positive FFT components after multiplying samples
         # with designated window function
         [np.fft.rfft(preemphasis(wav[i:i+buffer_size])*window_func(buffer_size))
          for i in xrange(0, count, buffer_size - overlap)
          if count - i >= buffer_size])
    return splitted


def calcMeanStd(features):
    """ Returns mean and standard deviation of a feature slice """
    # Rounds to sixth decimal point to pass the check
    res = [np.round(np.mean(features), 6), np.round(np.std(features), 6)]
    return res


def calcPerceptualFeatures(gt):
    """ Returns 1 X 26 vector for a given audio signal
        gt: (file path, category)
    """

    Fs, signal = readWav(gt[0])
    # Generate FFT frames after applying a Hamming Window
    frames = generateFFT(signal)
    # Generates Mel Filter banks
    fbank = getMelFilterBank(Fs)

    # Extracts MFCC features from each buffered FFT spectrum
    feature_vec = np.apply_along_axis(lambda frame: mfcc(frame, fbank), 1, frames)
    # Calculates mean and standard deviation for all extracted features
    mfcc_vectors = (np.apply_along_axis(calcMeanStd, 0, feature_vec)).flatten().tolist()

    # Append ground truth category extracted from music_speech.mf
    mfcc_vectors.append(gt[1])
    return mfcc_vectors


def generateARFF(gt, arff_path='assignment3.arff'):
    res = [','.join(map(str, calcPerceptualFeatures(x))) for x in gt]
    res = ARFF_HEADER + res
    with open(arff_path, 'w') as f:
        f.write('\n'.join(res))
        # Terminates file with newline to pass line count
        f.write('\n')


# Plotting

def plotMelFilters(n_filters=26, sampling_freq=22050, maxrange=12000,
                   save_path=None):
    """ Generate Mel filter bank plot
        maxrange: maximum frequency range to be plotted
        save_path: output plot destination
    """
    fbank = getMelFilterBank(sampling_freq)
    highfreq = sampling_freq / 2
    r, c = fbank.shape
    frequencies = np.linspace(0, highfreq, c)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("{} Triangular MFCC filters {} Hz signal, window size {}"
                 .format(n_filters, sampling_freq, 1024))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim([0, maxrange])
    for i in range(r):
        ax.plot(frequencies, fbank[i])
    if not save_path:
        plt.show()
    fig.savefig(save_path)


if __name__ == '__main__':
    gt = parseGroundTruth()
    generateARFF(gt, 'assignment3.arff')
    plotMelFilters(save_path='MelFilters_Full.png')
    plotMelFilters(maxrange=300, save_path='MelFilters_300HZ.png')
