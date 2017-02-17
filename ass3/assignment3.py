#!/usr/bin/env pythonG
from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import fftpack


PREFIX = 'music_speech/'
DENOM = 32768.0

MFCC_MEAN_HEADERS = ["@ATTRIBUTE MFCC-{}_MEAN".format(i) for i in xrange(1, 27)]
MFCC_STD_HEADERS = ["@ATTRIBUTE MFCC-{}_MEAN".format(i) for i in xrange(1, 27)]
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
        returns: sampling frequency, normalized audio
    """
    Fs, x = wavfile.read('{}{}'.format(PREFIX, path))
    return Fs, x / DENOM


# Utility functions

def mag(spectrum):
    """ Calculates magnitude of given FFT spectrum """
    return np.abs(spectrum)
    # return np.sqrt(spectrum.real**2 + spectrum.imag**2)


def preemphasis(signal, coeff=0.95):
    """ Performs preemphasis on signal
        coeff: preemphasis coefficient
    """
    return np.append(signal[0], signal[1:]-coeff*signal[:-1])


# Main code

def freq2mel(f):
    """ Converts frequency to mel Scale which is a closer
    representation of how human hear sound as it is not linear
    """
    return 1127.0 * np.log(1 + f / 700.0)


def mel2freq(mel):
    """ Converts Mel scale to frequency """
    return 700 * (np.exp(mel / 1127.0) - 1)


def getMelFilterBank(sampling_freq, lowfreq=0, highfreq=None, n_filters=26,
                     buffer_size=1024):
    """ Generates Mel filter bank from audio signal """

    # Use half of the sampling frequency as the upper bound
    highfreq = highfreq if highfreq else sampling_freq / 2
    # buffer_size = sampling_freq

    lowmel = freq2mel(lowfreq)
    highmel = freq2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, n_filters + 2)
    # print("Melpoints\n{}".format(melpoints))
    freqpoints = mel2freq(melpoints)
    # print("Frequency points\n{}".format(freqpoints))
    bin = (buffer_size+1) * freqpoints / sampling_freq

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
    """ Generates Mel filter bank from audio signal """

    # Magnitude of spectrum
    magnitude_spectrum = mag(spectrum)

    feats = np.dot(magnitude_spectrum, fbank.T)
    feats = np.log10(feats)

    feature_vec = fftpack.dct(feats, type=2)
    return feature_vec


def generateFFT(wav, buffer_size=1024, overlap=512,
                window_func=np.hamming):
    """ Divide wav file into individual buffer and apply fft to it
        buffer_size: length of window
        overlap: step size
        window_func: windowing function applied to samples
        Returns: positive frequency components
    """
    count = len(wav)
    splitted = np.array(
         # Generate only positive FFT components after multiplying samples
         # with designated window function
         [np.fft.rfft(wav[i:i+buffer_size]*window_func(buffer_size))
          for i in xrange(0, count, buffer_size - overlap)
          if count - i >= buffer_size])
    return splitted


def calcMeanStd(features):
    """ Returns mean and standard deviation of feature slice """
    res = [np.mean(features), np.std(features)]
    return res


def calcFeaturesMeanStd(gt):
    """ Returns 1 X 26 vector for each file
        gt: (path, category)
    """

    Fs, signal = readWav(gt[0])

    # Apply Preemphasis filter
    signal = preemphasis(signal)

    # Generate FFT frames after applying a Hamming Window
    frames = generateFFT(signal)
    fbank = getMelFilterBank(Fs)

    # Extracts MFCC features on each buffered FFT spectrum
    feature_vec = np.apply_along_axis(
        lambda frame: mfcc(frame, fbank), 1, frames)

    # Calculates mean and standard deviation for all extracted features
    mfcc_vectors = (np.apply_along_axis(calcMeanStd,
                    0, feature_vec)).flatten().tolist()

    # Append ground truth category extracted from music_speech.mf
    mfcc_vectors.append(gt[1])
    return mfcc_vectors


def generateARFF(gt, arff_path='assignment3.arff'):
    res = [','.join(map(str, calcFeaturesMeanStd(x))) for x in gt]
    res = ARFF_HEADER + res
    with open(arff_path, 'w') as f:
        f.write('\n'.join(res))
        # Terminates file with newline to pass line count
        f.write('\n')


def plotMelFilters():
    """ Use this to plot for whole frequency range """
    fbank = getMelFilterBank(sampling_freq=22050, buffer_size=22050)
    plt.plot(fbank.T)
    plt.show()
    # plt.savefig("MelFilters_full.png")
    partial = fbank[:, :300]
    plt.clf()
    plt.plot(partial.T)
    plt.show()
    # plt.savefig("MelFilters_300HZ.png")


if __name__ == '__main__':
    gt = parseGroundTruth()
    # generateARFF(gt)
    plotMelFilters()
    # plotPartial()
