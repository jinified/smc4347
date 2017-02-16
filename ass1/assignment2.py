#!/usr/bin/env pythonG
from __future__ import division
import numpy as np
from scipy.io import wavfile


PREFIX = 'music_speech/'
DENOM = 32768.0
ARFF_HEADER = ['@RELATION music_speech',
               '@ATTRIBUTE SC_MEAN NUMERIC',
               '@ATTRIBUTE SRO_MEAN NUMERIC',
               '@ATTRIBUTE SFM_MEAN NUMERIC',
               '@ATTRIBUTE PARFFT_MEAN NUMERIC',
               '@ATTRIBUTE FLUX_MEAN NUMERIC',
               '@ATTRIBUTE SC_STD NUMERIC',
               '@ATTRIBUTE SRO_STD NUMERIC',
               '@ATTRIBUTE SFM_STD NUMERIC',
               '@ATTRIBUTE PARFFT_STD NUMERIC',
               '@ATTRIBUTE FLUX_STD NUMERIC',
               '@ATTRIBUTE class {music,speech}\n',
               '@DATA']


# Input/Output

def parseGroundTruth(path='music_speech.mf'):
    """ Returns tuple of (filepath, category) """
    with open("music_speech.mf") as f:
        paths = f.readlines()
        return [(x[0], x[1]) for x in (x.strip().split('\t') for x in paths)]


def readWav(path):
    return wavfile.read('{}{}'.format(PREFIX, path))[1] / DENOM


# Spectral Features

def mag(spectrum):
    """ Calculates magnitude of given FFT spectrum """
    return np.sqrt(spectrum.real**2 + spectrum.imag**2)


def spectralCentroid(frequencies):
    magnitudes = mag(frequencies)
    centroid = np.sum(magnitudes*np.arange(frequencies.shape[0])) / np.sum(magnitudes)
    return centroid


def spectralRollOff(frequencies, L=0.85):
    """ Determines spectral rolloff, frequency below which 85% of energy is
        located
        L: energy limit
        Returns: bin index that corresponds to frequency below which L of energy is located
    """
    magnitudes = mag(frequencies)
    spectralSum = np.sum(magnitudes)
    rollOffSum = 0
    for i in range(len(frequencies)):
        rollOffSum = rollOffSum + magnitudes[i]
        if rollOffSum >= L*spectralSum:
            return i


def spectralFlatnessMeasure(frequencies):
    """ Calculates distribution of spectral power where 1.0 indicates
    spectrum has similar amount of power in all spectral bands
    """
    magnitudes = mag(frequencies)
    sfm = np.exp(np.mean(np.log(magnitudes))) / np.mean(magnitudes)
    return sfm


def spectralFlux(frequencies):
    """ Calculates spectral flux on buffered spectrum """
    spectralFlux = []
    flux = 0

    for bin in frequencies[0]:
        flux = flux + abs(bin)
    spectralFlux.append(flux)

    for s in range(1, len(frequencies)):
        prevSpectrum = frequencies[s - 1]
        spectrum = frequencies[s]
        flux = 0
        for bin in range(0, len(spectrum)):
            diff = abs(spectrum[bin]) - abs(prevSpectrum[bin])
            if diff < 0:
                diff = 0
            flux = flux + diff
        spectralFlux.append(flux)
    mean, std = np.mean(spectralFlux), np.std(spectralFlux)
    return mean, std


# Main code

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


def extractAudioFeatures(spectrum):
    """ Extract 4 spectral features given buffered FFT spectrum """
    sc = spectralCentroid(spectrum)
    sro = spectralRollOff(spectrum)
    sfm = spectralFlatnessMeasure(spectrum)
    rms = np.sqrt(np.mean(mag(spectrum)**2))
    parfft = np.max(mag(spectrum)) / rms
    return [sc, sro, sfm, parfft]


def calcMeanStd(features):
    """ Returns mean and standard deviation of feature slice """
    res = [np.mean(features), np.std(features)]
    return res


def calcFeaturesMeanStd(gt):
    """ Returns 1 X 10 vector for each file
        gt: (path, category)
    """
    # Return buffered FFT spectrum
    spectrum = generateFFT(readWav(gt[0]))

    # Extracts spectral features on each buffered FFT spectrum
    feature_vec = np.apply_along_axis(extractAudioFeatures, 1, spectrum)

    # Calculates mean and standard deviation for all extracted features
    final_vec = (np.apply_along_axis(calcMeanStd, 0, feature_vec)).flatten().tolist()

    # Spectral Flux is calculated separately and appended
    flux_mean, flux_std = spectralFlux(spectrum)
    final_vec.insert(4, flux_mean)
    final_vec.append(flux_std)

    # Append ground truth category extracted from music_speech.mf
    final_vec.append(gt[1])
    return final_vec


def generateARFF(gt, arff_path='assignment2.arff'):
    res = [','.join(map(str, calcFeaturesMeanStd(x))) for x in gt]
    res = ARFF_HEADER + res
    with open(arff_path, 'w') as f:
        f.write('\n'.join(res))
        # Terminates file with newline to pass line count
        f.write('\n')


if __name__ == '__main__':
    gt = parseGroundTruth()
    generateARFF(gt)
