#!/usr/bin/env python
from __future__ import division
import itertools
import csv
import numpy as np
from scipy.io import wavfile


PREFIX = 'music_speech/'
DENOM = 32768.0
ARFF_HEADER = ['@RELATION music_speech',
               '@ATTRIBUTE RMS_MEAN NUMERIC',
               '@ATTRIBUTE PAR_MEAN NUMERIC',
               '@ATTRIBUTE ZCR_MEAN NUMERIC',
               '@ATTRIBUTE MAD_MEAN NUMERIC',
               '@ATTRIBUTE MEAN_AD_MEAN NUMERIC',
               '@ATTRIBUTE RMS_STD NUMERIC',
               '@ATTRIBUTE PAR_STD NUMERIC',
               '@ATTRIBUTE ZCR_STD NUMERIC',
               '@ATTRIBUTE MAD_STD NUMERIC',
               '@ATTRIBUTE MEAN_AD_STD NUMERIC',
               '@ATTRIBUTE class {music,speech}\n',
               '@DATA']


def parseGroundTruth(path='music_speech.mf'):
    """ Returns tuple of (filepath, category) """
    with open("music_speech.mf") as f:
        paths = f.readlines()
        return [(x[0], x[1]) for x in (x.strip().split('\t') for x in paths)]


def readWav(path):
    return wavfile.read('{}{}'.format(PREFIX, path))[1] / DENOM


def zeroCrossingRate(frame):
    pairs = [1 for i, j in itertools.izip(frame, frame[1:]) if i*j < 0]
    return len(pairs) / (len(frame) - 1)


def extractAudioFeatures(wav):
    """ Extract 5 features (rms, par, zcr, mad, mean_ad) """
    rms = np.sqrt(np.mean(wav**2))
    par = np.max(np.fabs(wav)) / rms
    zcr = zeroCrossingRate(wav)
    med = np.median(wav)
    mad = np.median(np.abs(wav - med))
    mean = np.mean(wav)
    mean_ad = np.mean(np.abs(wav - mean))
    return [rms, par, zcr, mad, mean_ad]


def generateCsvFeatures(gt, csv_path='assignment1a.csv'):
    """ Generates csv of audio features for each wav files """
    features = [[x[0]] + extractAudioFeatures(readWav(x[0])) for x in gt]
    write_csv(features, csv_path)


def calcMeanStd(features):
    """ Returns mean and standard deviation of feature slice """
    arr = [np.mean(features), np.std(features)]
    return arr


def generateARFF(gt, arff_path='assignment1b.arff'):
    res = [','.join(map(str, calcFeaturesMeanStd(x))) for x in gt]
    res = ARFF_HEADER + res
    with open(arff_path, 'w') as f:
        f.write('\n'.join(res))
    # write_csv(res, arff_path)


def calcFeaturesMeanStd(gt):
    """ Returns 1 X 10 vector for each file
        gt: (path, category)
    """
    splitted = splitData(readWav(gt[0]))
    feature_vec = np.apply_along_axis(extractAudioFeatures, 1, splitted)
    final_vec = (np.apply_along_axis(calcMeanStd, 0, feature_vec)).flatten().tolist()
    final_vec.append(gt[1])
    return final_vec


def splitData(wav, buffer_size=1024, overlap=512):
    """ Convert 1D samples into 2D samples
        buffer_size: length of window
        overlap: step size
    """
    count = len(wav)
    splitted = np.array([wav[i:i+buffer_size]
                         for i in xrange(0, count, buffer_size - overlap)
                         if count - i >= buffer_size])
    print(splitted.shape)
    return splitted


def write_csv(data, path):
    with open(path, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)


if __name__ == '__main__':
    gt = parseGroundTruth()
    # Assignment 1A
    generateCsvFeatures(gt)
    # Assignment 1B
    generateARFF(gt)
