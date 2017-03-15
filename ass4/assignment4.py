#!/usr/bin/python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile

# Utilities


def adsr(dur, a, d, s, r):
    """ Generates a linear ADSR envelope.
        :param dur: duration, in number of samples, including the release time.
        :param a:   "Attack" time, in number of samples.
        :param d:   "Decay" time, in number of samples.
        :param s:   "Sustain" amplitude level (should be based on attack amplitude).
        :param r:   "Release" time, in number of samples.
    """
    m_a = 1. / a
    m_d = (s - 1.) / d
    m_r = - s * 1. / r
    len_a = int(a + .5)
    len_d = int(d + .5)
    len_r = int(r + .5)
    len_s = int(dur + .5) - len_a - len_d - len_r
    env = []
    for sample in xrange(len_a):
        env.append(sample * m_a)
    for sample in xrange(len_d):
        env.append(1. + sample * m_d)
    for sample in xrange(len_s):
        env.append(s)
    for sample in xrange(len_r):
        env.append(s + sample * m_r)
    return np.array(env, dtype=np.float64)


def readWav(path, bps=16):
    denom = (2**bps) / 2
    wav = wavfile.read(path)
    return wav[0], wav[1] / denom


def decibel(signal):
    """ Convert amplitude to decibel """
    return 20*np.log10(signal + (10**(-10)))


def getFundamentalFreq(note):
    """ Calculates fundamental frequency given midi note """
    if note == 0:
        # Rest note
        return 0
    return 440 * 2**((note - 69) / 12)


def generateFFT(wav, winsize, overlap, winfunc):
    """ Divide wav file into individual buffer and apply fft to it
        :param winsize: length of window
        :param overlap: step size
        :param winfunc: windowing function applied to samples
        :return:        positive frequency components
    """
    count = len(wav)
    splitted = np.array(
         # Generate only positive FFT components after multiplying samples
         # with designated window function
         [np.fft.rfft(wav[i:i+winsize]*winfunc(winsize))
          for i in xrange(0, count, winsize - overlap)
          if count - i >= winsize])
    return splitted.T


def drawSpectrogram(signal, Fs, img_path=None, winsize=512, overlap=256,
                    winfunc=np.blackman, fmax=None):
    """ Draw a spectrogram given a wav signal with log scale with constant 10**-10
        :param Fs: sampling frequency
    """
    spectrum = generateFFT(signal, winsize, overlap, winfunc)
    fmax = fmax if fmax else Fs / 2
    ylen = spectrum.shape[0]

    x = np.linspace(0, len(signal) / Fs, len(spectrum[0]))
    y = np.linspace(1, fmax, ylen)
    d = decibel(np.abs(spectrum))
    X, Y = np.meshgrid(x, y)
    plt.clf()
    plt.pcolormesh(X, Y, d)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    # ax = plt.gca()
    # liny = np.linspace(1, fmax, 10)
    # ax.set_yticks(liny)
    plt.colorbar().set_label("Amplitude (dB)")
    plt.axis('tight')
    if img_path:
        plt.savefig(img_path)
    else:
        plt.show()


# Main code

def modifyWav(signal, wav_path, dur, n, Fs, dtype=np.int16):
    """ Apply ADSR envelope to signal """
    adsr_params = dict(a=dur/3, d=dur/4, s=.6, r=dur/4)
    env = np.tile(adsr(dur, **adsr_params), n)
    modified = dtype(signal * env)
    if wav_path:
        wavfile.write(wav_path, Fs, modified)
    return modified


def generateSineWave(freq, Fs, N=[1, 3, 5, 7], duration=1.0, amplitude=1.0, dtype=np.int16):
    """ Draws a summation of multiple sine waves
        :param Fs: sampling frequency
        :param N: list of harmonic multipliers
        :param dtype: PCM data type of audio signal
    """
    t = np.arange(int(duration * Fs))
    signals = np.array([amplitude * np.sin(t * (n*freq / Fs) * (2 * np.pi)) for n in N])
    signal = np.sum(signals / len(N), axis=0)
    return dtype(signal)


def generateWav(notes, wav_path, d, bps, Fs):
    """ Generate wav file given MIDI notes and write as wav file
        :param wav_path: wav file output
        :return:         concatenated notes
    """
    # Maximum amplitude given bps
    maxAmp = ((2**bps) / 2) - 1
    waves = np.concatenate([generateSineWave(
        getFundamentalFreq(i), Fs, duration=d, amplitude=maxAmp) for i in notes])
    if wav_path:
        wavfile.write(wav_path, Fs, waves)
    return waves


def additiveSynth(notes, wav_path="notes.wav", d=0.25, bps=16, Fs=32000,
                  adsr=False):
    """ Generates spectrogram given a set of midi notes
        :param d:   duration of each note
        :param bps: number of bits per sample
        :param Fs:  sampling frequency
        :return:    spectrogram
    """
    filename = wav_path.split('.')[0]
    signal = generateWav(notes, wav_path, d, bps, Fs)
    dur = 0.25 * Fs
    if adsr:
        signal = modifyWav(signal, wav_path, dur, len(notes), Fs)
    drawSpectrogram(signal, Fs, img_path="spectrogram-{}.png".format(filename))


if __name__ == '__main__':
    midis = [60, 62, 64, 65, 67, 69, 71, 72, 72, 0, 67, 0, 64, 0, 60]
    additiveSynth(midis)
    # Applying ADSR envelope to original signal
    additiveSynth(midis, wav_path="notes-adsr.wav", adsr=True)
