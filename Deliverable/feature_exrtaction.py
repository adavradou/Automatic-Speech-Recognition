import librosa
from scipy.fftpack import fft
import numpy as np


def fourierTransform(fs,y,label):
    # Number of samplepoints
    N = len(y)
    # sampling period
    T = 1.0 / fs

    yf = fft(y)

    #Extract 1st quarter of frequency domain data
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2) * 0.5  # No need for both semi-axis the rfequency domain in Hertz
    yf = 2.0 * np.abs(yf[:N // 2])  # The frequency magnitude

    # plt.plot(xf,yf,label=str(label) + ' - file :' + file.name +"N" + str(N)) # plot the fourier transform
    # plt.show()

    return xf, yf, N


def fourierPeaks(fs,y,label,no_of_peaks):
    # Number of samplepoints
    N = len(y)
    # sampling period
    T = 1.0 / fs

    yf = fft(y)

    xf = np.linspace(0.0, 1.0/(2.0*T), N/2) * 0.5 # No need for both semi-axis the rfequency domain in Hertz
    yf = 2.0 * np.abs(yf[:N//2]) # The frequency magnitude

    #plt.plot(xf,yf,label=str(label) + ' - file :' + file.name +"N" + str(N)) # plot the fourier transform
    #plt.show()

    # Find dominant freq peak centers and train matching them with their frequency domains
    peaks=[]

    i=0
    step=10 # Peaks greter than step Hz in distance
    for point in yf:
        peaks.append([xf[i],point])
        i+=1

    # Choose greatest magnitude
    peaks=sorted(peaks, key=lambda x: x[1],reverse=True)
    peaks=peaks[:no_of_peaks]

    return xf,yf,N,peaks


def rawData(audioStreams,sr):
    i = 0

    all_features = []
    print("Extracting Features from " + str(len(audioStreams))  + " files")
    for audio in audioStreams:
        samples = librosa.resample(audio, sr, 8000)

        if len(samples) < 8000:
            zero_padded = np.lib.pad(audio, (0, (8000 - len(samples))), 'constant', constant_values=0)
            all_features.append(zero_padded)
        else:
            all_features.append(samples[:8000])

        i += 1
    print("The dataset has been created")
    return all_features


def rawDataStretched(audioStreams,sr,meanDuration):
    sumdur = 0
    i = 0

    all_features = []
    print("Extracting Features from " + str(len(audioStreams))  + " files, mean duration=" + str(meanDuration))
    for audio in audioStreams:
        samples = librosa.resample(audio, sr, 8000)
        duration = librosa.get_duration(samples)
        sumdur += duration

        ratio = duration / meanDuration
        if ratio < 0.05:
            ratio = 0.05

        # Stretrch audio to normalise spoken word duration
        audio = librosa.effects.time_stretch(samples, ratio)

        if len(audio) < 8000:
            zero_padded = np.lib.pad(audio, (0, (8000 - len(audio))), 'constant', constant_values=0)
            all_features.append(zero_padded)
        else:
            all_features.append(audio[:8000])


        i += 1
    print("The dataset has been stretched to the mean duration of the tracks, without affecting pitch")
    return all_features


def extract_mfccs(audioStreams,sr,meanDuration):
    sumdur=0
    i=0
    fsizeEQ = 0
    fsizeL = 0
    fsizeG = 0
    all_features = []
    desiredNoOfFeatures=13
    # Features Extraction an time normalisation
    print("Extracting Features from " + str(len(audioStreams)) + " files, mean duration="+str(meanDuration))
    for audio in audioStreams:

        #sd.play(audio, sr)
        #status = sd.wait()

        duration = librosa.get_duration(audio)
        sumdur+=duration

        ratio = duration / meanDuration
        if ratio < 0.05:
            ratio = 0.05

        # Stretrch audio to normalise spoken word duration
        audio = librosa.effects.time_stretch(audio, ratio)

        #Extract features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=desiredNoOfFeatures, hop_length=178)  # Get specific number of components
        mfccs_flat = mfccs.flatten()

        #Normalize feature data set size
        featuresSize = desiredNoOfFeatures * 44  # mfccs.shape[1]

        if len(mfccs_flat) < featuresSize:
            zero_padded = np.lib.pad(mfccs_flat, (0, (featuresSize - len(mfccs_flat))), 'constant', constant_values=0)
            all_features.append(zero_padded)
            fsizeL += 1
        elif len(mfccs_flat) > featuresSize:
            all_features.append(mfccs_flat[:featuresSize])
            fsizeG += 1
        else:
            fsizeEQ += 1
            all_features.append(mfccs_flat)

        i+=1
    print("MFCC features have been extracted")
    return all_features


def fourier_transform(audioStreams,sr):
    no_of_features = 3500
    sumdur = 0
    i = 0
    features_size = 0
    all_features = []
    print("Extracting Features from " + str(len(audioStreams)) )
    for audio in audioStreams:


        frequency, fourierMagnitude, sampleCount = fourierTransform(sr, audio, str(i))
        length = 3000

        if len(fourierMagnitude) < length:
            zero_padded = np.lib.pad(fourierMagnitude, (0, length - len(fourierMagnitude)), 'constant',
                                     constant_values=0)
            all_features.append(zero_padded)

        else:
            all_features.append(fourierMagnitude[:length])

        i += 1
    print("The dataset has been converted from time space to frequency space")
    return all_features


def fourier_transform_stretched(audioStreams,sr,meanDuration):
    no_of_features=3500
    sumdur=0
    i=0
    features_size=0
    all_features = []
    print("Extracting Features from "+str(len(audioStreams)) +" files, mean duration="+str(meanDuration))
    for audio in audioStreams:

        duration = librosa.get_duration(audio)
        sumdur+=duration

        ratio = duration / meanDuration
        if ratio < 0.05:
            ratio = 0.05

        # Stretrch audio to normalise spoken word duration
        audio = librosa.effects.time_stretch(audio, ratio)

        frequency, fourierMagnitude, sampleCount = fourierTransform(sr, audio, str(i))
        length=3000

        if len(fourierMagnitude) < length:
            zero_padded = np.lib.pad(fourierMagnitude, (0, length - len(fourierMagnitude)), 'constant', constant_values=0)
            all_features.append(zero_padded)

        else:
            all_features.append(fourierMagnitude[:length])


        i+=1
    print("The dataset has been converted from time space to frequency space")
    return all_features


def extract_fourier_peaks(audioStreams,sr,meanDuration):
    sumdur=0
    i=0

    all_features = []
    desiredNoOfFeatures=13
    # Features Extraction an time normalisation
    print("Extracting Features from "+str(len(audioStreams)) +" files, mean duration="+str(meanDuration))
    for audio in audioStreams:

        #sd.play(audio, sr)
        #status = sd.wait()

        duration = librosa.get_duration(audio)
        sumdur+=duration

        ratio = duration / meanDuration
        if ratio < 0.05:
            ratio = 0.05

        # Stretrch audio to normalise spoken word duration
        audio = librosa.effects.time_stretch(audio, ratio)

        #sd.play(audio, sr)
        #status = sd.wait()

        peaksNo=50

        frequency, fourierMagnitude, sampleCount, peaks = fourierPeaks(sr, audio, str(i), peaksNo)

        peaks_flat=np.array(peaks).flatten()

        peaksNo=peaksNo*2

        if len(peaks_flat) < peaksNo:
            zero_padded = np.lib.pad(peaks_flat, (0, peaksNo - len(peaks_flat)), 'constant',
                                     constant_values=0)
            all_features.append(zero_padded)
        else :
            all_features.append(peaks_flat[:peaksNo])

        i+=1
    print("Fourier Peaks have been extracted")
    return all_features


