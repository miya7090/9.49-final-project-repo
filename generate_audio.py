import numpy as np
from scipy.io import wavfile

#defaultAudioFolder = "Audio files/"

#defaultAudioSources = ["Power A.wav", "Comeback A.wav", "Speechless A.wav"]

defaultAudioFolder = "Do not sync_audio files/edited_what kind of future/"
#defaultAudioSources = ["Cdhk97.wav","Eng.wav","Enigmaticstar.wav","Fluffywoozi.wav","Moe.wav","Seongjijun.wav","Vobo.wav"]
defaultAudioSources = ["Cdhk97.wav","Moe.wav","Seongjijun.wav"]

def mix_signals(signals, noiseFactor):
    ''' Mix the signals '''
    numSignals = signals.shape[1]
    avg = 1/numSignals # don't exceed maximum volume
    A = np.random.uniform(low=0, high=avg, size=(numSignals,numSignals)) # the mixing array
    print("mixing array:", A)
    X = np.dot(signals, A.T) # observed signal
    X += noiseFactor * np.random.normal(size=X.shape) # add noise
    return X

def getAudioInfo(n_secs, audioFolder = defaultAudioFolder, audioSources = defaultAudioSources):
    framerate, _ = wavfile.read(audioFolder+audioSources[0]) # get frame rate
    n_samples = int(n_secs*framerate) # number of samples for 15 seconds of audio
    return framerate, n_samples

def load_audio_signals(n_secs, audioFolder = defaultAudioFolder, audioSources = defaultAudioSources, noiseFactor = 0, mixSignals = True):
    # set mixSignals to False if using real-world (already mixed) recordings
    framerate, n_samples = getAudioInfo(n_secs)

    ''' Begin loading the signals '''
    audioData = []
    for filename in audioSources: # iterate through each audio file
        fs, data = wavfile.read(audioFolder+filename) # load audio data
        if fs != framerate:
            raise ValueError # all framerates must match
        audioData.append(data[:n_samples, 0]) # trim to 15 seconds and keep only one channel

    print("Loading complete")
    audioData = np.array(audioData, dtype=np.float).T # reformatting

    ''' Mix the signals '''
    if mixSignals:
        X = mix_signals(audioData, noiseFactor)
    else:
        X = audioData
    # export the observed signals
    export = X.T.astype(np.int16)
    for observed in range(len(export)):
        wavfile.write('Audio files/source_observed_'+str(observed)+'.wav', fs, export[observed])
    
    ''' Pre-process the received signal '''
    print("Pre-processing signals of shape", X.shape)
    X_max = np.max(X) # will map to this max later
    X = np.subtract(X, X.mean(axis=0)) # de-mean
    X = np.divide(X, X.std(axis=0)) # de-std # to-do: replace with denoise through PCA

    return X_max, X, audioData

