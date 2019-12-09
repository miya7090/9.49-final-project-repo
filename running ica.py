import numpy as np
from scipy.io import wavfile
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt

import generate_audio as audio_gen

np.random.seed(100)
n_secs = 15
X_max, X, trueAudio = audio_gen.load_audio_signals(n_secs, mixSignals=True) # load the received signals
numSources = X.shape[1]

# load other variables
fs, n_samples = audio_gen.getAudioInfo(n_secs)
time = np.linspace(0, n_secs, n_samples) # time vector

''' ICA '''
ica = FastICA(n_components=numSources)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

''' Post-process the estimated signal '''
S_ -= np.min(S_) # bound left edge at zero
S_ *= (X_max / np.max(S_))*2 # bound right edge at max*2
S_ -= X_max # shift downwards

# export the estimated signals
export = S_.T.astype(np.int16)
for observed in range(len(export)):
    wavfile.write('Audio files/Ica_estimated_'+str(observed)+'.wav', fs, export[observed])

# For comparison, compute PCA
pca = PCA(n_components=numSources)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

# #############################################################################
# Plot results
plt.figure()
models = [X, trueAudio, S_, H]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals',
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange']
for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.tight_layout()
plt.show()
