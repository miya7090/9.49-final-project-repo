# conda activate D:\condaEnvs\pyemmaEnv
# or, conda install pyemma and scikit-learn

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from sklearn.decomposition import FastICA, PCA
import pyemma.msm as msm
import msmtools.generation as msmgen # changed to github msmtools
import msmtools.analysis as msmana # changed
import pyemma.coordinates as coor

import generate_audio as audio_gen

np.random.seed(100)
n_secs = 5
X_max, X, trueAudio = audio_gen.load_audio_signals(n_secs, mixSignals=False) # load the received signals
numSources = X.shape[1]

# load other variables
fs, n_samples = audio_gen.getAudioInfo(n_secs)
time = np.linspace(0, n_secs, n_samples) # time vector

''' Perform analyses! '''
S_ = {}
A_ = {}

ica = FastICA(n_components=numSources)  # ICA!
S_["ica"] = ica.fit_transform(X)  # Reconstruct signals
A_["ica"] = ica.mixing_  # Get estimated mixing matrix

pca = coor.pca(data = X) # PCA!
pc = pca.eigenvectors
S = pca.eigenvalues
km = coor.cluster_kmeans(data = X, k=1000, max_iter=100)
tica = coor.tica(data = X) # TICA!
ic = tica.eigenvectors
L = tica.eigenvalues
S_["pca"] = pca.get_output()[0]
S_["tica"] = tica.get_output()[0]

''' Post-process the estimated signal '''
for estimate in S_.keys():
    S_[estimate] -= np.min(S_[estimate]) # bound left edge at zero
    S_[estimate] *= (X_max / np.max(S_[estimate]))*2 # bound right edge at max*2
    S_[estimate] -= X_max # shift downwards

    # export the estimated signals
    export = S_[estimate].T.astype(np.int16)
    for observed in range(len(export)):
        wavfile.write('Audio files/'+estimate+'_estimated_'+str(observed)+'.wav', fs, export[observed])
