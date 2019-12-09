# Conda activate D:\condaEnvs\pyemmaEnv

# Code edited from http://docs.markovmodel.org/lecture_tica.html

import numpy as np
import pyemma.msm as msm
import msmtools.generation as msmgen # changed to github msmtools
import msmtools.analysis as msmana # changed
import pyemma.coordinates as coor
import matplotlib.pylab as plt

# omitted: original generation of gaussian data
# replacing with:

import generate_audio as audio_gen
np.random.seed(20)
n_secs = 5
X_max, X, trueAudio = audio_gen.load_audio_signals(n_secs, mixSignals=True) # load in signals
# here our relevant quantities will be our X (observed signal, timesteps by num channels) and for reference we have our trueAudio (also timesteps by num channels)
print(X_max, X.shape, trueAudio.shape)

''' Calculate PCA (for comparison) and TICA '''
pca = coor.pca(data = X)
pc = pca.eigenvectors
S = pca.eigenvalues
km = coor.cluster_kmeans(data = X, k=100)
tica = coor.tica(data = X)
ic = tica.eigenvectors
L = tica.eigenvalues

# omitted plots

Ypca = pca.get_output()[0]
Ytica = tica.get_output()[0] # tica outputs a list of length 1 for some reason so access that
# you get an array of shape (timesamples x channels)

print(Ypca.shape, Ytica.shape, print(np.max(Ypca), np.max(Ytica)))