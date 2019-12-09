# Conda activate D:\condaEnvs\pyemmaEnv

# Code edited from http://docs.markovmodel.org/lecture_tica.html

import numpy as np
import pyemma.msm as msm
import msmtools.generation as msmgen # changed to github msmtools
import msmtools.analysis as msmana # changed
import pyemma.coordinates as coor
import matplotlib.pylab as plt

def assign(X, cc):
    T = X.shape[0]
    I = np.zeros((T),dtype=int)
    for t in range(T):
        dists = X[t] - cc
        dists = dists ** 2
        I[t] = np.argmin(dists)
    return I

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

exit()


def draw_arrow(a, v, color):
    plt.arrow(0, 0, a*v[0], a*v[1], color=color, width=0.02, linewidth=3)

plt.figure(figsize=(4,7))
plt.scatter(X[:,0], X[:,1], marker = '.', color='black') # change: debugged
draw_arrow(S[0], pc[:,0], color='blue')
#draw_arrow(S[1], pc[:,1], color='blue')
plt.text(0.5, 2.5, 'PCA', color='blue', fontsize = 20, fontweight='bold', rotation='vertical')
draw_arrow(4*L[0], ic[:,0], color='orange')
#draw_arrow(4*L[1], ic[:,1], color='orange')
plt.text(-1.5, 1.7, 'TICA', color='orange', fontsize = 20, fontweight='bold', rotation='vertical')
plt.xlim(-4,4) # change: debugged
plt.ylim(-7,7) # change: debugged
plt.show() # change: added

Ypca = pca.get_output()[0][:,0]
Ytica = tica.get_output()[0][:,0]

plt.hist(Ypca, bins=50, histtype='step', linewidth=3, label='PCA', color='blue') # change: debugged
plt.hist(4*Ytica, bins=50, histtype='step', linewidth=3, label='TICA', color='orange') # change: debugged
plt.xlabel('essential coordinate (1st principal or indipendent component)') # change: debugged
plt.ylabel('projected histogram') # change: debugged
plt.legend() # change: debugged
plt.show() # change: added

cc_pca = np.linspace(np.min(Ypca), np.max(Ypca))
I_pca = assign(Ypca[:,None], cc_pca)
cc_tica = np.linspace(np.min(Ypca), np.max(Ypca))
I_tica = assign(Ytica[:,None], cc_tica)

'''lags = [1,2,5,7,10,15,20]
its_pca = msm.its([I_pca], lags)
its_tica = msm.its([I_tica], lags)
plt.plot([1,20],[msmana.timescales(P)[1],msmana.timescales(P)[1]], linewidth=3, linestyle='dashed', color='black', label='exact') # change: debugged
# plt.plot(its_pca.get_lagtimes(), its_pca.get_timescales()[:,0], linewidth=3, color='blue', label='PCA') # change: added plt prefix but still not working
#plt.plot(its_tica.get_lagtimes(), its_tica.get_timescales()[:,0], linewidth=3, color='orange', label='TICA') # # change: added plt prefix but still not working
plt.xlabel('lag time') # change: debugged
plt.ylabel('relaxation timescale') # change: debugged
plt.legend(loc='center') # change: debugged
plt.show() # change: added''' # change: omitted
