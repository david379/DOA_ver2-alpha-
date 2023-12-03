# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:35:29 2023

@author: david
"""

import time
import numpy as np
import soundfile as sf
import DOA_ang as da
import matplotlib.pyplot as plt
import Process as p
import multiprocessing as m

#x4 = np.array_split(np.array(x), 16)
if __name__ == '__main__':
    x_TD,samplerate = sf.read('audio/signalOG.wav');
    sp = np.arange(1600, np.size(x_TD,0), 1600)
    #sp2 = np.arange(2400, np.size(x_TD,0), 1600)
    x = np.split(x_TD, sp, 0)
    pool = m.pool.Pool(processes = 8);
#x2 = np.split(x_TD, sp2, 0)
#c = [val for pair in zip(x, x2) for val in pair]
    l = [0] * (len(x))
    t = time.time()
    l = pool.map(da.DOA_ang, x)
#for i in range(16):
#    l[i] = pool.map(da.DOA_ang, x4[i])
    pool.close()
#l = [j for i in l for j in i]
    print(str(t-time.time()))
'''for j,i in enumerate(x[:-1]):
    t = time.time()
#    sf.write('audio/signal-1.wav', i, 32000)
    l[j]=(da.DOA_ang(i, 32000))
    print(str(t-time.time()))'''
#%%
'''
t = np.arange(0, 0.05*len(x)-0.05,0.05)
plt.figure(dpi = 1200)
plt.plot(t,l[:-1])
#%%
def remove_isolated_spikes(datan, window_size=3, min_isolation_distance=1, deviation_factor=2.0):
    """
    Remove isolated spikes from numpy array by comparing each point to its neighbors.

    Parameters:
    - data: numpy array
        Input data with potential spikes.
    - window_size: int
        Size of the window for computing the local median.
    - min_isolation_distance: int
        Minimum distance between spikes to be considered isolated.
    - deviation_factor: float
        Factor for determining the threshold based on local median.

    Returns:
    - numpy array
        Data with isolated spikes removed.
    """
    data = datan.copy()
    # Find the indices of potential spikes
    spike_indices = np.arange(len(data))

    # Remove isolated spikes
    for idx in spike_indices:
        if idx > 0 and idx < len(data) - 1:
            window_start = max(0, idx - window_size // 2)
            window_end = min(len(data), idx + (window_size + 1) // 2)
            local_median = np.median(data[window_start:window_end])

            if np.abs(data[idx] - local_median) > deviation_factor * np.median(np.abs(data - local_median)):
                # If the point deviates significantly from the local median, consider it an isolated spike
                data[idx] = local_median

    return data
#%%

yhat = remove_isolated_spikes(l,window_size=5,deviation_factor=0.05) # window size 51, polynomial order 3
plt.figure(dpi = 1200)
#plt.plot(t[:579],l,'.')
plt.plot(t, yhat, c="b")
#%%
t = time.time()
j = p.process()
print(str(t-time.time()))
#for i in j:
#    print(i)
'''