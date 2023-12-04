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

if __name__ == '__main__': #Thi line is necessery for the multiprocessing to work
    x_TD,samplerate = sf.read('audio/signalOG.wav'); #Loads the entire multichanneled audio file, assumed to be sampled at 32k
    sp = np.arange(int(samplerate*0.05), np.size(x_TD,0), int(samplerate*0.05)) 
    x = np.split(x_TD, sp, 0) #Splits the entire file into 50ms bins

    pool = m.pool.Pool(processes = 8); #Creates a pool of 8 processes in order to process multiple bins at once
    l = [0] * (len(x)) #Initialize the list of results, which will contain a DOA angle for every bin
    t = time.time() #Starts timing the algorithem
    l = pool.map(da.DOA_ang, x) #Calculates for each bin the DOA angle
    pool.close()

    print(str(t-time.time())) #Prnts the total run time
#%%

    t = np.arange(0, 0.05*len(x)-0.05,0.05)
    plt.figure(dpi = 1200)
    plt.plot(t,l[:-1]) #plot the doa over time
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
    
    yhat = remove_isolated_spikes(l,window_size=5,deviation_factor=0.05)
    plt.figure(dpi = 1200)
    #plt.plot(t[:579],l,'.')
    plt.plot(t, yhat[:-1], c="b") #plot the doa over time after the isolated spike removel
        #%%
        #t = time.time()
        #j = p.process()
        #print(str(t-time.time()))
        #for i in j:
        #    print(i)
