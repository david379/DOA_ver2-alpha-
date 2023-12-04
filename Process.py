# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 10:38:56 2023

@author: david
"""

import numpy as np
import soundfile as sf
import DOA_ang as da
def process(w = 5, d = 0.5, m = 1):
    
    x_TD,samplerate = sf.read('audio/signalOG.wav');
    sp = np.arange(1600, np.size(x_TD,0), 1600)
    x = np.split(x_TD, sp, 0)
    l = []
    
    for i in x[:-1]:
        #sf.write('audio/signal-1.wav', i, 32000)
        l.append(da.DOA_ang(i))
    t = np.arange(0, 0.05*len(x) - 0.05, 0.05)
    
    
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
    return list(zip(t,remove_isolated_spikes(l,window_size=w,deviation_factor=d, min_isolation_distance=m))) 