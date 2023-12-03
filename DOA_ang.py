# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:10:36 2023

@author: david
"""

"""
Created on Wed Nov 22 16:20:23 2023

@author: david
"""
import numpy as np
import time
import scipy.io
import numba
import math
from calc_sampleParam import calc_sampleParam
from calc_FD_GCC import calc_FD_GCC
from calc_SRP import calc_SRP
from calc_STFT import calc_STFT
import soundfile as sf
from numba import jit
#@jit()
def DOA_ang(x_TD):
    ### ACOUSTIC SETUP
#    from numpy import linalg as LA
    
    
#    from numpy import  matlib as mb
#    import scipy.io as spio
    
    #speed of sound
    c = 343;
    #sample rate
    fs = 32000;
    #bandlimit
    pi=math.pi
    w_0 = pi*fs/2;
    #SNR in dB
    
    
    ## MICROPHONE ARRAY
    # circular array, 10cm radius, six microphones
    
    #import matplotlib.pyplot as plt
    tmp = scipy.io.loadmat('mic_arrays/platform_mic_array.mat')
    # array center
    #arrayCenterPos = tmp.get('arrayCenterPos')
    # microphone positions
    micPos = tmp.get('micPos');
    # number of microphones
    M = len(micPos)
    P = M*(M - 1)/2
    
    tmp = scipy.io.loadmat('DOAs/DOA.mat');
    DOAvec_list = tmp.get('DOA_list');
    Delta_t_list = tmp.get('Delta_list');
    ### SOURCE LOCATIONS
    # 8 different locations
#    tmp = scipy.io.loadmat('locations/randLoc.mat');
#    true_loc = tmp.get('true_loc');
    # compute ground truth DOA vectors for source locations
    
#    from DOA import DOA
#    true_DOAvec, DOAang=DOA(true_loc,arrayCenterPos)
    L = 1;
    
    
    # STFT PARAMETERS
    # window size
    win_time = 0.05;
    N_STFT = fs*win_time;
    # shift
    R_STFT = N_STFT/2;
    # window
    win = np.sqrt(np.hanning(N_STFT));
    N_STFT_half = math.floor(N_STFT/2)+1;
    # frequency vector
    omega = 2*pi*np.transpose(np.linspace(0,(fs/2),N_STFT_half));
    
    
    
    # SRP APPROXIMATION PARAMETERS
    # compute sampling period and number of samples within TDOA interval
    
    T, N_mm  =calc_sampleParam(micPos, w_0, c);
    #number of auxilary samples (approximation will be computed for all values in vector)
    N_aux = np.array([2]);
    
    ## PROCESSING
    
    # init results (per source location, frame, number of auxilary samples)
    # approximation error in dB
    
    
    
    
    Tictoc = np.zeros((1, 1));
    

    
    
    
    #GENERATE MICROPHONE SIGNALS
    #speech componentfor selected source
    #x_TD,samplerate = sf.read('audio/signal-1.wav');
    #x_TDmono, Fs = sf.read('audio/signal1.wav');
    #x_TDmono = x_TDmono[:, 1];
    #from calc_INPUT_SIGNAL import calc_INPUT_SIGNAL
    #x_TD = calc_INPUT_SIGNAL(x_TDmono, micPos, true_loc[3,:], c, Fs, fs);
    #noise component
    
    
    
    # transform to STFT domain
    
    t = time.time();
    x_STFT,f_x = calc_STFT(x_TD, fs, win, N_STFT, R_STFT, 'onesided');
    
    
        # discard frames that do not contain speech energy(local SNR 15 dB below average)
    
        # final microphone signal in STFT domain
    y_STFT = np.reshape(x_STFT[:, 0, :],(np.size(x_STFT,0),1,np.size(x_STFT,2)))
    
    
    
        ## PROCESSING
    #from calc_SRPapprFast import calc_SRPapprFast
    
    psi_STFT = calc_FD_GCC(y_STFT); #sorun yok
    maxIdx_conv = 0;
    maxIdx_conv_prev = maxIdx_conv;
    size_prev = 0;
    
    xi_mm_samp = np.zeros((L,28),dtype=object);
    
    for i in range(0,np.size(Delta_t_list, 1)):
        Delta_t = Delta_t_list[0,i];
        DOAvec = DOAvec_list[0, i];
        index = (maxIdx_conv_prev ) * size_prev + maxIdx_conv;
        if i: 
            Delta_t = np.reshape(Delta_t[:, :, index],(np.size(Delta_t,0),np.size(Delta_t,1)))#Delta_t[:, :, index];
            DOAvec = np.reshape(DOAvec[:, :, index],(np.size(DOAvec,0),np.size(DOAvec,1)))#DOAvec[:, :, index];
        Delta_t_i = Delta_t;
        DOAvec_i = DOAvec;
        size_prev = np.size(DOAvec, 0);
            # SRP approximation based on shannon nyquist sampes
            #print('* compute SRP approximation...')
        
        (SRP_appr, xi_mm_samp) = calc_SRP(psi_STFT, omega, T, N_mm, N_aux, Delta_t_i, i, xi_mm_samp);
        
            ####
        
 #       approxErr_dB = np.zeros([L, len(N_aux)]);
 #       locErr = np.zeros([L, len(N_aux) + 1]);
      
        
        for N_aux_ind in range (1,len(N_aux)+1):
    
            maxIdx_conv_prev = index;
            maxIdx_conv = np.argmax(SRP_appr[:,:, N_aux_ind-1], 1);
            estim_DOAvec = DOAvec_i[0:len(maxIdx_conv),:];
    elapsed = time.time() - t;
    
    Tictoc[0] = elapsed;
    
    print('mean time per target: '+ str(np.mean(Tictoc)))
    print('DONE.')
    return(np.arctan2(estim_DOAvec[0, 1],estim_DOAvec[0, 0])*180.0/np.pi)


