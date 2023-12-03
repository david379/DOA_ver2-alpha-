import numpy as np
import pandas as pd
import math
import torch

def calc_SRP(Psi_STFT, omega, T, N_mm, N_aux, Delta_t_i, i, xi_mm_samp):

    J = np.size(Delta_t_i,0);
    P = np.size(Delta_t_i,1);
    K = np.size(Psi_STFT,0);
    L = np.size(Psi_STFT,1);
    P = np.size(Psi_STFT,2);

    SRP_stack = np.zeros([L, J, len(N_aux)]);

    nT = [];
    #xi_mm_samp = [];
    '''
    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else: 
        dev = "cpu" 
    device = torch.device("cpu")
    print(dev)
    '''
    if i == 0:
        
        xi_mm_samp = np.zeros((L,P),dtype=object);
        #xi_mm_samp_t = torch.tensor(xi_mm_samp)
        #xi_mm_samp_t = xi_mm_samp_t.to(device)

    for l in range (1,L+1):
        SRP = np.zeros([J, len(N_aux)]);
        xi_mm_int = np.zeros([J, len(N_aux)]);

        for p in range (1,P+1):

            psi = Psi_STFT[:, l-1, p-1];

            N_half = N_mm[p-1] + max(N_aux);
            N_half = int(N_half);
            n = range(-N_half, N_half+1);
            n = np.array(n);
            nT.append( np.transpose(n * T));
            aa = nT[p - 1];
            aa = aa.reshape(-1, 1);
            
            if i == 0:
                #psi_t = torch.tensor(psi, dtype = torch.cdouble)
                #psi_t = psi_t.to(device)
                #omega_t = torch.tensor(omega.reshape(1, -1), dtype = torch.cdouble)
                #omega_t = omega_t.to(device)
                #aa_t = torch.tensor(aa, dtype = torch.cdouble)
                #aa_t = aa_t.to(device)
                xi_mm_samp[l - 1,p - 1] = np.real((np.dot(np.exp(np.dot((1j * aa), omega.reshape(1, -1))), psi)))
                #xi_mm_samp = xi_mm_samp_t.numpy()

            for N_aux_ind in range(1, len(N_aux) + 1):

                N_offset = max(N_aux) - N_aux[N_aux_ind-1];
                tmp2 = xi_mm_samp[l - 1][p - 1];
                tmp3 = n[N_offset: len(n) - N_offset];  # 9,
                tmp4 = np.tile(tmp3, (len(Delta_t_i), 1));  # 8101,9
                yy = Delta_t_i[:, p - 1];  # 8101,
                zz = yy.reshape(-1, 1);  # 8101,1
                gg = np.dot(np.sinc(zz / T - tmp4), tmp2[N_offset: len(tmp2) - N_offset]);
                gg = gg.reshape(-1, 1);
                xi_mm_int[:, N_aux_ind - 1: N_aux_ind] = gg;


            SRP = SRP + 2 * xi_mm_int;

        SRP_stack[l-1,:,:] = SRP;

    return (SRP_stack, xi_mm_samp);


