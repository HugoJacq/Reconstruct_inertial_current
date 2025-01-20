"""
This file gather all functions related to the simple model
    'Unsteady Ekman N layers models'

Including function needed for 4Dvar (TGL, adjoint, cost function and gradient of cost function)

To be used with 'reconstruct_inertial.py'
"""
import numpy as np
import matplotlib as mpl
import scipy.signal
import scipy.interpolate
import warnings
import time as clock
from tools import *

def unstek(time, fc, TAx, TAy, k, return_traj=False):
    """
    Unsteady Ekman model

    INPUT:
        - time  : array of time
        - fc    : scalar Coriolis f
        - TAx   : array wind stress U
        - TAy   : array wind stress V
        - k     : list of boundaries of layer(s)
        - return_traj : if True, return current as complex number
    OUTPUT:
        - array of surface current
        
    Note: this function works with numpy arrays
    """
    K = np.exp(k)

    # initialisation
    nl = int(len(K)//2)
    U = [None]*nl
    for ik in range(nl):
        U[ik]=np.zeros((len(time)), dtype='complex')

    # wind = windU + j*windV
    TA = TAx + 1j*TAy

    # timestep
    dt=time[1]-time[0]

    # model
    with warnings.catch_warnings(action="ignore"): # dont show overflow results
        for it in range(len(time)-1):
            for ik in range(nl):
                if ((ik==0)&(ik==nl-1)): 
                    U[ik][it+1] = U[ik][it] + dt*( -1j*fc*U[ik][it] +K[2*ik]*TA[it] - K[2*ik+1]*(U[ik][it]) )
                else:
                    if ik==0: U[ik][it+1] = U[ik][it] + dt*( -1j*fc*U[ik][it] +K[2*ik]*TA[it] - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) )
                    elif ik==nl-1: U[ik][it+1] = U[ik][it] + dt*( -1j*fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*U[ik][it] )
                    else: U[ik][it+1] = U[ik][it] + dt*( -1j*fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) )

    if return_traj: return U
    else: return np.real(U[0]), np.imag(U[0])

def unstek_tgl(time, fc, TAx, TAy, k, dk):
    """
    Computes the Tangent Linear model for 'unstek'.
    This is a linear approximation of the 'unstek' model derived with control theory.

    INPUT:
        - time  : array of time
        - fc    : scalar Coriolis f
        - TAx   : array wind stress U
        - TAy   : array wind stress V
        - k     : list of boundaries of layer(s)
        - dk    : 
    OUTPUT:
        - First layer values of the TGL for zonal/meridional current, along the time dimension
        
    Note: this function works with numpy arrays
    """    
    K = np.exp(k)
    dK = np.exp(k)*dk

    nl = int(len(K)//2)
    U = [None]*nl
    dU = [None]*nl
    for ik in range(nl):
        U[ik]=np.zeros((len(time)), dtype='complex')
        dU[ik]=np.zeros((len(time)), dtype='complex')
    

    TA = TAx + 1j*TAy

    dt=time[1]-time[0]

    for it in range(len(time)-1):
        for ik in range(nl):
            if ((ik==0)&(ik==nl-1)): 
                U[ik][it+1] = U[ik][it] + dt*( -1j*fc*U[ik][it] +K[2*ik]*TA[it] - K[2*ik+1]*(U[ik][it]) )
                dU[ik][it+1] = dU[ik][it] + dt*( -1j*fc*dU[ik][it] +dK[2*ik]*TA[it] - dK[2*ik+1]*(U[ik][it]) - K[2*ik+1]*(dU[ik][it]) )
            else:
                if ik==0: 
                    U[ik][it+1] = U[ik][it] + dt*( -1j*fc*U[ik][it] +K[2*ik]*TA[it] - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) )
                    dU[ik][it+1] = dU[ik][it] + dt*( -1j*fc*dU[ik][it] +dK[2*ik]*TA[it] - dK[2*ik+1]*(U[ik][it]-U[ik+1][it]) - K[2*ik+1]*(dU[ik][it]-dU[ik+1][it]))
                elif ik==nl-1: 
                    U[ik][it+1] = U[ik][it] + dt*( -1j*fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*U[ik][it] )
                    dU[ik][it+1] = dU[ik][it] + dt*( -1j*fc*dU[ik][it] -dK[2*ik]*(U[ik][it]-U[ik-1][it]) -K[2*ik]*(dU[ik][it]-dU[ik-1][it])    - dK[2*ik+1]*U[ik][it] - K[2*ik+1]*dU[ik][it])
                else: 
                    U[ik][it+1] = U[ik][it] + dt*( -1j*fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) )
                    dU[ik][it+1] = dU[ik][it] + dt*( -1j*fc*dU[ik][it] -dK[2*ik]*(U[ik][it]-U[ik-1][it]) -K[2*ik]*(dU[ik][it]-dU[ik-1][it]) - dK[2*ik+1]*(U[ik][it]-U[ik+1][it]) - K[2*ik+1]*(dU[ik][it]-dU[ik+1][it]) )

    return np.real(dU[0]), np.imag(dU[0])

def unstek_adj(time, fc, TAx, TAy, k, d):
    """
    Computes the adjoint of the vector K for 'unstek'

    INPUT:
        - time  : array of time
        - fc    : scalar Coriolis f
        - TAx   : array wind stress U
        - TAy   : array wind stress V
        - k     : list of k values for each boundaries of the layer(s)
        - d  : innovation vector, observation forcing for [zonal,meridional] currents
    OUTPUT:
        - returns: - adjoint of vectors K
        
    Note: this function works with numpy arrays
    """
    K = np.exp(k)

    U = unstek(time, fc, TAx, TAy, k, return_traj=True)

    # timestep
    dt=time[1]-time[0]

    nl = int(len(k)//2)
    ad_U = [None]*nl
    for ik in range(nl):
        ad_U[ik]=np.zeros((len(time)), dtype='complex')
    ad_U[0] = d[0] + 1j*d[1]

    TA = TAx + 1j*TAy

    ad_K = np.zeros((len(k)), dtype='complex')

    for it in np.arange(len(time)-1)[::-1]:
        for ik in range(nl):
            if ((ik==0)&(ik==nl-1)): 
                ad_U[ik][it] += ad_U[ik][it+1]*np.conjugate(1+dt*( -1j*fc- K[2*ik+1]))
                ad_K[2*ik] += dt*(ad_U[ik][it+1]*np.conjugate(TA[it]))
                ad_K[2*ik+1] += dt*(-ad_U[ik][it+1]*np.conjugate(U[ik][it]))

                #dU[ik][it+1] = dU[ik][it] + dt*( -1j*fc*dU[ik][it] +dK[2*ik]*TA[it] - dK[2*ik+1]*(U[ik][it]) - K[2*ik+1]*(dU[ik][it]) )
            else:
                if ik==0: 
                    ad_U[ik][it] += ad_U[ik][it+1]*np.conjugate(1+dt*( -1j*fc- K[2*ik+1])) 
                    ad_U[ik+1][it] += ad_U[ik][it+1]*np.conjugate(dt*K[2*ik+1])
                    ad_K[2*ik] += dt*(ad_U[ik][it+1]*np.conjugate(TA[it])) 
                    ad_K[2*ik+1] += dt*(-ad_U[ik][it+1]*np.conjugate(U[ik][it]-U[ik+1][it]))
                    #dU[ik][it+1] = dU[ik][it] + dt*( -1j*fc*dU[ik][it] +dK[2*ik]*TA[it] - dK[2*ik+1]*(U[ik][it]-U[ik+1][it]) - K[2*ik+1]*(dU[ik][it]-dU[ik+1][it]))
                elif ik==nl-1: 
                    ad_U[ik][it] += ad_U[ik][it+1]*np.conjugate(1+dt*( -1j*fc- K[2*ik]- K[2*ik+1]))
                    ad_U[ik-1][it] += ad_U[ik][it+1]*np.conjugate(dt*K[2*ik])
                    ad_K[2*ik] += ad_U[ik][it+1]*(-dt*np.conjugate(U[ik][it]-U[ik-1][it]))
                    ad_K[2*ik+1] += ad_U[ik][it+1]*dt*np.conjugate(-U[ik][it])
                    #dU[ik][it+1] = dU[ik][it] + dt*( -1j*fc*dU[ik][it] -dK[2*ik]*(U[ik][it]-U[ik-1][it]) -K[2*ik]*(dU[ik][it]-dU[ik-1][it])    - dK[2*ik+1]*U[ik][it] - K[2*ik+1]*dU[ik][it])
                else: 
                    ad_U[ik][it] += ad_U[ik][it+1]*np.conjugate(1+dt*( -1j*fc- K[2*ik]- K[2*ik+1]))
                    ad_U[ik+1][it] += ad_U[ik][it+1]*np.conjugate(dt*K[2*ik+1])
                    ad_U[ik-1][it] += ad_U[ik][it+1]*np.conjugate(dt*K[2*ik])
                    ad_K[2*ik] += ad_U[ik][it+1]*np.conjugate(-dt*(U[ik][it]-U[ik-1][it]))
                    ad_K[2*ik+1] += dt*(-ad_U[ik][it+1]*np.conjugate(U[ik][it]-U[ik+1][it]))

                    #dU[ik][it+1] = dU[ik][it] + dt*( -1j*fc*dU[ik][it] -dK[2*ik]*(U[ik][it]-U[ik-1][it]) -K[2*ik]*(dU[ik][it]-dU[ik-1][it]) - dK[2*ik+1]*(U[ik][it]-U[ik+1][it]) - K[2*ik+1]*(dU[ik][it]-dU[ik+1][it]) )

    ad_k = np.exp(k)*ad_K

    return np.real(ad_k)

def cost(pk, time, fc, TAx, TAy, Uo, Vo, Ri):
    """
    Computes the cost function of reconstructed current vs observations for 'unstek'

    INPUT:
        - pk    :
        - time  : array of time
        - fc    : scalar Coriolis f
        - TAx   : array wind stress U
        - TAy   : array wind stress V
        - Uo    : U current observation
        - Vo    : V current observation
        - Ri    :
    OUTPUT:
        - scalar cost
        
    Note: this function works with numpy arrays
    """
    U, V = unstek(time, fc, TAx, TAy, pk)
    with warnings.catch_warnings(action="ignore"): # dont show overflow results
        J = 0.5 * np.nansum( ((Uo - U)*Ri)**2 + ((Vo - V)*Ri)**2 )
    if np.sum( np.isnan(U) ) + np.sum(np.isnan(V))>0:
        # some nan have been detected, the model has crashed with 'pk'
        # so J is nan.
        J = np.nan
    return J

def grad_cost(pk, time, fc, TAx, TAy, Uo, Vo, Ri,):  
    """
    Computes the gradient of the cost function for 'unstek'

    INPUT:
        - pk    : vector with K first guess values (len = nb layers)
        - time  : array of time
        - fc    : scalar Coriolis f
        - TAx   : array wind stress U
        - TAy   : array wind stress V
        - Uo    : U current observation
        - Vo    : V current observation
        - Ri    : error statistics (for now is 1)
    OUTPUT:
        - gradient of the cost function
        
    Note: this function works with numpy arrays
    """
    U, V = unstek(time, fc, TAx, TAy, pk,)

    # distance to observations (innovation)
    # this is used in the adjoint to add a forcing where obs is available
    d_U = (Uo - U)*Ri
    d_V = (Vo - V)*Ri
    #   = 0 where no observation available
    d_U[np.isnan(d_U)]=0.
    d_V[np.isnan(d_V)]=0.
    # computing the gradient of cost function with TGL
    dJ_pk = unstek_adj(time, fc, TAx, TAy, pk, [d_U,d_V])

    return - dJ_pk

def score_RMSE_component(Ua, Va, Ut, Vt):
    """
    Measure of the error from 'unsteady Ekman model' to observations (Uo,Vo)

    INPUT:
        - Ua    : reconstructed zonal current (m/s)
        - Va    : reconstructed meridional current (m/s)
        - Ut    : true zonal current (m/s)
        - Vt    : true meridional current (m/s)
    OUTPUT:
        - scalar, RMSe score for each current component
        
    Note: this function works with numpy arrays
    """
    RMS_u = score_RMSE( Ua,Ut )
    RMS_v = score_RMSE( Va,Vt )

    return RMS_u,RMS_v

def score_PSDerr(time, Ua, Va, Ut, Vt, show_score=False, smooth_PSD=False):
    """
    Measure of the error from 'unsteady Ekman model' to observations (Uo,Vo)
    
    METHOD: 
    
        Score = 1- PSD( Ua-Ut ) / PSD(Ut), if score > 0.5 then it is well reconstructed

    INPUT:
        - time  : dimensions of U
        - Ua    : reconstructed zonal current (m/s)
        - Va    : reconstructed meridional current (m/s)
        - Ut    : true zonal current (m/s)
        - Vt    : true meridional current (m/s)
        - show_score : boolean, if true then the value of score=0.5 is returned
        - smooth_PSD : boolean, whether to smooth or not
    OUTPUT:
        - f : frequency
        - score : 1- PSD( Ua-Ut ) / PSD(Ut)
        
    Tobedone : V component
    """
    # for smoothing filter
    window = 51 # odd
    order = 4
    
    
    Err = np.abs(Ua - Ut)
    f,PSD_e = detrended_PSD(time,Err)
    f,PSD_s = detrended_PSD(time,Ut)
    # f,PSD_e = PSD(time,Err)
    # f,PSD_s = PSD(time,Ut)
    score = 1 - PSD_e/PSD_s
    
    if smooth_PSD or show_score:
        smoothed = savitzky_golay(score,window,order)
    
    if show_score:
        f_score = smoothed[0]
        # find first occurence of score = 0.5
        for ik in range(len(f)-1,0,-1):
            if smoothed[ik] < 0.5:
                f_score = f[ik]
                break
        return f, score, f_score, smoothed
    elif smooth_PSD and not show_score:
        return f, score, smoothed
    else: return f,score
    
    
# rotary spectra
# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022JD037139
# et demander à Clément ses scripts