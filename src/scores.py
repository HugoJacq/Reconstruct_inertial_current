"""
Tool box about measuring score for reconstructed series

to be used with 'reconstruct_inertial.py'
"""
import numpy as np
from .tools import *

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
    
    
def rotary_PSD():
    """
    """
    print('rotary_PSD: tbd')
# rotary spectra
# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022JD037139
# et demander à Clément ses scripts