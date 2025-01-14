"""
Tool module for "reconstruct_inertial.py"
"""


import numpy as np

def PSD(time_vect, signal_vect):
    """This function automates the computation of the Power Spectral Density of a signal.
    """
    # Same as amplitude spectrum START ===================
    total_time = time_vect[-1] - time_vect[0]
    dt = time_vect[1] - time_vect[0]
    freq_resolution = 1.0 / total_time
    freq_nyquist = 1. / dt
    
    # samples
    N = len(time_vect)
    
    # build frequency
    frequency = np.arange(0, freq_nyquist, freq_resolution, dtype=float)
    frequency = frequency[: int(N / 2) - 1]  # limit to the first half
    
    raw_fft = np.fft.fft(signal_vect, norm="backward")
    raw_fft /= N
    
    # Takes half, but double the content, excepted the first component
    amplitude_spectrum = 2*np.absolute(raw_fft)[:int(N/2) - 1] 
    amplitude_spectrum[0] /= 2
    # Same as amplitude spectrum END===================
    
    power_spectral_density = 1/2 * np.absolute(
        amplitude_spectrum
        *np.conjugate(amplitude_spectrum)
    )/freq_resolution 
    
    power_spectral_density[0] *= 2
    
    return frequency, power_spectral_density

def detrended_PSD(time_vect,signal_vect,order=1):
	"""
	This is a function that computes the PSD with before a detrending of signal_vect
	By default the detrending is linear (removing ax+b)
	"""
	poly = np.polynomial.Polynomial.fit(time_vect,signal_vect,order)

	detrended = signal_vect - poly(time_vect)
	return PSD(time_vect,detrended)