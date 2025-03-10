"""
tool box
"""
import numpy as np

def nearest(array,value):
	"""
	Array is 1D,
	value is 0D
	"""
	return np.argmin(np.abs(array-value))	

def score_RMSE(U,Ut):
    """
    RMSE between U and true Ut
    
    INPUT:
		- U: 1D array, shape of Ut
		- Ut: 1D array
    OUTPUT:
		- scalar, RMSE
    """
    nt = len(U)
    return np.sqrt( np.sum( (U-Ut)**2 )/ nt )
