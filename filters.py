import numpy as np
from joblib import Parallel, delayed
import scipy as sp

# @jit(nopython=True, fastmath=True)
# def meshgrid(x, y, indexing='ij'):
#     xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
#     yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
#     if indexing=='ij':
#         for i in range(x.size):
#             for j in range(y.size):
#                 xx[i,j] = i  # change to x[k] if indexing xy
#                 yy[i,j] = j  # change to y[j] if indexing xy
#     else:
#         for i in range(x.size):
#             for j in range(y.size):
#                 xx[i,j] = x[j]  # change to x[k] if indexing xy
#                 yy[i,j] = y[i]  # change to y[j] if indexing xy
#     return xx, yy

def my2dfilter(s,sigmax,sigmay, ns=2):
    """
    Spatial 2D filter, using gaussian kernel.
    This is used to get large scale trend of SSH

    INPUT:
        - s : signal to smooth
        - sigmax : for gaussian kernel, std in x direction
        - sigmay : for gaussian kernel, std in y direction
        - ns : width of the gaussian filter, in number of std
    OUTPUT:
        - smoothed signal 'sf', same dimensions as 's'
    """

    x, y = np.meshgrid(np.arange(-int(ns*sigmax), int(ns*sigmax)+1), np.arange(-int(ns*sigmay), int(ns*sigmay)+1), indexing='ij')
    #x, y = meshgrid(np.arange(-int(ns*sigmax), int(ns*sigmax)+1), np.arange(-int(ns*sigmay), int(ns*sigmay)+1), indexing='ij')
    cf=np.exp(-(x**2/sigmax**2+y**2/sigmay**2))
    m=~np.isnan(s)*1.
    s = np.where(np.isnan(s),0,s)
    #s[np.isnan(s)]=0.
    # if type(s)=='xarray.core.dataarray.DataArray':
    #     conv2d = lambda x: sp.signal.convolve2d(x, cf, mode="same")
    #     s_sum = xr.apply_ufunc(conv2d, s)
    #     w_sum = xr.apply_ufunc(conv2d, m)
    s_sum = (sp.signal.convolve2d(s, cf, mode='same'))
    w_sum = (sp.signal.convolve2d(m, cf, mode='same'))

    sf = s*0.
    sf[w_sum!=0] = s_sum[w_sum!=0] / w_sum[w_sum!=0]
    return sf

def my2dfilter_over_time(s,sigmax,sigmay, nt, N_CPU, ns=2):
    """
    'my2dfilter' but over each time step and in parallel with Joblib
    """  
    if N_CPU==1:
        list_results = []
        for it in range(nt):
            list_results.append( my2dfilter(s[it,:,:], sigmax, sigmay) )
    else:
        list_results = Parallel(n_jobs=N_CPU)(delayed(my2dfilter)(s[it,:,:], sigmax, sigmay) for it in range(nt))
    
    return np.array(list_results)

def mytimefilter(Hf0):
    """
    This is a time smoothing operator
    
    INPUT:
        - Hf0: array 3D (time,y,x)
    OUTPUT:
        - Hf: smoothed field, same dimensions as Hf0
    """
    
    _, ny, nx = np.shape(Hf0) # shape of array
    time_conv = np.arange(-1*86400,1*86400+3600,3600) # 1D array, time boundaries for kernel (in sec)
    Hf = Hf0*0. # initialization
    
    # time kernel for convolution
    taul = 2*86400
    gl = np.exp(-taul**-2 * time_conv**2) 
    gl = (gl / np.sum(np.abs(gl)))
    # doing the convolution in time
    for ix in range(nx):
        for iy in range(ny):
            Hf[:,iy,ix] = np.convolve(Hf0[:,iy,ix],gl,'same')
    return Hf
