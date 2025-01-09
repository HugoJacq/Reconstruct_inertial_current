import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
import pdb 
import glob
from netCDF4 import Dataset
import scipy
import scipy.signal
import scipy.interpolate
#import scipy.constants
#from matplotlib.gridspec import GridSpec
#import cartopy.crs as ccrs
#import matplotlib.ticker as mticker
#import cartopy.mpl.ticker as cticker
from datetime import datetime, timedelta
import scipy.optimize as opt
import xarray as xr
import time as clock

start = clock.time()

##### FUNCTIONS
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
        - smoothed signal
    """

    if type(s)=='xarray.core.dataarray.DataArray':
        s = s.data
    x, y = np.meshgrid(np.arange(-int(ns*sigmax), int(ns*sigmax)+1), np.arange(-int(ns*sigmay), int(ns*sigmay)+1), indexing='ij')
    cf=np.exp(-(x**2/sigmax**2+y**2/sigmay**2))
    m=~np.isnan(s)*1.
    s = np.where(np.isnan(s),0,s)
    #s[np.isnan(s)]=0.
    if type(s)=='xarray.core.dataarray.DataArray':
        conv2d = lambda x: scipy.signal.convolve2d(x, cf, mode="same")
        s_sum = xr.apply_ufunc(conv2d, s)
        w_sum = xr.apply_ufunc(conv2d, m)
    else:
        s_sum = (scipy.signal.convolve2d(s, cf, mode='same'))
        w_sum = (scipy.signal.convolve2d(m, cf, mode='same'))

    sf = s*0.
    sf[w_sum!=0] = s_sum[w_sum!=0] / w_sum[w_sum!=0]
    return sf

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
    """

    K = np.exp(k)

    U = unstek(time, fc, TAx, TAy, k, return_traj=True)

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
    """
    U, V = unstek(time, fc, TAx, TAy, pk)

    
        
    
    J = 0.5 * np.nansum( ((Uo - U)*Ri)**2 + ((Vo - V)*Ri)**2 )

    if np.sum( np.isnan(U) ) + np.sum(np.isnan(V))>0:
        # some nan have been detected, the model has crashed with 'pk'
        # so J is nan.
        J = np.nan

    return J

def grad_cost(pk, time, fc, TAx, TAy, Uo, Vo, Ri):  
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
    """
    U, V = unstek(time, fc, TAx, TAy, pk)

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

###############
# bottom lat, top lat, left lon, right lon, 
box= [-25, -24, 45., 46, 24000., 24010.]
grav = 10 # m/s2

XARRAY = False
NDIM = 1

# UNSTEADY EKMAN MODEL
dt=60 # timestep
# LAYER DEFINITION
#   -> number of values = number of layers
#   -> values = turbulent diffusion coefficient
#pk=np.array([-3,-12])  # 1 layers
pk=np.array([-3,-2])  # 1 layers
#pk=np.array([-3,-8,-10,-12]) # 2 layers
#pk=np.array([-2,-4,-6,-8,-10,-12]) # 3 layers

# spatial location. We work in 1D
ir=4
jr=4
path_save_png = './png_1D/'

# File location
files = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/U_V/llc2160_2020-11-*_SSU-SSV.nc"))
filesH = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/SSH/llc2160_2020-11-*_SSH.nc"))
filesW = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/v10m/llc2160_2020-11-*_v10m.nc"))
filesD = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/KPPhbl/llc2160_2020-11-*_KPPhbl.nc"))

# On Jackzilla
# files = np.sort(glob.glob("/data2/nobackup/clement/Data/Llc2160/llc2160_daily_latlon_SSC/llc2160_2020-11-*_SSU-SSV.nc"))
# filesH = np.sort(glob.glob("/data2/nobackup/clement/Data/Llc2160/llc2160_daily_latlon_SSH/llc2160_2020-11-*_SSH.nc"))
# filesW = np.sort(glob.glob("/data2/nobackup/clement/Data/Llc2160/llc2160_daily_latlon_wind/llc2160_2020-11-*_v10m.nc"))
# filesD = np.sort(glob.glob("/data2/nobackup/clement/Data/Llc2160/llc2160_daily_latlon_KPPhbl/llc2160_2020-11-*_KPPhbl.nc"))


print('* Opening files ...')
if XARRAY: # xarray version
    # opening datasets
    dsUV = xr.open_mfdataset(files)
    dsH = xr.open_mfdataset(filesH)
    dsW = xr.open_mfdataset(filesW)
    dsD = xr.open_mfdataset(filesD)
    # searching for lon,lat indexes
    glon = dsUV.lon
    glat = dsUV.lat
    ix = np.where((glon>=box[0])&(glon<=box[1]))[0]
    iy = np.where((glat>=box[2])&(glat<=box[3]))[0]
    glon = glon[ix[0]:ix[-1]+1]
    glat = glat[iy[0]:iy[-1]+1]
    # selecting data
    gU = dsUV.SSU.isel(lat=slice(iy[0],iy[-1]+1),lon=slice(ix[0],ix[-1]+1))
    gV = dsUV.SSV.isel(lat=slice(iy[0],iy[-1]+1),lon=slice(ix[0],ix[-1]+1))
    gTx = dsW.geo5_u10m.isel(lat=slice(iy[0],iy[-1]+1),lon=slice(ix[0],ix[-1]+1))   
    gTy = dsW.geo5_v10m.isel(lat=slice(iy[0],iy[-1]+1),lon=slice(ix[0],ix[-1]+1))           
    gH = dsH.SSH.isel(lat=slice(iy[0],iy[-1]+1),lon=slice(ix[0],ix[-1]+1))
    gMLD = dsD.KPPhbl.isel(lat=slice(iy[0],iy[-1]+1),lon=slice(ix[0],ix[-1]+1))

    # filtering out erronous values
    gU = xr.where(gU>1e5,np.nan,gU)
    gV = xr.where(gV>1e5,np.nan,gV)
    gTx = xr.where(gTx>1e5,np.nan,gTx)
    gTy = xr.where(gTy>1e5,np.nan,gTy)
    gH = xr.where(gH>1e5,np.nan,gH)

    # note: a voir pour que la suite fonctionne !


else:
    for it in range(len(files)):
        with Dataset(files[it], 'r') as fcid:
            if it==0:
                glon = np.array(fcid.variables['lon'][:])
                glat = np.array(fcid.variables['lat'][:])
                ix = np.where((glon>=box[0])&(glon<=box[1]))[0]
                iy = np.where((glat>=box[2])&(glat<=box[3]))[0]
                glon=glon[ix[0]:ix[-1]+1]
                glat=glat[iy[0]:iy[-1]+1]
                gU = np.zeros((len(files)*24,len(iy),len(ix))) + np.nan
                gV = np.zeros((len(files)*24,len(iy),len(ix))) + np.nan
                gTx = np.zeros((len(files)*24,len(iy),len(ix))) + np.nan
                gTy = np.zeros((len(files)*24,len(iy),len(ix))) + np.nan            
                gH = np.zeros((len(files)*24,len(iy),len(ix))) + np.nan
                gMLD = np.zeros((len(files)*24,len(iy),len(ix))) + np.nan
            gU[it*24:it*24+24,:,:] = np.array(fcid.variables['SSU'][:,iy[0]:iy[-1]+1,ix[0]:ix[-1]+1])
            gV[it*24:it*24+24,:,:] = np.array(fcid.variables['SSV'][:,iy[0]:iy[-1]+1,ix[0]:ix[-1]+1])
        with Dataset(filesH[it], 'r') as fcid:
            gH[it*24:it*24+24,:,:] = np.array(fcid.variables['SSH'][:,iy[0]:iy[-1]+1,ix[0]:ix[-1]+1])
        with Dataset(filesW[it], 'r') as fcid:
            gTx[it*24:it*24+24,:,:] = np.array(fcid.variables['geo5_u10m'][:,iy[0]:iy[-1]+1,ix[0]:ix[-1]+1])  
            gTy[it*24:it*24+24,:,:] = np.array(fcid.variables['geo5_v10m'][:,iy[0]:iy[-1]+1,ix[0]:ix[-1]+1])     
        with Dataset(filesD[it], 'r') as fcid:
            gMLD[it*24:it*24+24,:,:] = np.array(fcid.variables['KPPhbl'][:,iy[0]:iy[-1]+1,ix[0]:ix[-1]+1])  

    # filtering out erronous values
    gU[gU>1e5] = np.nan
    gV[gV>1e5] = np.nan
    gTx[gTx>1e5] = np.nan
    gTy[gTy>1e5] = np.nan
    gH[gH>100] = np.nan        

    

print('         done')
# shape of gridded U current
nt,ny,nx = np.shape(gU)
# time array
gtime = np.arange(0,nt)*3600



# LARGE SCALE MOTIONS
# > getting large scale SSH to get geostrophy
#     for now, only SSH but if we add advection, we'll need to smooth U and V too

print('* getting large scale motion ...')
#time_conv = np.arange(-3*86400,3*86400+dt,dt) A REMETTRE
time_conv = np.arange(-1*86400,1*86400+3600,3600)
# taul = np.zeros((len(iy),len(ix))) + 1*86400
# gl = (np.exp(-1j*np.outer(fc[:],time_conv))*np.exp(-np.outer(taul**-2,time_conv**2))).reshape(len(time),len(time_conv))
# gl = np.exp(np.outer(-taul**-2,time_conv**2)) 
# gl = (gl.T / np.sum(np.abs(gl), axis=1).T)
if XARRAY:
    Hf0 = gH.load()
else:
    Hf0 = gH
if NDIM>1:
    Hf0 = Hf0*0.
    #Uf0 = gU*0.
    #Vf0 = gV*0.

    # doing the convolution in space
    for it in range(nt):
        #print(it)
        Hf0[it,:,:] = my2dfilter(gH[it,:,:],3,3)
        #Uf0[it,:,:] = my2dfilter(gU[it,:,:],3,3)
        #Vf0[it,:,:] = my2dfilter(gV[it,:,:],3,3)



Hf = gH*0.
#Uf = gU*0.
#Vf = gV*0.

# time kernel for convolution
taul = 2*86400
gl = np.exp(-taul**-2 * time_conv**2) 
gl = (gl / np.sum(np.abs(gl)))
# doing the convolution in time
for ix in range(nx):
    for iy in range(ny):
        Hf[:,iy,ix] = np.convolve(Hf0[:,iy,ix],gl,'same')
        #Uf[:,iy,ix] = np.convolve(Uf0[:,iy,ix],gl,'same')
        #Vf[:,iy,ix] = np.convolve(Vf0[:,iy,ix],gl,'same')

# geostrophic current from SSH
dlon=glon[1]-glon[0]
dlat=glat[1]-glat[0]
if XARRAY:
    dlon,dlat = dlon.values,dlat.values
glon2,glat2 = np.meshgrid(glon,glat)
fc = 2*2*np.pi/86164*np.sin(glat2*np.pi/180)
if XARRAY:
    Hf = Hf.values
gUg=Hf*0.
gVg=Hf*0.

print('* geostrophic current from SSH ...')
for it in range(nt):
    gVg[it,:,1:-1] = grav/fc[:,1:-1]*(Hf[it,:,2:]-Hf[it,:,:-2])/(dlon*110000*np.cos(glat2[:,1:-1]*np.pi/180))/2
    gUg[it,1:-1,:] = -grav/fc[1:-1,:]*(Hf[it,2:,:]-Hf[it,:-2,:])/(dlat*110000)/2
gUg[:,0,:]=gUg[:,1,:]
gUg[:,-1,:]=gUg[:,-2,:]
gVg[:,:,0]=gVg[:,:,1]
gVg[:,:,-1]=gVg[:,:,-2]


# stress as: C x wind**2
gTAx = 8e-6*np.sign(gTx)*gTx**2
gTAy = 8e-6*np.sign(gTy)*gTy**2




## INTERPOLATION
print('* interpolaton on unsteady ekman model timestep')
# This is interpolation on reconstructed time grid (every dt)
# this is necessary for the simple model 'unstek'
time = np.arange(0,gtime[-1]+dt,dt)

# ageostrophic current
finterpU = scipy.interpolate.RegularGridInterpolator([gtime],gU[:,jr,ir]-gUg[:,jr,ir],bounds_error=False, fill_value=None)
finterpV = scipy.interpolate.RegularGridInterpolator([gtime],gV[:,jr,ir]-gVg[:,jr,ir],bounds_error=False, fill_value=None)
# wind stress
finterpTAx = scipy.interpolate.RegularGridInterpolator([gtime],gTAx[:,jr,ir],bounds_error=False, fill_value=None)
finterpTAy = scipy.interpolate.RegularGridInterpolator([gtime],gTAy[:,jr,ir],bounds_error=False, fill_value=None)
# MLD
finterpMLD = scipy.interpolate.RegularGridInterpolator([gtime],gMLD[:,jr,ir],bounds_error=False, fill_value=None)
# Coriolis value at jr,ir
fc = 2*2*np.pi/86164*np.sin(glat2[jr,ir]*np.pi/180)

# interpolate on reconstructed grid times
U = finterpU(time)
V = finterpV(time)
TAx = finterpTAx(time)
TAy = finterpTAy(time)
MLD = finterpMLD(time)


# building "observations" from MITgcm
Uo = U*np.nan
Vo = V*np.nan
Uo[::86400//dt] = U[::86400//dt]
Vo[::86400//dt] = V[::86400//dt]

# qualité de la représentation par le modèle
# -> permet de donner du poids aux erreurs modèles (le modèle représente d'autres truc que l'inertiel)
# -> Si fonction cout = defaut, pas d'influence.
Ri=Uo*0.+1 

# inverse problem
# verification of tangeant linear func with adjoint.
eps=1e-8
if False:
    Ua, Va = unstek(time, fc, TAx,TAy, pk)
    Ua1,Va1 = unstek(time, fc, TAx,TAy, pk+eps*pk)

    dUa, dVa = unstek_tgl(time, fc, TAx,TAy, pk, pk)

    print(dUa)
    print((Ua1-Ua)/eps)

    Ua, Va = unstek(time, fc, TAx,TAy, pk)
    X=+pk

    MX = unstek_tgl(time, fc, TAx, TAy, pk, X)
    Y = [Ua,Va]
    MtY =   (time, fc, TAx, TAy, pk, Y)

    print(np.sum(MX[0]*Y[0]+MX[1]*Y[1]))
    print(np.sum(X*MtY))


#################################################################################
dpi=200
maxiter=100

# minimization procedure of the cost function
if False:
    Nlayers = len(pk)//2
    print('* Minimization with '+str(Nlayers)+' layers')
    
    J = cost(pk, time, fc, TAx, TAy, Uo, Vo, Ri)
    dJ = grad_cost(pk, time, fc, TAx, TAy, Uo, Vo, Ri)

    res = opt.minimize(cost, pk, args=(time, fc, TAx, TAy, Uo, Vo, Ri),
                    method='L-BFGS-B',
                    jac=grad_cost,
                    options={'disp': True, 'maxiter': maxiter})
    
    if np.isnan(cost(res['x'], time, fc, TAx, TAy, Uo, Vo, Ri)):
        print('The model has crashed.')
    else:
        print(' vector K solution ('+str(res.nit)+' iterations)',res['x'])
        print(' cost function value with K solution:',cost(res['x'], time, fc, TAx, TAy, Uo, Vo, Ri))
    # vector K solution: [-3.63133021 -9.46349552]
    # cost function value with K solution: 0.36493136309782287


    # using the value of K from minimization, we get currents
    Ua, Va = unstek(time, fc, TAx,TAy, res['x'])

    plt.figure(figsize=(10,3),dpi=dpi)
    plt.plot(time/86400,U, c='k', lw=2, label='LLC ref ageostrophic zonal')
    plt.plot(time/86400,Ua, c='g', label='Unsteady-Ekman zonal reconstructed from wind')
    plt.scatter(time/86400,Uo, c='r', label='obs')
    # plt.axis([31.5,60, -0.7,1.2])
    plt.xlabel('Time (days)')
    plt.ylabel('Zonal current (m/s)')
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig(path_save_png+'series_reconstructed_long_'+str(Nlayers)+'layers.png')

# plotting tool for trajectory of 'unstek' for specific vector_k
if False:
    print('* Plotting trajectories/step response for different vector K')
    list_k = [[-3.63133021,  -9.46349552], # 1 layer, minimization from [-3,-12]
              [-5,-12], # 1 layer, hand chosen
              [-2,-7], # 1 layer, hand chosen
              [ -3.47586165, -9.06189063, -11.22302904, -12.43724667], # 2 layer, minimization from [-3,-8,-10,-12]
              ]
    
    A_wind = 10 # m/s
    indicator = np.zeros(len(time))
    indicator[10:] = 1
    step_stressY = 8e-6*np.sign(A_wind)*(indicator*A_wind)**2
    step_stressX = np.zeros(len(time))
    
    for vector_k in list_k:
        print('     ',vector_k)
        Ua, Va = unstek(time, fc, TAx,TAy, vector_k)
        U_step,V_step = unstek(time, fc, step_stressX,step_stressY, vector_k)
        txt_add = ''
        title = ''
        for k in range(len(vector_k)):
            txt_add += '_k'+str(k)+str(vector_k[k])
            title += 'k'+str(k)+','    
        title = title[:-1]+' = '+str(vector_k)
        
        # trajectory
        plt.figure(figsize=(10,3),dpi=dpi)
        plt.plot(time/86400,U, c='k', lw=2, label='LLC ref ageostrophic zonal')
        plt.plot(time/86400,Ua, c='g', label='Unsteady-Ekman zonal reconstructed from wind')
        plt.scatter(time/86400,Uo, c='r', label='obs')
        plt.xlabel('Time (days)')
        plt.ylabel('Zonal current (m/s)')
        plt.title(title)
        plt.legend(loc=1)
        plt.tight_layout()
        plt.savefig(path_save_png+'series_reconstructed_'+str(len(vector_k)//2)+'layers'+txt_add+'.png')

        # step response
        fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
        ax.plot(U_step,V_step,c='k')
        #ax.scatter(U_step,V_step,marker='x',c='k')
        ax.set_xlabel(r'U$_{E}$ (m/s)')
        ax.set_ylabel(r'V$_{E}$ (m/s)')
        ax.set_xlim([0,0.15])
        ax.set_ylim([0,0.15])
        fig.savefig(path_save_png+'Hodograph_from_stepTau_'+str(len(vector_k)//2)+'layers'+txt_add+'.png')
    
# looking at the cost function
#   for a slab model (1 layer)
if True:
    print('* Looking at cost function for k0,k1 in 1 layer model')
    
    PLOT_ITERATIONS = True
    tested_values = np.arange(-15,0,0.25)  # -15,0,0.25
    maxiter = 15
    vector_k = np.array([-12,-6]) # initial vector k
    Jmap_cmap = 'terrain'
    #vector_k = np.array([-3,-12])
    
    # map of cost function
    J = np.zeros((len(tested_values),len(tested_values)))*np.nan
    for i,k0 in enumerate(tested_values):
        for j,k1 in enumerate(tested_values):
            vector_k0k1 = np.array([k0,k1])
            J[j,i] = cost(vector_k0k1, time, fc, TAx, TAy, Uo, Vo, Ri)
    
    # plotting iterations of minimization
    if PLOT_ITERATIONS:
        cost_iter = np.zeros(maxiter)
        vector_k_iter = np.zeros((len(vector_k),maxiter))
        vector_k_iter[:,0] = vector_k
        txt_add = 'k0'+str(vector_k[0])+'_k1'+str(vector_k[1])
        for k in range(1,maxiter):
            res = opt.minimize(cost, vector_k, args=(time, fc, TAx, TAy, Uo, Vo, Ri),
                        method='L-BFGS-B',
                        jac=grad_cost,
                        options={'disp': True, 'maxiter': k})
            vector_k_iter[:,k] = res['x']
            cost_iter[k] = cost(vector_k_iter[:,k], time, fc, TAx, TAy, Uo, Vo, Ri)
    else:
        txt_add = ''

    # PLOTTING
    cmap = mpl.colormaps.get_cmap(Jmap_cmap)
    cmap.set_bad(color='indianred')
    fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
    s = ax.pcolormesh(tested_values,tested_values,J,cmap=cmap,norm=mpl.colors.LogNorm(0.1,100)) #,vmin=0.1,vmax=100
    plt.colorbar(s,ax=ax)
    if PLOT_ITERATIONS:
        plt.scatter(vector_k_iter[0,:],vector_k_iter[1,:],c=['r']+['k']*(len(cost_iter[:])-2)+['g'],marker='x') # ,s=0.1
    ax.set_xlabel('log(k1)')
    ax.set_ylabel('log(k0)')
    plt.savefig(path_save_png+'k0k1_J_1layer'+txt_add+'.png')
    

# plot de J(k1,k2), à k0 et k3 fixés à leur valeur convergée
if False:
    print('* Looking at cost function for k1,k2 in 2 layer model')
    # k0 a un effet linéaire (facteur de tau)
    # k3 est juste en fonction du fond
    # les 2 autres font intervenir les autres couches
    
    # LAYER DEFINITION
    # -> number of values = number of layers
    # -> values = turbulent diffusion coefficient
    """
       vecteur solution :  [ -3.47586165  -9.06189063 -11.22302904 -12.43724667]
       cost function value : 0.28825167461378703
    """
    k0 = -3.47586165
    k3 = -12.43724667

    tested_values = np.arange(-15,-4,0.25)
    J = np.zeros((len(tested_values),len(tested_values)))
    for i,k1 in enumerate(tested_values):
        for j,k2 in enumerate(tested_values):
            vector_k = np.array([k0,k1,k2,k3])
            J[j,i] = cost(vector_k, time, fc, TAx, TAy, Uo, Vo, Ri)
    
    fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
    s = ax.pcolormesh(tested_values,tested_values,J,cmap='plasma_r',norm=matplotlib.colors.LogNorm(0.1,1000)) #,vmin=0.1,vmax=100
    plt.colorbar(s,ax=ax)
    ax.scatter(-9.06189063, -11.22302904,marker='x',c='g')
    ax.set_xlabel('log(k1)')
    ax.set_ylabel('log(k2)')
    plt.savefig(path_save_png+'k1k2_J_2layer.png')


end = clock.time()
print('Total execution time = '+str(np.round(end-start,2))+' s')
plt.show()

#stop
