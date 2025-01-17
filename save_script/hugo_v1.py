import numpy
import matplotlib.pylab as plt
import pdb 
import glob
from netCDF4 import Dataset
import scipy
import scipy.signal
import scipy.interpolate
import scipy.constants
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker
from datetime import datetime, timedelta
import scipy.optimize as opt
import xarray as xr 

def my2dfilter(s,sigmax,sigmay, ns=2):
    x, y = numpy.meshgrid(numpy.arange(-int(ns*sigmax), int(ns*sigmax)+1), numpy.arange(-int(ns*sigmay), int(ns*sigmay)+1), indexing='ij')
    cf=numpy.exp(-(x**2/sigmax**2+y**2/sigmay**2))
    m=~numpy.isnan(s)*1.
    s[numpy.isnan(s)]=0.
    s_sum = (scipy.signal.convolve2d(s, cf, mode='same'))
    w_sum = (scipy.signal.convolve2d(m, cf, mode='same'))
    sf = s*0.
    sf[w_sum!=0] = s_sum[w_sum!=0] / w_sum[w_sum!=0]
    return sf




def unstek(time, fc, TAx, TAy, k, return_traj=False):

    K = numpy.exp(k)

    nl = int(len(K)//2)
    U = [None]*nl
    for ik in range(nl):
        U[ik]=numpy.zeros((len(time)), dtype='complex')

    TA = TAx + 1j*TAy

    dt=time[1]-time[0]

    for it in range(len(time)-1):
        for ik in range(nl):
            if ((ik==0)&(ik==nl-1)): U[ik][it+1] = U[ik][it] + dt*( -1j*fc*U[ik][it] +K[2*ik]*TA[it] - K[2*ik+1]*(U[ik][it]) )
            else:
                if ik==0: U[ik][it+1] = U[ik][it] + dt*( -1j*fc*U[ik][it] +K[2*ik]*TA[it] - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) )
                elif ik==nl-1: U[ik][it+1] = U[ik][it] + dt*( -1j*fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*U[ik][it] )
                else: U[ik][it+1] = U[ik][it] + dt*( -1j*fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) )

    if return_traj: return U
    else: return numpy.real(U[0]), numpy.imag(U[0])


def unstek_tgl(time, fc, TAx, TAy, k, dk):

    K = numpy.exp(k)
    dK = numpy.exp(k)*dk

    nl = int(len(K)//2)
    U = [None]*nl
    dU = [None]*nl
    for ik in range(nl):
        U[ik]=numpy.zeros((len(time)), dtype='complex')
        dU[ik]=numpy.zeros((len(time)), dtype='complex')
    

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

    return numpy.real(dU[0]), numpy.imag(dU[0])



def unstek_adj(time, fc, TAx, TAy, k, ad_Y):


    K = numpy.exp(k)

    U = unstek(time, fc, TAx, TAy, k, return_traj=True)

    nl = int(len(k)//2)
    ad_U = [None]*nl
    for ik in range(nl):
        ad_U[ik]=numpy.zeros((len(time)), dtype='complex')
    ad_U[0] = ad_Y[0] + 1j*ad_Y[1]

    TA = TAx + 1j*TAy

    ad_K = numpy.zeros((len(k)), dtype='complex')

    for it in numpy.arange(len(time)-1)[::-1]:
        for ik in range(nl):
            if ((ik==0)&(ik==nl-1)): 
                ad_U[ik][it] += ad_U[ik][it+1]*numpy.conjugate(1+dt*( -1j*fc- K[2*ik+1]))
                ad_K[2*ik] += dt*(ad_U[ik][it+1]*numpy.conjugate(TA[it]))
                ad_K[2*ik+1] += dt*(-ad_U[ik][it+1]*numpy.conjugate(U[ik][it]))

                #dU[ik][it+1] = dU[ik][it] + dt*( -1j*fc*dU[ik][it] +dK[2*ik]*TA[it] - dK[2*ik+1]*(U[ik][it]) - K[2*ik+1]*(dU[ik][it]) )
            else:
                if ik==0: 
                    ad_U[ik][it] += ad_U[ik][it+1]*numpy.conjugate(1+dt*( -1j*fc- K[2*ik+1])) 
                    ad_U[ik+1][it] += ad_U[ik][it+1]*numpy.conjugate(dt*K[2*ik+1])
                    ad_K[2*ik] += dt*(ad_U[ik][it+1]*numpy.conjugate(TA[it])) 
                    ad_K[2*ik+1] += dt*(-ad_U[ik][it+1]*numpy.conjugate(U[ik][it]-U[ik+1][it]))
                    #dU[ik][it+1] = dU[ik][it] + dt*( -1j*fc*dU[ik][it] +dK[2*ik]*TA[it] - dK[2*ik+1]*(U[ik][it]-U[ik+1][it]) - K[2*ik+1]*(dU[ik][it]-dU[ik+1][it]))
                elif ik==nl-1: 
                    ad_U[ik][it] += ad_U[ik][it+1]*numpy.conjugate(1+dt*( -1j*fc- K[2*ik]- K[2*ik+1]))
                    ad_U[ik-1][it] += ad_U[ik][it+1]*numpy.conjugate(dt*K[2*ik])
                    ad_K[2*ik] += ad_U[ik][it+1]*(-dt*numpy.conjugate(U[ik][it]-U[ik-1][it]))
                    ad_K[2*ik+1] += ad_U[ik][it+1]*dt*numpy.conjugate(-U[ik][it])
                    #dU[ik][it+1] = dU[ik][it] + dt*( -1j*fc*dU[ik][it] -dK[2*ik]*(U[ik][it]-U[ik-1][it]) -K[2*ik]*(dU[ik][it]-dU[ik-1][it])    - dK[2*ik+1]*U[ik][it] - K[2*ik+1]*dU[ik][it])
                else: 
                    ad_U[ik][it] += ad_U[ik][it+1]*numpy.conjugate(1+dt*( -1j*fc- K[2*ik]- K[2*ik+1]))
                    ad_U[ik+1][it] += ad_U[ik][it+1]*numpy.conjugate(dt*K[2*ik+1])
                    ad_U[ik-1][it] += ad_U[ik][it+1]*numpy.conjugate(dt*K[2*ik])
                    ad_K[2*ik] += ad_U[ik][it+1]*numpy.conjugate(-dt*(U[ik][it]-U[ik-1][it]))
                    ad_K[2*ik+1] += dt*(-ad_U[ik][it+1]*numpy.conjugate(U[ik][it]-U[ik+1][it]))

                    #dU[ik][it+1] = dU[ik][it] + dt*( -1j*fc*dU[ik][it] -dK[2*ik]*(U[ik][it]-U[ik-1][it]) -K[2*ik]*(dU[ik][it]-dU[ik-1][it]) - dK[2*ik+1]*(U[ik][it]-U[ik+1][it]) - K[2*ik+1]*(dU[ik][it]-dU[ik+1][it]) )


    ad_k = numpy.exp(k)*ad_K

    return numpy.real(ad_k)

  #######################################################



def cost(pk, time, fc, TAx, TAy, Uo, Vo, Ri):

    U, V = unstek(time, fc, TAx, TAy, pk)

    J = 0.5 * numpy.nansum( ((Uo - U)*Ri)**2 + ((Vo - V)*Ri)**2 )

    return J

def grad_cost(pk, time, fc, TAx, TAy, Uo, Vo, Ri):  

    U, V = unstek(time, fc, TAx, TAy, pk)

    sensU = (Uo -U)*Ri
    sensV = (Vo -V)*Ri
    sensU[numpy.isnan(sensU)]=0.
    sensV[numpy.isnan(sensV)]=0.

    dJ_pk = unstek_adj(time, fc, TAx, TAy, pk, [sensU,sensV])

    return -dJ_pk


box= [-25, -24, 45., 46, 24000., 24010.]



# files = numpy.sort(glob.glob("/data2/nobackup/clement/Data/Llc2160/llc2160_daily_latlon_SSC/llc2160_2020-11-*_SSU-SSV.nc"))
# filesH = numpy.sort(glob.glob("/data2/nobackup/clement/Data/Llc2160/llc2160_daily_latlon_SSH/llc2160_2020-11-*_SSH.nc"))
# filesW = numpy.sort(glob.glob("/data2/nobackup/clement/Data/Llc2160/llc2160_daily_latlon_wind/llc2160_2020-11-*_v10m.nc"))
# filesD = numpy.sort(glob.glob("/data2/nobackup/clement/Data/Llc2160/llc2160_daily_latlon_KPPhbl/llc2160_2020-11-*_KPPhbl.nc"))

files = numpy.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/U_V/llc2160_2020-11-*_SSU-SSV.nc"))
filesH = numpy.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/SSH/llc2160_2020-11-*_SSH.nc"))
filesW = numpy.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/v10m/llc2160_2020-11-*_v10m.nc"))
filesD = numpy.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/KPPhbl/llc2160_2020-11-*_KPPhbl.nc"))
for it in range(len(files)):
    with Dataset(files[it], 'r') as fcid:
        if it==0:
            glon = numpy.array(fcid.variables['lon'][:])
            glat = numpy.array(fcid.variables['lat'][:])
            ix = numpy.where((glon>=box[0])&(glon<=box[1]))[0]
            iy = numpy.where((glat>=box[2])&(glat<=box[3]))[0]
            glon=glon[ix[0]:ix[-1]+1]
            glat=glat[iy[0]:iy[-1]+1]
            gU = numpy.zeros((len(files)*24,len(iy),len(ix))) + numpy.nan
            gV = numpy.zeros((len(files)*24,len(iy),len(ix))) + numpy.nan
            gTx = numpy.zeros((len(files)*24,len(iy),len(ix))) + numpy.nan
            gTy = numpy.zeros((len(files)*24,len(iy),len(ix))) + numpy.nan            
            gH = numpy.zeros((len(files)*24,len(iy),len(ix))) + numpy.nan
            gMLD = numpy.zeros((len(files)*24,len(iy),len(ix))) + numpy.nan
        gU[it*24:it*24+24,:,:] = numpy.array(fcid.variables['SSU'][:,iy[0]:iy[-1]+1,ix[0]:ix[-1]+1])
        gV[it*24:it*24+24,:,:] = numpy.array(fcid.variables['SSV'][:,iy[0]:iy[-1]+1,ix[0]:ix[-1]+1])
    with Dataset(filesH[it], 'r') as fcid:
        gH[it*24:it*24+24,:,:] = numpy.array(fcid.variables['SSH'][:,iy[0]:iy[-1]+1,ix[0]:ix[-1]+1])
    with Dataset(filesW[it], 'r') as fcid:
        gTx[it*24:it*24+24,:,:] = numpy.array(fcid.variables['geo5_u10m'][:,iy[0]:iy[-1]+1,ix[0]:ix[-1]+1])  
        gTy[it*24:it*24+24,:,:] = numpy.array(fcid.variables['geo5_v10m'][:,iy[0]:iy[-1]+1,ix[0]:ix[-1]+1])     
    with Dataset(filesD[it], 'r') as fcid:
        gMLD[it*24:it*24+24,:,:] = numpy.array(fcid.variables['KPPhbl'][:,iy[0]:iy[-1]+1,ix[0]:ix[-1]+1])          

gU[gU>1e5]=numpy.nan
gV[gV>1e5]=numpy.nan
gTx[gTx>1e5]=numpy.nan
gTy[gTy>1e5]=numpy.nan
gH[gH>100]=numpy.nan

nt,ny,nx=numpy.shape(gU)

#time_conv = numpy.arange(-3*86400,3*86400+dt,dt) A REMETTRE
time_conv = numpy.arange(-1*86400,1*86400+3600,3600)

taul = numpy.zeros((len(iy),len(ix))) + 1*86400
# gl = (numpy.exp(-1j*numpy.outer(fc[:],time_conv))*numpy.exp(-numpy.outer(taul**-2,time_conv**2))).reshape(len(time),len(time_conv))
gl = numpy.exp(numpy.outer(-taul**-2,time_conv**2))
gl = (gl.T / numpy.sum(numpy.abs(gl), axis=1).T)

Hf0 = gH*0.
Uf0 = gU*0.
Vf0 = gV*0.
for it in range(nt):
    Hf0[it,:,:] = my2dfilter(gH[it,:,:],3,3)
    Uf0[it,:,:] = my2dfilter(gU[it,:,:],3,3)
    Vf0[it,:,:] = my2dfilter(gV[it,:,:],3,3)

Hf = gH*0.
Uf = gU*0.
Vf = gV*0.
taul = 2*86400
gl = numpy.exp(-taul**-2 * time_conv**2)
gl = (gl / numpy.sum(numpy.abs(gl)))
for ix in range(nx):
    for iy in range(ny):
        Hf[:,iy,ix] = numpy.convolve(Hf0[:,iy,ix],gl,'same')
        Uf[:,iy,ix] = numpy.convolve(Uf0[:,iy,ix],gl,'same')
        Vf[:,iy,ix] = numpy.convolve(Vf0[:,iy,ix],gl,'same')

dlon=glon[1]-glon[0]
dlat=glat[1]-glat[0]
glon2,glat2 = numpy.meshgrid(glon,glat)
fc = 2*2*numpy.pi/86164*numpy.sin(glat2*numpy.pi/180)

gUg=Hf*0.
gVg=Hf*0.

for it in range(nt):
    gVg[it,:,1:-1] = 10/fc[:,1:-1]*(Hf[it,:,2:]-Hf[it,:,:-2])/(dlon*110000*numpy.cos(glat2[:,1:-1]*numpy.pi/180))/2
    gUg[it,1:-1,:] = -10/fc[1:-1,:]*(Hf[it,2:,:]-Hf[it,:-2,:])/(dlat*110000)/2

gUg[:,0,:]=gUg[:,1,:]
gUg[:,-1,:]=gUg[:,-2,:]
gVg[:,:,0]=gVg[:,:,1]
gVg[:,:,-1]=gVg[:,:,-2]

gTAx = 8e-6*numpy.sign(gTx)*gTx**2
gTAy = 8e-6*numpy.sign(gTy)*gTy**2


gtime = numpy.arange(0,nt)*3600

dt=60
#time  = numpy.arange(0,gtime[-1]+dt,dt)
time  = numpy.arange(0,gtime[-1],dt)
ir=4
jr=4

finterpU = scipy.interpolate.RegularGridInterpolator([gtime],gU[:,jr,ir]-gUg[:,jr,ir],bounds_error=False, fill_value=None)
finterpV = scipy.interpolate.RegularGridInterpolator([gtime],gV[:,jr,ir]-gVg[:,jr,ir],bounds_error=False, fill_value=None)

finterpTAx = scipy.interpolate.RegularGridInterpolator([gtime],gTAx[:,jr,ir],bounds_error=False, fill_value=None)
finterpTAy = scipy.interpolate.RegularGridInterpolator([gtime],gTAy[:,jr,ir],bounds_error=False, fill_value=None)

finterpMLD = scipy.interpolate.RegularGridInterpolator([gtime],gMLD[:,jr,ir],bounds_error=False, fill_value=None)

fc = 2*2*numpy.pi/86164*numpy.sin(glat2[jr,ir]*numpy.pi/180)


U = finterpU(time)
#V = finterpV(time)
TAx = finterpTAx(time)
#TAy = finterpTAy(time)
#MLD = finterpMLD(time)


# exp
ds_i = xr.open_dataset('../Interp_1D_LON-24.8_LAT45.2.nc')
ds_LS = xr.open_dataset('../LS_fields_SSH_-25_-24_45.0_46_24000.0_24010.0.nc')
TAx2 = ds_i.TAx.values
TAx_x32 = numpy.float32(TAx)
print(TAx2.dtype,TAx_x32.dtype)
print('TAx vs TAx2',numpy.allclose(TAx,TAx_x32))
print('distance moyenne TAx et TAx2', numpy.mean( TAx-TAx_x32))
Ug2 = ds_LS.Ug
print('Ug vs Ug2',numpy.array_equal(gUg,Ug2))
print('distance moyenne Ug et Ug2', numpy.mean( gUg-Ug2))
U2 = ds_i.SSU
print('Ug vs Ug2',numpy.array_equal(U,U2))
print('distance moyenne Ug et Ug2', numpy.mean( U-U2))
ULS = ds_LS.U_LS
print('ULS vs Uf',numpy.array_equal(ULS,Uf))
print('distance moyenne Ug et Ug2', numpy.mean( ULS-Uf))

# I want to check differences between Clément code and mine
# -> OK  position lat lon
# -> LS field (spatial and time smoothing)
# -> time interpolation on unstek model


raise Exception
#



Uo = U*numpy.nan
Vo = V*numpy.nan
Uo[::86400//dt] = U[::86400//dt]
Vo[::86400//dt] = V[::86400//dt]

Ri=Uo*0.+1



eps=1e-8
# pk=numpy.array([-3,-8,-10,-12])
pk=numpy.array([-3,-12])

Ua, Va = unstek(time, fc, TAx,TAy, pk)
Ua1,Va1 = unstek(time, fc, TAx,TAy, pk+eps*pk)


dUa, dVa = unstek_tgl(time, fc, TAx,TAy, pk, pk)

print(dUa)
print((Ua1-Ua)/eps)

Ua, Va = unstek(time, fc, TAx,TAy, pk)
X=+pk

MX = unstek_tgl(time, fc, TAx, TAy, pk, X)
Y = [Ua,Va]
MtY = unstek_adj(time, fc, TAx, TAy, pk, Y)

print(numpy.sum(MX[0]*Y[0]+MX[1]*Y[1]))
print(numpy.sum(X*MtY))


#################################################################################

J = cost(pk, time, fc, TAx, TAy, Uo, Vo, Ri)

dJ = grad_cost(pk, time, fc, TAx, TAy, Uo, Vo, Ri)

maxiter=100
res = opt.minimize(cost, pk, args=(time, fc, TAx, TAy, Uo, Vo, Ri),
                   method='L-BFGS-B',
                   jac=grad_cost,
                   options={'disp': True, 'maxiter': maxiter})


Ua, Va = unstek(time, fc, TAx,TAy, res['x'])





plt.close()
plt.figure(figsize=(10,3))
plt.plot(time,U, c='k', lw=2, label='LLC ref ageostrophic zonal')
plt.plot(time,Ua, c='g', label='Unsteady-Ekman zonal reconstructed from wind')
plt.scatter(time,Uo, c='r', label='obs')
# plt.axis([31.5,60, -0.7,1.2])
plt.xlabel('Time (days)')
plt.ylabel('Zonal current (m/s)')
plt.legend(loc=1)
plt.tight_layout()
plt.savefig('series_reconstructed_long.png')


#stop
