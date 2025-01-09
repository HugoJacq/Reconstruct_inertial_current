"""
This file is the main file of the study 'Inertial_current_from_MITgcm'

python reconstruct_intertial.py

by: Hugo Jacquet and Clément Ubelmann

Updates: 
    - CU 6/01/25: creation 
    - HJ 7/01/25: add comments everywhere
    - HJ 9/01/25: add build large scale variable file
"""
from model_unstek import *
from build_LS_fields import *
import time as clock
import scipy.optimize as opt
from datetime import datetime, timedelta
start = clock.time()


##############
# PARAMETERS #
##############
# -> area of interest
box= [-25, -24, 45., 46, 24000., 24010.] # bottom lat, top lat, left lon, right lon
# -> spatial location. We work in 1D
ir=4
jr=4

# -> MINIMIZATION OF THE UNSTEADY EKMAN MODEL
dt=60 # timestep of the model
# LAYER DEFINITION
#        number of values = number of layers
#        values = turbulent diffusion coefficient
# pk=np.array([-3,-12])         # 1 layers
pk=np.array([-3,-8,-10,-12])    # 2 layers
# pk=np.array([-2,-4,-6,-8,-10,-12]) # 3 layers
MINIMIZE = True            # find the vector K starting from pk
maxiter=100                 # number of iteration max for MINIMIZE
ONE_LAYER_COST_MAP = False  # maps the cost function values
TWO_LAYER_COST_MAP_K1K2 = False # maps the cost function values, K0 K4 fixed

# -> PLOT
dpi=200

# -> List of files
filesUV = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/U_V/llc2160_2020-11-*_SSU-SSV.nc"))
filesH = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/SSH/llc2160_2020-11-*_SSH.nc"))
filesW = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/v10m/llc2160_2020-11-*_v10m.nc"))
filesD = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/KPPhbl/llc2160_2020-11-*_KPPhbl.nc"))

# END PARAMETERS #################################



# LARGE SCALE FIELDS --------
print('* Getting large scale motion ...')
path_save = './LS_fields'
#build_LS_files(filesUV,box, my2dfilter, mytimefilter, path_save)
build_LS_files(filesH, box, path_save)
#build_LS_files(filesW,box, my2dfilter, mytimefilter, path_save)
#build_LS_files(filesD,box, my2dfilter, mytimefilter, path_save)
dsSSH_LS = xr.open_dataset('LS_fields_SSH_'+str(box[0])+'_'+str(box[1])+'_'+str(box[2])+'_'+str(box[3])+'_'+str(box[4])+'_'+str(box[5])+'.nc')
gUg,gVg = dsSSH_LS['Ug'],dsSSH_LS['Vg']
# ---------------------------
# FILE OPENING
dsUV = xr.open_mfdataset(filesUV)
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

# Fixed values
nt,ny,nx = np.shape(gU) # shape of gridded U current
gtime = np.arange(0,nt)*3600 # time array
glon2,glat2 = np.meshgrid(glon,glat) # lat lon
fc = 2*2*np.pi/86164*np.sin(glat2[jr,ir]*np.pi/180) # Coriolis value at jr,ir

# stress as: C x wind**2
gTAx = 8e-6*np.sign(gTx)*gTx**2
gTAy = 8e-6*np.sign(gTy)*gTy**2

# ---------------------------
## INTERPOLATION 
# This is interpolation on reconstructed time grid (every dt)
# this is necessary for the simple model 'unstek'

print('* interpolation on unsteady ekman model timestep')

interp_at_model_t_1D(dt, ir, jr, filesUV, box, path_save)
interp_at_model_t_1D(dt, ir, jr, filesH, box, path_save)
interp_at_model_t_1D(dt, ir, jr, filesW, box, path_save)
interp_at_model_t_1D(dt, ir, jr, filesD, box, path_save)

raise Exception

time = np.arange(gU.time.values[0], gU.time.values[-1], timedelta(seconds=dt),dtype='datetime64[ns]')
print(len(time))

U = (gU - gUg)[:,jr,ir].interp({'time':time})
print(U)
U = U.values
V = (gV - gVg)[:,jr,ir].interp({'time':time}).values
TAx = gTAx[:,jr,ir].interp({'time':time}).values
TAy = gTAy[:,jr,ir].interp({'time':time}).values
MLD = gMLD[:,jr,ir].interp({'time':time}).values

#time = np.arange(0,gtime[-1]+dt,dt)

# # ageostrophic current
# finterpU = scipy.interpolate.RegularGridInterpolator([gtime],gU[:,jr,ir]-gUg[:,jr,ir],bounds_error=False, fill_value=None)
# finterpV = scipy.interpolate.RegularGridInterpolator([gtime],gV[:,jr,ir]-gVg[:,jr,ir],bounds_error=False, fill_value=None)
# # wind stress
# finterpTAx = scipy.interpolate.RegularGridInterpolator([gtime],gTAx[:,jr,ir],bounds_error=False, fill_value=None)
# finterpTAy = scipy.interpolate.RegularGridInterpolator([gtime],gTAy[:,jr,ir],bounds_error=False, fill_value=None)
# # MLD
# finterpMLD = scipy.interpolate.RegularGridInterpolator([gtime],gMLD[:,jr,ir],bounds_error=False, fill_value=None)

# interpolate on reconstructed grid times
# U = finterpU(time)
# V = finterpV(time)
# TAx = finterpTAx(time)
# TAy = finterpTAy(time)
# MLD = finterpMLD(time)
raise Exception
# -----------------------------------------------
# OBSERVATIONS from MITgcm
Uo = U*np.nan
Vo = V*np.nan
Uo[::86400//dt] = U[::86400//dt]
Vo[::86400//dt] = V[::86400//dt]
# -----------------------------------------------
# INVERSE PROBLEM

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

# minimization procedure of the cost function (n layers)
if MINIMIZE:
    Nlayers = len(pk)//2
    print('* Minimization with '+str(Nlayers)+' layers')
    
    J = cost(pk, time, fc, TAx, TAy, Uo, Vo, Ri)
    dJ = grad_cost(pk, time, fc, TAx, TAy, Uo, Vo, Ri)

    res = opt.minimize(cost, pk, args=(time, fc, TAx, TAy, Uo, Vo, Ri),
                    method='L-BFGS-B',
                    jac=grad_cost,
                    options={'disp': True, 'maxiter': maxiter})
    

    print('     vector K solution ('+str(res.nit)+' iterations)',res['x'])
    print('     cost function value with K solution:',cost(res['x'], time, fc, TAx, TAy, Uo, Vo, Ri))
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
    plt.savefig('series_reconstructed_long_'+str(Nlayers)+'layers.png')
    
# looking at the cost function
#   for a slab model (1 layer)
if ONE_LAYER_COST_MAP:
    print('* Looking at cost function for k0,k1 in 1 layer model')
   
    maxiter = 15
    cost_iter = np.zeros(maxiter)
    vector_k = np.array([-3,-12])
    vector_k_iter = np.zeros((len(vector_k),maxiter))
    vector_k_iter[:,0] = vector_k
    for k in range(maxiter):
        res = opt.minimize(cost, vector_k, args=(time, fc, TAx, TAy, Uo, Vo, Ri),
                    method='L-BFGS-B',
                    jac=grad_cost,
                    options={'disp': True, 'maxiter': k})
    
        vector_k_iter[:,k] = res['x']
        cost_iter[k] = cost(vector_k_iter[:,k], time, fc, TAx, TAy, Uo, Vo, Ri)

    # LAYER DEFINITION
    # -> number of values = number of layers
    # -> values = turbulent diffusion coefficient
    tested_values = np.arange(-13,-2,0.25)
    J = np.zeros((len(tested_values),len(tested_values)))
    for i,k0 in enumerate(tested_values):
        for j,k1 in enumerate(tested_values):
            vector_k = np.array([k0,k1])
            J[j,i] = cost(vector_k, time, fc, TAx, TAy, Uo, Vo, Ri)
    
    

    fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
    s = ax.pcolormesh(tested_values,tested_values,J,cmap='plasma_r',norm=matplotlib.colors.LogNorm(0.1,100)) #,vmin=0.1,vmax=100
    plt.colorbar(s,ax=ax)
    plt.scatter(vector_k_iter[0,:],vector_k_iter[1,:],c=['r']+['k']*(len(cost_iter[:])-2)+['g'],marker='x') # ,s=0.1
    ax.set_xlabel('log(k0)')
    ax.set_ylabel('log(k1)')
    plt.savefig('k0k1_J_1layer.png')
    
# plot de J(k1,k2), à k0 et k3 fixés à leur valeur convergée
if TWO_LAYER_COST_MAP_K1K2:
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
    plt.savefig('k1k2_J_2layer.png')

end = clock.time()
print('Total execution time = '+str(np.round(end-start,2))+' s')
plt.show()