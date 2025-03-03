"""
This file is the main file of the study 'Inertial_current_from_MITgcm'

python reconstruct_intertial.py

by: Hugo Jacquet and Clément Ubelmann

Updates: 
    - CU 6/01/25: creation 
    - HJ 7/01/25: add comments everywhere
    - HJ 9/01/25: add build large scale variable file
"""
import os
#os.environ["OMP_NUM_THREADS"] = "1" # force mono proc
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '.125' # for jax, percentage of pre allocated GPU mem
#print(os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"])
#print(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"])
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
#os.environ['XLA_GPU_MEMORY_LIMIT_SLOP_FACTOR'] = '50'

import time as clock
import scipy.optimize as opt
from dask.distributed import Client,LocalCluster
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import glob

#from optimparallel import minimize_parallel
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import ticker as mticker
#import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_platform_name', 'cpu')



# custom imports
sys.path.append('./src/')
from src.OSSE import *
from src.observations import *
from src.forcing import *
from src.unstek import *
from src.junstek import *
from src.inv import *
from src.tools import *
from src.scores import *
from src.benchmark import *

start = clock.time()

###############################################
# PARAMETERS                                  #
###############################################
# //
DASHBOARD   = True     # when using dask
N_CPU       = 8         # when using joblib, if >1 then use // code
JAXIFY      = True      # whether to use JAX or not
ON_HPC      = False      # on HPC

# -> area of interest
# 1D
point_loc_source = {'MITgcm':[-24.8,45.2], # °E,°N
                    'Croco':[-50.,35.]}
# 2D
LON_bounds = [-50.2,-49.8]
LAT_bounds = [34.8,35.2]

# -> Observations : OSSE
SOURCE              = 'Croco'   # MITgcm Croco
TRUE_WIND_STRESS    = True      # whether to use Cd.U**2 or Tau
dt                  = 60        # timestep of the model (s) 
dt_OSSE             = 3600      # timestep of the OSSE (s)
period_obs          = 86400     # s, how many second between observations
MODE                = '2D'      # '1D' or '1D_kt' or '2D'

# -> LAYER DEFINITION
#        number of values = number of layers
#        values = turbulent diffusion coefficient
#        enter float values !!
#vector_k = np.array([-9.,-11.]) # [-3,-12]         # 1 layers
vector_k=np.array([-9.,-10.,-11.,-12.])    # 2 layers

# -> MINIMIZATION OF THE UNSTEADY EKMAN MODEL
PRINT_INFO              = False     # only if PARALLEL_MINIMIZED
save_iter               = False     # save iteration during minimize
maxiter                 = 100       # max iteration of minimization
PARALLEL_MINIMIZED      = False     # there is a bug with TRUE
PRINT_INFO              = False     # show info during minimization

# -> ANALYSIS   
MAP_1D_LOCATION         = False     # plot a map to show where we are working
MINIMIZE                = False     # find the vector K starting from 'pk'
PLOT_TRAJECTORY         = False     # plot u(t) for a specific vector_k
ONE_LAYER_COST_MAP      = False     # maps the cost function values
TWO_LAYER_COST_MAP_K1K2 = False     # maps the cost function values, K0 K4 fixed
LINK_K_AND_PHYSIC       = False     # link the falues of vector K with physical variables
CHECK_MINI_HYPERCUBE    = False     # check of minimum, starting at corner of an hypercube
PLOT_CROCO_PROFILES     = False      # WIP: plot the vertical profiles from croco, at point_loc

# tests
TEST_ROTARY_SPECTRA         = False     # implementing rotary spectra
TEST_JUNSTEK1D_KT           = False     # implementing junstek1D_kt
TEST_JUNSTEK1D_KT_SPATIAL   = False     # implementing jUnstek1D_spatial
REGRID_vs_FINE              = True     # compare at 'point_loc' original and regrid 

BENCHMARK_ALL           = False     # performance benchmark

    
# note: i need to tweak score_PSD with rotary spectra

# -> PLOT
dpi=200

# -> List of files
# Jackzilla
if ON_HPC:
    files_dict = {"MITgcm":{'filesUV': np.sort(glob.glob("/data2/nobackup/clement/Data/Llc2160/llc2160_daily_latlon_SSC/llc2160_2020-11-*_SSU-SSV.nc")),
                            'filesH': np.sort(glob.glob("/data2/nobackup/clement/Data/Llc2160/llc2160_daily_latlon_SSH/llc2160_2020-11-*_SSH.nc")),
                            'filesW': np.sort(glob.glob("/data2/nobackup/clement/Data/Llc2160/llc2160_daily_latlon_wind/llc2160_2020-11-*_v10m.nc")),
                            'filesD': np.sort(glob.glob("/data2/nobackup/clement/Data/Llc2160/llc2160_daily_latlon_KPPhbl/llc2160_2020-11-*_KPPhbl.nc")),
                            'filesTau': np.sort(glob.glob("/data2/nobackup/clement/Data/Llc2160/llc2160_daily_latlon_wind/llc2160_2020-11-*_oceTAUX-oceTAUY.nc")),
                            },
                'Croco':{'surface':[#'/data2/nobackup/clement/Data/Lionel_coupled_run/croco_1h_inst_surf_2006-01-01-2006-01-31.nc',
                                    '/data2/nobackup/clement/Data/Lionel_coupled_run/croco_1h_inst_surf_2006-02-01-2006-02-28.nc'],
                            '3D':['/data2/nobackup/clement/Data/Lionel_coupled_run/croco_3h_U_aver_2006-02-01-2006-02-28.nc',
                                '/data2/nobackup/clement/Data/Lionel_coupled_run/croco_3h_V_aver_2006-02-01-2006-02-28.nc']},
                }
    files_dict['MITgcm']['files_sfx'] = ( list(files_dict['MITgcm']['filesUV']) + 
                                    list(files_dict['MITgcm']['filesH']) + 
                                    list(files_dict['MITgcm']['filesW']) + 
                                    list(files_dict['MITgcm']['filesD']) + 
                                    list(files_dict['MITgcm']['filesTau']) )
# Local
else:
    if SOURCE=='MITgcm':
        raise Exception('No data on local laptop, please use HPC or download the data')
    files_dict = {
                # "MITgcm":{'filesUV': np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/U_V/llc2160_2020-11-*_SSU-SSV.nc")),
                #             'filesH': np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/SSH/llc2160_2020-11-*_SSH.nc")),
                #             'filesW': np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/v10m/llc2160_2020-11-*_v10m.nc")),
                #             'filesD': np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/KPPhbl/llc2160_2020-11-*_KPPhbl.nc")),
                #             'filesTau': np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/oceTau/llc2160_2020-11-*_oceTAUX-oceTAUY.nc")),
                #             },
                'Croco':{'surface':'/home/jacqhugo/Datlas_2025/DATA_Crocco/croco_1h_inst_surf_2006-02-01-2006-02-28.nc',
                            '3D':['/home/jacqhugo/Datlas_2025/DATA_Crocco/croco_3h_U_aver_2006-02-01-2006-02-28.nc',
                                '/home/jacqhugo/Datlas_2025/DATA_Crocco/croco_3h_V_aver_2006-02-01-2006-02-28.nc']},
                }    

# concatenate files
files_dict['Croco']['files_sfx'] = files_dict['Croco']['surface']

# -> list of save path
path_save_interped = './'           # where to save interpolated (on model dt) currents

dico_pk_solution ={'MITgcm':{'[-24.8, 45.2]':
                                        {
                                        '1':{'TRUE_TAU'  :[-9.39170447, -9.21727385],
                                            'Cd.UU'     :[-8.44024616, -9.43221639],
                                            
                                            },
                                        '2':{'TRUE_TAU'  :[-9.11293915, -8.73891447, -11.15877101, -12.56505214],
                                            'Cd.UU'     :[-8.26241066,  -9.06349461, -11.31082486, -12.47552063]
                                            } 
                                        }
                                    },
                   'Croco':{ '[-50.0, 35.0]':
                                        {
                                        '1':{'TRUE_TAU'  :[-11.31980127, -10.28525189]
                                             },
                                        '2':{'TRUE_TAU'  :[-10.76035344, -9.3901326, -10.61707124, -12.66052074]
                                             },
                                        '3':{'TRUE_TAU'  :[-8,-9,-10.76035344, -9.3901326, -10.61707124, -12.66052074] # this is WIP
                                             }
                                        }
                                     } 
                            }



# END PARAMETERS #################################
##################################################
# MAIN LOOP
# This avoids infinite subprocess creation
if __name__ == "__main__":  
    
    
    if SOURCE=='Crocco':
        SOURCE='Croco'
    if JAXIFY:
        nameJax = 'JAX_'
    else:
        nameJax = ''
    if SOURCE=='Croco':
        TRUE_WIND_STRESS = True
        
    point_loc = point_loc_source[SOURCE]

    print('')
    print('*********************************')
    print('* SOURCE     = '+SOURCE)
    print('* MODE       = '+MODE)
    if MODE=='1D':
        print('*    LOCATION [W,E]= '+str(point_loc))
    elif MODE=='2D':
        print('*    LAT_BOUNDS (°E) = '+str(LAT_bounds))
        print('*    LON_BOUNDS (°N) = '+str(LON_bounds))
    print('* TRUE WINDSTRESS = '+str(TRUE_WIND_STRESS))
    print('* JAXIFY     = '+str(JAXIFY))
    print('*********************************')
    print('')
    # where to save pngs for 1D study
    path_save_png1D = './png_1D/'
    if not os.path.isdir(path_save_png1D): os.system('mkdir '+path_save_png1D)
    #
    path_save_png1D += SOURCE+'/'
    if not os.path.isdir(path_save_png1D): os.system('mkdir '+path_save_png1D)
    #
    path_save_png1D += 'LON'+str(point_loc[0])+'_LAT'+str(point_loc[1])+'/'
    if not os.path.isdir(path_save_png1D): os.system('mkdir '+path_save_png1D)
    #   
    if TRUE_WIND_STRESS or SOURCE=='Croco': path_save_png1D += 'stress_Tau/'
    else: path_save_png1D += 'stress_UU/'
    if not os.path.isdir(path_save_png1D): os.system('mkdir '+path_save_png1D)


    global client
    client = None
    if DASHBOARD:
        # sometimes dask cluster can cause problems "memoryview is too large"
        # (writing a big netcdf file for eg, hbudget_file)
        cluster = LocalCluster(n_workers=8) # threads_per_worker=1,
        
        client = Client(cluster)
        print("Dashboard at :",client.dashboard_link)
    
    ## INTERPOLATION 
    # This is interpolation on reconstructed time grid (every dt)
    # this is necessary for the simple model 'unstek'
    print('* Interpolation on unsteady ekman model timestep 1D')
    files_list_all = files_dict[SOURCE]['files_sfx']
    model_source = Model_source_OSSE(SOURCE, files_list_all)
        
    interp_at_model_t_1D(model_source, dt, point_loc, N_CPU, path_save_interped)
    #print('* Interpolation on unsteady ekman model timestep 2D')
    # building 2D+time file on Junstek_Kt_spatial time grid
    # this is now done on the fly in junstek1D_Kt_spatial
    # interp_at_model_t_2D(model_source, dt, LON_bounds, LAT_bounds, N_CPU, path_save_interped)
        
    (nameLon_u, nameLon_v, nameLon_rho,
     nameLat_u, nameLat_v, nameLat_rho, 
     nameSSH, nameU, nameV, nameTime,
     nameOceTaux,nameOceTauy)= model_source.get_name_dim()
    
    # observation and forcing
    path_file = SOURCE+'_Interp_1D_LON'+str(point_loc[0])+'_LAT'+str(point_loc[1])+'.nc'
    forcing = Forcing1D_SAVE(dt, path_file, TRUE_WIND_STRESS=True)   # to be modified for Forcing1D
    observations = Observation1D(period_obs, dt, path_file)
    Uo,Vo = observations.get_obs()
    U, V = forcing.U, forcing.V
    
    
        
    # INVERSE PROBLEM
    # verification of tangeant linear func with adjoint.
    # -> permet de donner du poids aux erreurs modèles (le modèle représente d'autres truc que l'inertiel)
    # -> Si fonction cout = defaut, pas d'influence.
    if False:
        print('gradient test')
        pk=np.array([-3,-12])
        
        Nl = len(pk)//2
        model = Unstek1D(Nl, forcing, observations)
        
        eps=1e-8

        _, Ca = model.do_forward(pk)
        Ua, Va = np.real(Ca), np.imag(Ca)

        _, Ca1 = model.do_forward(pk+eps*pk)
        Ua1, Va1 = np.real(Ca1), np.imag(Ca1)
        
        print(Ua)
        print(Ua1)
        
        dCa = model.tgl(pk, pk)
        dUa, dVa = np.real(dCa), np.imag(dCa)
        print(dUa)
        
        
        print((Ua1-Ua)/eps)
        #print(np.sum((Ua1-Ua)))
        
        _, Ca = model.do_forward(pk)
        Ua, Va = np.real(Ca), np.imag(Ca)
        X=+pk

        dU = model.tgl(pk, X)
        MX = [np.real(dU),np.imag(dU)]
        Y = [Ua,Va]
        MtY =  model.adjoint(pk, Y)

        # the two next print should be equal
        print(np.sum(MX[0]*Y[0]+MX[1]*Y[1]))
        print(np.sum(X*MtY))
        # the next print should be << 1
        print( (np.abs(np.sum(MX[0]*Y[0]+MX[1]*Y[1]) - np.sum(X*MtY)))/ np.abs(np.sum(X*MtY)))
        
        raise Exception(' the gradient test is finished')
    # -----------------------------------------------
    
    # ANALYSIS
    #
    # Where am i ?
    if MAP_1D_LOCATION:
        print(" * Plotting the data set around 'point_loc'")
        # box= [-25, # °E
        #       -24, # °E
        #       45, # °N
        #       46]  # °N *
        L_AREA = ['small','large','xlarge']
        
        dict_box = {"small":{'incr':1,
                           'major_xtick':0.5,
                           'minor_xtick':0.1,
                           'grid_tick':1},
                    "large":{'incr':5,
                           'major_xtick':5,
                           'minor_xtick':1,
                           'grid_tick':1},
                    "xlarge":{'incr':10,
                           'major_xtick':5,
                           'minor_xtick':1,
                           'grid_tick':1},
                    }
        for AREA in L_AREA:
            incr = dict_box[AREA]['incr']
            grd_tick = dict_box[AREA]['grid_tick']
            box= [point_loc[0]-incr, # °E
                point_loc[0]+incr, # °E
                point_loc[1]-incr, # °N
                point_loc[1]+incr]  # °N    
            
            step_minor_xtick = dict_box[AREA]['minor_xtick']
            step_minor_ytick = step_minor_xtick
            step_major_xtick = dict_box[AREA]['major_xtick']
            step_major_ytick = step_major_xtick
            
            ds = model_source.dataset
            if model_source.source == 'MITgcm':
                glon,glon_u,glon_v = ds[nameLon_rho],ds[nameLon_rho],ds[nameLon_rho]
                glat,glat_u,glat_v = ds[nameLat_rho],ds[nameLat_rho],ds[nameLat_rho]
                #
                cmapSSH = 'seismic'
                levels_SSH = np.arange(-0.3, 0.32, 0.02)
                ticks_label_SSH = np.arange(-0.3,0.31,0.1)
                #
                cmapTaux = 'plasma'
                levels_Taux = np.arange(0.1, 0.31, 0.01)
                ticks_label_Taux = np.arange(0.1, 0.35, 0.05)
                # 
                cmapTauy = 'plasma'
                levels_Tauy = np.arange(0.4, 0.61, 0.01)
                ticks_label_Tauy = np.arange(0.4, 0.62, 0.02)

            elif model_source.source == 'Croco':
                glon,glon_u,glon_v = ds[nameLon_rho],ds[nameLon_u],ds[nameLon_v]
                glat,glat_u,glat_v = ds[nameLat_rho],ds[nameLat_u],ds[nameLat_v]
                #
                levels_SSH = np.arange(-0.5, 0.55, 0.05)
                ticks_label_SSH = np.arange(-0.5, 0.52, 0.2)
                cmapSSH = 'seismic'
                #
                levels_Taux = np.arange(-0.1, 0.11, 0.01)
                ticks_label_Taux = np.arange(-0.1, 0.14, 0.04)
                cmapTaux = 'seismic'
                #
                levels_Tauy = levels_Taux # np.arange(0.5, 0.565, 0.005)
                ticks_label_Tauy = ticks_label_Taux #np.arange(0.5, 0.56, 0.01)
                cmapTauy = cmapTaux
            
            U, V = ds[nameU],ds[nameV]   
            TauX, TauY = ds[nameOceTaux],ds[nameOceTauy]
            #U10m, V10m = ds['geo5_u10m'],ds['geo5_v10m']
            SSH = ds[nameSSH]
            
            proj = ccrs.PlateCarree() # ccrs.Mercator()
            clb_or = 'horizontal'
            lon_formatter = LongitudeFormatter()
            lat_formatter = LatitudeFormatter()
            Nlevel_cb = 3
            
            tick_locator = mticker.MultipleLocator
            
            
            fig, ax = plt.subplots(1, 3, figsize = (12,4), subplot_kw={'projection':proj}, constrained_layout=True,dpi=dpi)
            s = ax[0].contourf(glon.values,glat.values,SSH[0].values, levels=levels_SSH, cmap=cmapSSH, extend='both')
            cb = plt.colorbar(s,ax=ax[0], aspect=50, pad=0.01, label='SSH (m)', orientation=clb_or,
                            ticks=ticks_label_SSH)
            s = ax[1].contourf(glon_u.values,glat_u.values,TauX[0].values, levels=levels_Taux, cmap=cmapTaux, extend='both')
            cb = plt.colorbar(s,ax=ax[1], aspect=50, pad=0.01, label=r'$\tau_x$ (m2/s2)', orientation=clb_or,
                            ticks=ticks_label_Taux )        
            s = ax[2].contourf(glon_v.values,glat_v.values,TauY[0].values, levels=levels_Tauy, cmap=cmapTauy, extend='both')
            cb = plt.colorbar(s,ax=ax[2], aspect=50, pad=0.01, label=r'$\tau_y$ (m2/s2)', orientation=clb_or,
                            ticks=ticks_label_Tauy)
            
            
            for axe in ax.flatten():
                axe.set_xticks(np.arange(-180,180,0.1), crs=ccrs.PlateCarree())
                axe.set_yticks(np.arange(-90,90,0.1), crs=ccrs.PlateCarree())
                axe.xaxis.set_major_locator(mticker.MultipleLocator(step_major_xtick))
                axe.xaxis.set_minor_locator(mticker.MultipleLocator(step_minor_xtick))
                axe.yaxis.set_major_locator(mticker.MultipleLocator(step_major_ytick))
                axe.yaxis.set_minor_locator(mticker.MultipleLocator(step_minor_ytick))
                axe.xaxis.set_major_formatter('{x}E')
                axe.yaxis.set_major_formatter('{x}N')
                
                gl = axe.gridlines(draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
                gl.xlocator = tick_locator(grd_tick)
                gl.ylocator = tick_locator(grd_tick)
                gl.xformatter = LongitudeFormatter(grd_tick)
                gl.yformatter = LatitudeFormatter(grd_tick)
                axe.set_extent(box, crs=proj)
                axe.tick_params(which='both',bottom=True, top=True, left=True, right=True)
            fig.savefig(path_save_png1D+nameJax+'Local_map_'+AREA+'.png')
        
    # minimization procedure of the cost function (n layers)
    if MINIMIZE:
        Nl = len(vector_k)//2
        print('* Minimization with '+str(Nl)+' layers')
        
        # intialisation of the model
        if JAXIFY:
            model = jUnstek1D(Nl, forcing, observations)
        else:
            model = Unstek1D(Nl, forcing, observations)
        var = Variational(model, observations)
        
        # _, Ca = model.do_forward(vector_k)
        # Ua, Va = jnp.real(Ca), jnp.imag(Ca)
        
        t1 = clock.time()
        J = var.cost(vector_k)
        print('J',J)
        t2 = clock.time()
        dJ = var.grad_cost(vector_k)
        print('dJ',dJ)
        print('-> time for J = ',np.round(t2 - t1,4) )
        print('-> time for dJ = ',np.round(clock.time() - t2,4) )

        t1 = clock.time()
        J = var.cost(vector_k)
        print('J',J)
        t2 = clock.time()
        dJ = var.grad_cost(vector_k)
        print('dJ',dJ)
        print('-> time for J = ',np.round(t2 - t1,4) )
        print('-> time for dJ = ',np.round(clock.time() - t2,4) )
        
        if PARALLEL_MINIMIZED:
            if JAXIFY:
                raise Exception('STOP: // minimize with JAX is not yet available.')
            # this does not work with Unstek1D class
            # res = minimize_parallel(fun=var.cost, x0=vector_k, #  args=(time, fc, TAx, TAy, Uo, Vo, Ri)
            #                   jac=var.grad_cost,
            #                   parallel={'loginfo': True, 'max_workers':N_CPU,'verbose':PRINT_INFO,'time':True},
            #                   options={'maxiter': maxiter})
        else:
            res = opt.minimize(var.cost, vector_k, args=(save_iter), # , args=(Uo, Vo, Ri)
                        method='L-BFGS-B',
                        jac=var.grad_cost,
                        options={'disp': True, 'maxiter': maxiter})
            
        if np.isnan(var.cost(res['x'])): # , Uo, Vo, Ri
            print('The model has crashed.')
        else:
            print(' vector K solution ('+str(res.nit)+' iterations)',res['x'])
            print(' cost function value with K solution:',var.cost(res['x'])) # , Uo, Vo, Ri
        vector_k = res['x']
        # vector K solution: [-3.63133021 -9.46349552]
        # cost function value with K solution: 0.36493136309782287
        
        
        # using the value of K from minimization, we get currents
        _, Ca = model.do_forward(vector_k)
        Ua, Va = np.real(Ca)[0],np.imag(Ca)[0]
        RMSE = score_RMSE(Ua, U)      
        
        title = ''
        for k in range(len(res['x'])):
            title += 'k'+str(k)+','   
        title = title[:-1]+' = '+str(res['x']) + ' ('+str(np.round(RMSE,3))+')'
        
            
        # PLOT trajectory
        plt.figure(figsize=(10,3),dpi=dpi)
        plt.plot(forcing.time/86400,U, c='k', lw=2, label='LLC ref')
        plt.plot(forcing.time/86400,Ua, c='g', label='Unstek')
        plt.scatter(observations.time_obs/86400,Uo, c='r', label='obs')
        plt.title(title)
        plt.xlabel('Time (days)')
        plt.ylabel('Ageo zonal current (m/s)')
        plt.legend(loc=1)
        plt.tight_layout()
        plt.savefig(path_save_png1D+nameJax+'series_reconstructed_long_'+str(Nl)+'layers.png')
        
        # Plot with MLD
        fig, ax = plt.subplots(2,1,figsize = (10,6),constrained_layout=True,dpi=dpi)
        ax[0].plot(forcing.time/86400,U, c='k', lw=2, label='LLC ref')
        ax[0].plot(forcing.time/86400,Ua, c='g', label='Unstek')
        ax[0].scatter(observations.time_obs/86400,Uo, c='r', label='obs')
        ax[0].set_title(title)
        ax[0].set_ylabel('Ageo zonal current (m/s)')
        ax[0].legend(loc=1)
        ax[1].plot(forcing.time/86400, - forcing.MLD, c='k')
        ax[1].set_xlabel('Time (days)')
        ax[1].set_ylabel('Ekman depth (m)')
        fig.savefig(path_save_png1D+nameJax+'series_reconstructed_long_'+str(Nl)+'layers_withMLD.png')
      
    # Plotting trajectories of given vector_k
    if PLOT_TRAJECTORY:
        print('* Plotting trajectories/step response for different vector K')
        
        if TRUE_WIND_STRESS:
            list_k = [[-9.39864959, -9.23882757], # mini with 2 layers
                      [ -9.14721791, -8.79469884, -11.20512638, -12.5794675], # mini with 2 layers
                      [ -9.14721791, -8.79469884, -14, -12.5794675], # hand chosen
                      ] 
        else:
            list_k = [[-3.63133021,  -9.46349552], # 1 layer, minimization from [-3,-12]
                    [-5,-12], # 1 layer, hand chosen
                    [-2,-7], # 1 layer, hand chosen
                    [ -3.47586165, -9.06189063, -11.22302904, -12.43724667], # 2 layer, minimization from [-3,-8,-10,-12]
                    ]
        PLOT_MLD = True
                
        A_wind = 10 # m/s
        indicator = np.zeros(len(forcing.time))
        indicator[10:] = 1
        step_stressY = 8e-6*np.sign(A_wind)*(indicator*A_wind)**2
        step_stressX = np.zeros(len(forcing.time))
        
        forcing_step = Forcing1D(dt, path_file, TRUE_WIND_STRESS=False)   
        forcing_step.TAx = step_stressX
        forcing_step.TAy = step_stressY
        
        
        for vector_k in list_k:
            print('     ',vector_k)
            Nl = len(vector_k)//2
            
            # regular model
            model = Unstek1D(Nl, forcing, observations)
            _, Ca = model.do_forward(vector_k)
            Ua, Va = np.real(Ca), np.imag(Ca)
            RMSE_U = score_RMSE(Ua, U)
            
            # step response
            model_step = Unstek1D(Nl, forcing_step, observations)
            _, Cstep = model_step.do_forward(vector_k)
            U_step,V_step = np.real(Cstep), np.imag(Cstep)
            
            # freq,PSD_score,f_score,smoothed_PSD = score_PSDerr(timeO/3600, Ua[::3600//dt], Va[::3600//dt], U[::3600//dt], V[::3600//dt],
            #                                                    show_score=True,smooth_PSD=True)
            #print(smoothed_PSD[-1])
            
            txt_add = ''
            title = ''
            for k in range(len(vector_k)):
                txt_add += '_k'+str(k)+str(vector_k[k])
                title += 'k'+str(k)+','    
            title = title[:-1]+' \n '+str(vector_k) + ' \nRMSE = '+str(np.round(RMSE_U,3))
            #title = title + ', PSDscore = '+str(np.round(1/f_score,3)) + ' hours'
            
            # trajectory
            plt.figure(figsize=(10,3),dpi=dpi)
            plt.plot(forcing.time/86400,U, c='k', lw=2, label='LLC ref ageostrophic zonal')
            plt.plot(forcing.time/86400,Ua, c='g', label='Unsteady-Ekman zonal reconstructed from wind')
            plt.scatter(observations.time_obs/86400,Uo, c='r', label='obs')
            plt.xlabel('Time (days)')
            plt.ylabel('Zonal current (m/s)')
            plt.title(title)
            plt.legend(loc=1)
            plt.tight_layout()
            plt.savefig(path_save_png1D+nameJax+'series_reconstructed_'+str(len(vector_k)//2)+'layers'+txt_add+'.png')

            # Plot PSD score
            # TO DO: i need to plot rotary spectra
            # fig, ax = plt.subplots(1,1,figsize = (7,5),constrained_layout=True,dpi=dpi)
            # ax.plot(1/freq[1:],PSD_score[1:],c='k')
            # if True:
            #     ax.plot(1/freq[1:], smoothed_PSD[1:],c='r')
            # ax.set_title(title)
            # ax.set_xlabel('hours')
            # ax.set_ylabel('PSD score')
            # ax.hlines(0.5,0.03,720,colors='grey',linestyles='--')
            # #ax.set_ylim([0,1])
            # ax.set_xlim([1/freq[1],1/freq[-1]])
            # ax.set_xscale('log')
            # fig.savefig(path_save_png1D+'PSDscore_'+str(Nlayers)+'layers'+txt_add+'.png')# PSD err
            
            

            # step response, hodograph
            if False:
                fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
                ax.plot(U_step,V_step,c='k')
                #ax.scatter(U_step,V_step,marker='x',c='k')
                ax.set_xlabel(r'U$_{E}$ (m/s)')
                ax.set_ylabel(r'V$_{E}$ (m/s)')
                ax.set_xlim([0,0.15])
                ax.set_ylim([0,0.15])
                fig.savefig(path_save_png1D+nameJax+'Hodograph_from_stepTau_'+str(len(vector_k)//2)+'layers'+txt_add+'.png')    
      
    # looking at the cost function
    #   for a slab model (1 layer)
    if ONE_LAYER_COST_MAP:
        print('* Looking at cost function for k0,k1 in 1 layer model')
    
        PLOT_ITERATIONS = False
        kmin = -20
        kmax = 0
        step = 0.25
        maxiter = 10
        vector_k = np.array([-12,-12]) # initial vector k
        Nl = len(vector_k)//2
        Jmap_cmap = 'terrain'
        
        tested_values = np.arange(kmin,kmax,step)  # -15,0,0.25

        if JAXIFY:
            model = jUnstek1D(Nl, forcing, observations)
            vector_k = jnp.asarray(vector_k)
        else:
            model = Unstek1D(Nl, forcing, observations)
        var = Variational(model, observations)
        
        
        if JAXIFY:
            # Vectorized
            jtested_values = jnp.asarray(tested_values)
            array_pk = jnp.asarray( [ [k0,k1] for k0 in jtested_values 
                                                for k1 in jtested_values] )
            indexes = jnp.asarray( [ [i,j] for i in jnp.arange(len(jtested_values)) 
                                             for j in jnp.arange(len(jtested_values))] )
            J = var.jax_cost_vect(array_pk, indexes)
            
        else:
            if N_CPU>1:
                # joblib //
                J = Parallel(n_jobs=N_CPU)(delayed(
                    var.cost)(np.array([k0,k1]))
                        for k0 in tested_values for k1 in tested_values)
                J = np.transpose(np.reshape(np.array(J),(len(tested_values),len(tested_values)) ))
            else:
                # serial
                J = np.zeros((len(tested_values),len(tested_values)))*np.nan
                for i,k0 in enumerate(tested_values):
                    print(i)
                    for j,k1 in enumerate(tested_values):
                        print('     ',j)
                        vector_k0k1 = np.array([k0,k1])
                        J[j,i] = var.cost(vector_k0k1)
        
        # plotting iterations of minimization
        if PLOT_ITERATIONS:
            cost_iter = np.zeros(maxiter)
            vector_k_iter = np.zeros((len(vector_k),maxiter))
            vector_k_iter[:,0] = vector_k
            txt_add = 'k0'+str(vector_k[0])+'_k1'+str(vector_k[1])
            for k in range(1,maxiter):
                res = opt.minimize(var.cost, vector_k,
                            method='L-BFGS-B',
                            jac=var.grad_cost,
                            options={'disp': True, 'maxiter': k})
                vector_k_iter[:,k] = res['x']
                cost_iter[k] = var.cost(vector_k_iter[:,k])
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
        ax.set_xlabel('log(k0)')
        ax.set_ylabel('log(k1)')
        plt.savefig(path_save_png1D+nameJax+'k0k1_J_1layer'+txt_add+'.png')
        
    # plot de J(k1,k2), à k0 et k3 fixés à leur valeur convergée
    if TWO_LAYER_COST_MAP_K1K2:
        print('* Looking at cost function for k1,k2 in 2 layer model')
        # k0 a un effet linéaire (facteur de tau)
        # k3 est juste en fonction du fond
        # les 2 autres font intervenir les autres couches
        
        # LAYER DEFINITION
        # -> number of values = number of layers
        # -> values = turbulent diffusion coefficient
        step = 1 # 0.25
        kmin,kmax = -28, -4.5
        Jmap_cmap = 'terrain'
        Nl = 2
        
        if TRUE_WIND_STRESS:
            pk_sol = dico_pk_solution[SOURCE][str(point_loc)][str(Nl)]['TRUE_TAU']
        else:
            pk_sol = dico_pk_solution[SOURCE][str(point_loc)][str(Nl)]['Cd.UU']
        
        k0 = pk_sol[0]
        k1_mini = pk_sol[1]
        k2_mini = pk_sol[2]
        k3 = pk_sol[3]
        
        #     k0 = -9.14721791
        #     k3 = -12.5794675
        #     k1_mini = -8.79469884
        #     k2_mini = -11.20512638
        #     kmin,kmax = -20, -1
        # else:
        #     k0 = -3.47586165
        #     k3 = -12.43724667
        #     k1_mini = -9.06189063
        #     k2_mini = -11.22302904
        #     kmin,kmax = -20, -4
        
        tested_values = np.arange(kmin,kmax,step)
        
        if JAXIFY: model = jUnstek1D(Nl, forcing, observations)
        else: model = Unstek1D(Nl, forcing, observations)
        var = Variational(model, observations)
    
    
        if JAXIFY:
            # Vectorized
            jtested_values = jnp.asarray(tested_values)
            array_pk = jnp.asarray( [ [k0,k1,k2,k3] for k1 in jtested_values 
                                                for k2 in jtested_values] )
            indexes = jnp.asarray( [ [i,j] for i in jnp.arange(len(jtested_values)) 
                                             for j in jnp.arange(len(jtested_values))] )
            J = var.jax_cost_vect(array_pk, indexes)
            
        else:
            if N_CPU>1:
                # joblib //
                # not working if some jax is in the file ...
                print(model.isJax)
                print(var.inJax)
                # J = Parallel(n_jobs=N_CPU)(delayed(
                #     var.cost)(np.array([k0,k1,k2,k3]))
                #         for k1 in tested_values for k2 in tested_values)
                # J = np.transpose(np.reshape(np.array(J),(len(tested_values),len(tested_values)) ))
            else:
                # serial
                J = np.zeros((len(tested_values),len(tested_values)))*np.nan
                for i,k1 in enumerate(tested_values):
                    print(i)
                    for j,k2 in enumerate(tested_values):
                        print('     ',j)
                        vector_k0k1 = np.array([k0,k1,k2,k3])
                        J[j,i] = var.cost(vector_k0k1)
            
        # plotting
        cmap = mpl.colormaps.get_cmap(Jmap_cmap)
        cmap.set_bad(color='indianred')
        fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
        s = ax.pcolormesh(tested_values,tested_values,J,cmap=cmap,norm=mpl.colors.LogNorm(0.1,10000)) #,vmin=0.1,vmax=100
        plt.colorbar(s,ax=ax)
        ax.scatter(k1_mini, k2_mini,marker='x',c='g')
        ax.set_xlabel('log(k1)')
        ax.set_ylabel('log(k2)')
        plt.savefig(path_save_png1D+nameJax+'k1k2_J_2layer.png')

    # vector_k components and physic
    if LINK_K_AND_PHYSIC:
        """
        BC:
            z = 0: Tau = TAx
            z = MLD: Uageo = 0

        There is a transormation between vector K and actual coefficient.
        K = exp(vector_k)
        K = [A0, ..., An]
        
        vertical stress is parametrized as Kz.dU/dz
        
        Hi is the thickness of layer i
        
        Discretisation (finite differences) is:
            Mid layers when N >= 3 : Central 
            Boundaries (and so also when N<=2): upward of backward
                if needed, ghost point
        
        1 layer: 
            A0 = 1 / (rho*H)
            A1 = A0*Kz1 / H
            
            H = 1/ (rho*A0)
            Kz0 = A1*h/A0
            
        2 layers:
            A0 = 1/(H1*rho)
            A1 = Kz1 / (rho*H1) * 2/(H1+H2) = A0*2*Kz1/(H1+H2)
            A2 = Kz2 / (rho*H2**2)
            A3 = Kz2 / (rho*H2) * 2/(H1+H2) = H2*A2.2/(H1+H2)        
            
            H1 = 1/(A[0]*rho)
            H2 = H1*1/(A[2]/A[3]-1)
            Kz1 = A[1]/(2*A[0])*(H1+H2)
            Kz2 = A[2]*rho*H2**2
            
        TO DO: 
            Save a text file with all the informations
        """
        print('* Link vector_k and physic')
        if TRUE_WIND_STRESS:
            txt_dico = 'TRUE_TAU'
        else:
            txt_dico = 'Cd.UU'  
        rhow = rho
        
        # 1 layer
        Nl = '1'
        pk = dico_pk_solution[SOURCE][str(point_loc)][Nl][txt_dico]
        K = np.exp(pk)
        H = 1/ (rhow*K[0])
        Kz0 = K[1]*H**2
        print(' -> 1 layer')
        print('     pk=',pk)
        print('     Ekman depth=',H,'m')
        print('     K=',Kz0,'m2/s')     
        
        # 2 layers
        Nl = '2'
        pk = dico_pk_solution[SOURCE][str(point_loc)][Nl][txt_dico]
        K = np.exp(pk)
        H1 = 1/(K[0]*rhow)
        H2 = H1*1/(K[2]/K[3]-1)
        Kz1 = K[1]/(2*K[0])*(H1+H2)
        Kz2 = K[2]*rhow*H2**2
        print(' -> 2 layers')
        print('     pk=',pk)
        print('     H1',H1,'m')
        print('     H2',H2,'m')
        print('     Ekman depth=',H1+H2,'m')
        print('     K1=',Kz1,'m2/s')
        print('     K2=',Kz2,'m2/s')
    
    # check of minimum, starting at corner of an hypercube
    #   for the two layer model
    if CHECK_MINI_HYPERCUBE:
        """
        Model is : 2 layers Junstek
        We get the extreme values of vector_k by looking at the physic.
        
        Recall the link between vector_k and physic:
        
                K0 = 1/(H1*rho)
                K1 = Kz1 / (rho*H1) * 2/(H1+H2) = K0*2*Kz1/(H1+H2)
                K2 = Kz2 / (rho*H2**2)
                K3 = Kz2 / (rho*H2) * 2/(H1+H2) = H2*K2.2/(H1+H2)
        
        
        Ekman depth Ed=H1+H2: from 1m to 100m
        Kz1, Kz2: from 10e-5 to 10e-2 (m2/s2)
        lets assume that 0.01H1<=  H2 <= H1
        
        pk = ln(K)
        
        this gives boundaries for each Ki:
            K0 : max H1 = 1m 
                 min H1 = 100m
                 -> [0.00001, 0.01] -> pk0 = [-11.5, -4.5]
            K1 : max K0max, Kz1 = 10e-2, Ed = 1m
                 min K0min, Kz1 = 10e-5, Ed = 100m
                 -> [2e-12, 2e-4] -> pk1 = [-27, -8.5]
            K2 : max Kz2max=1e-2, min H2=1m
                 min Kz2min = 1e-5, max H2 =100m
                 -> [1e-12, 1e-8] -> pk2 = [-27.5, -18.5]
            k3 : max H2max=100m,K2max=1e-8,Ed=100
                 min H2min=1,K2min=1e-12,Ed=1
                 -> [2e-8, 2e-12] -> pk3 = [-27, -17.5]   
                 
        Results:
            Croco
                LOCATION [W,E]= [-50.0, 35.0]
                    pk = []
                    cost = 
                    -> 
            MITgcm
                LOCATION [W,E]= [-24.8, 45.2]
                    pk = []
                    cost = 
                    ->
                 
        """ 
        print('* Testing if minimum is a local or global minimum')
        maxiter = 50
        dico_bounds = {'pk0':[-11.5, -4.5],
                       'pk1':[-27, -8.5],
                       'pk2':[-27.5, -18.5],
                       'pk3':[-27, -17.5]  }


        name_hypercube = 'results_hypercube_'+SOURCE+'_LON'+str(point_loc[0])+'_LAT'+str(point_loc[1])
        if pathlib.Path(name_hypercube).is_file():
            print(' file with hypercube results is here')
        else:
            model = jUnstek1D(2, forcing, observations)
            var = Variational(model, observations)
            
            solution = np.zeros((2**4,2,4)) # n° corner, (ini,final), vector_k
            nIter = np.zeros((2**4))     
            nCost = np.zeros((2**4))         
            npk = 0
            # looping over all corners of the hypercube
            for pk0 in dico_bounds['pk0']:
                for pk1 in dico_bounds['pk1']:
                    for pk2 in dico_bounds['pk2']:
                        for pk3 in dico_bounds['pk3']:
                            pk = jnp.asarray([pk0,pk1,pk2,pk3])
                            res = opt.minimize(var.cost, pk, args=(save_iter), # , args=(Uo, Vo, Ri)
                                method='L-BFGS-B',
                                jac=var.grad_cost,
                                options={'disp': True, 'maxiter': maxiter})

                            solution[npk,0,:] = np.asarray([pk0,pk1,pk2,pk3])
                            solution[npk,1,:] = res['x']
                            nIter[npk] = res.nit
                            nCost[npk] = var.cost(jnp.asarray(res['x']))
    
                            print('corner n°'+str(npk))
                            print('     pk(ini) =',solution[npk,0,:])
                            print('     pk(end) =',solution[npk,1,:])
                            print('       niter =',nIter[npk])
                            print('        cost =',nCost[npk])
                            npk+=1
                            
            # writing in the file
            with open(name_hypercube+".txt", "w") as f:     
                f.write("* HEADER ========================================\n")
                f.write('* MODEL = '+str(SOURCE)+'\n')
                f.write('* LOCATION = LON'+str(point_loc[0])+'_LAT'+str(point_loc[1])+'\n')
                f.write('*\n')
                f.write('* num of corner'+'\n')
                f.write('* pk(ini)'+'\n')
                f.write('* pk(end)'+'\n')
                f.write('* niter'+'\n')
                f.write('* cost'+'\n')
                f.write('* ===============================================\n')
                for npk in range(solution.shape[0]):
                    f.write(str(npk)+'\n')
                    f.write(str(solution[npk,0,:])+'\n')
                    f.write(str(solution[npk,1,:])+'\n')
                    f.write(str(nIter[npk])+'\n')
                    f.write(str(nCost[npk])+'\n')
            
        # Getting the values from file        
           
    # Looking at profiles in Croco          
    if PLOT_CROCO_PROFILES:
        """
        """              
        print('* Plotting Croco 3D profile')
        raise Exception('This is WIP, opening file is long i need to fix that')
        
        # open model
        model_croco = Model_source_3D('Croco',files_dict['Croco']['3D'])
        ds = model_croco.dataset
        
        # reduce to 'point_loc'
        indx, indy = find_indices(point_loc, ds.lon_rho, ds.lat_rho)[0]
        #ds = 
        
        print(ds)        
                               
    # TESTS
    if TEST_ROTARY_SPECTRA:
        
        # pour l'instant je le fait en 1D à 1 point lat,lon donc le spectre est bruité.
        print(' * Plotting rotary spectra')
        vector_k = np.asarray([-9.11293915, -8.73891447, -11.15877101, -12.56505214])
        Nl = len(vector_k)//2
        
        if JAXIFY:
            vector_k = jnp.asarray(vector_k)
            model = jUnstek1D(Nl, forcing, observations)
        else:
            model = Unstek1D(Nl, forcing, observations)
        var = Variational(model, observations)
        
        # real data
        _, Ca = model.do_forward(vector_k)
        Ua, Va = jnp.real(Ca)[0], jnp.imag(Ca)[0]
        Ur, Vr = U[::dt],V[::dt]
        Ua, Va = Ua[::dt],Va[::dt]
        # synthetic data
        Ur = U[2*dt::dt].copy()
        Vr = V[2*dt::dt].copy()
        Ua = 1*U[:-2*dt:dt]
        Va = 1*V[:-2*dt:dt]
        
        fig, axs = plt.subplots(1,1,figsize=(7,6))
        axs.plot(Ur)
        axs.plot(Ua)
        plt.show()
        
        dtH = 1. # 1 hour
        ff, MPr2, MPr1, MPe2, MPe1 = rotary_spectra(dtH,Ua,Va,Ur,Vr)
        
        fig, axs = plt.subplots(2,1,figsize=(7,6), gridspec_kw={'height_ratios': [4, 1.5]})
        axs[0].loglog(ff,MPr2, c='k', label='reference')
        axs[0].loglog(ff,MPe2, c='b', label='error')
        axs[0].axis([2e-3,2e-1, 1e-4,10])
        axs[0].grid('on', which='both')
        plt.xlabel('hours-1')
        axs[0].legend()
        axs[0].title.set_text('Clockwise PSD first layer', )
        axs[1].semilogx(ff,(1-MPe2/MPr2)*100, c='b', label='Reconstruction Score')
        axs[1].axis([2e-3,2e-1,0,100])
        axs[1].grid('on', which='both')
        axs[1].title.set_text('Scores (%)')
        #plt.savefig('diag.png')  
        
        fig, axs = plt.subplots(2,1,figsize=(7,6), gridspec_kw={'height_ratios': [4, 1.5]})
        axs[0].loglog(ff,MPr1, c='k', label='reference')
        axs[0].loglog(ff,MPe1, c='b', label='error')
        axs[0].axis([2e-3,2e-1, 1e-4,2e0])
        axs[0].grid('on', which='both')
        plt.xlabel('hours-1')
        axs[0].legend()
        axs[0].title.set_text('Counter-Clockwise PSD 15m', )
        axs[1].semilogx(ff,(1-MPe1/MPr1)*100, c='b', label='Reconstruction Score')
        axs[1].axis([2e-3,2e-1,0,100])
        axs[1].grid('on', which='both')
        axs[1].title.set_text('Scores (%)')
        #plt.savefig('diag.png')
       
    if TEST_JUNSTEK1D_KT:
        """
        Tests of class jUnstek1D_Kt
        """  
        
        dT = 28*86400 # s
        #vector_k = jnp.asarray([-11.31980127, -10.28525189])
        vector_k = jnp.asarray([-10.76035344, -9.3901326, -10.61707124, -12.66052074])
        
        Nl = len(vector_k)//2
        forcing1D = Forcing1D(dt, path_file, TRUE_WIND_STRESS)
        model = jUnstek1D_Kt(Nl, forcing=forcing1D, dT=dT)
        var = Variational(model, observations)
        
        print('* test junstek1D_kt '+str(Nl)+' layers')
        # time varying vector_k
        vector_kt = model.kt_ini(vector_k)
        vector_kt_1D = model.kt_2D_to_1D(vector_kt) # scipy.minimize only accept 1D array

        
        t1 = clock.time()
        _, Ca = model.do_forward_jit(vector_kt_1D)
        Ua, Va = np.real(Ca)[0], np.imag(Ca)[0]
        t2 = clock.time()
        print('time, forward model (with compile)',t2-t1)
        
        _, Ca = model.do_forward_jit(vector_kt_1D)
        Ua, Va = np.real(Ca)[0], np.imag(Ca)[0]
        print('time, forward model (no compile)',clock.time()-t2)
        
        res = opt.minimize(var.cost, vector_kt_1D, args=(save_iter),
                        method='L-BFGS-B',
                        jac=var.grad_cost,
                        options={'disp': True, 'maxiter': maxiter})
            
        print_info(var.cost,res)
        vector_kt_1D = res['x']
        _, Ca = model.do_forward_jit(vector_kt_1D)
        Ua, Va = np.real(Ca)[0], np.imag(Ca)[0]
        
        RMSE = score_RMSE(Ua, U) 
        print('RMSE is',RMSE)
        # PLOT trajectory
        plt.figure(figsize=(10,3),dpi=dpi)
        plt.plot(forcing.time/86400,U, c='k', lw=2, label='LLC ref')
        plt.plot(forcing.time/86400,Ua, c='g', label='Unstek')
        plt.scatter(observations.time_obs/86400,Uo, c='r', label='obs')
        plt.title('RMSE='+str(np.round(RMSE,4))+' cost='+str(np.round(var.cost(res['x']),4)))
        plt.xlabel('Time (days)')
        plt.ylabel('Ageo zonal current (m/s)')
        plt.legend(loc=1)
        plt.tight_layout()
        plt.savefig('JAX_test_junstek1D_kt_'+str(Nl)+'layers.png')
        
    if TEST_JUNSTEK1D_KT_SPATIAL:
        """
        Note:
        Hugo 13/02/2025: Forward model is ok, if dT = 28 days, it gives the same results as JUNSTEK1D_KT.
                        But, performance when computing the gradient needs to be tweaked.
                        Memory usage is too high, do I need to code better in JAX ? 
                        I need to look more into JAX checkpoint.
                        Maybe ask Florian how he does it with MSSH ?
                        
        Hugo 18/02/2025 : I regridded the forcing dataset, now at 0.1deg so its super fast.
                        But my minimization doesnt work anymore
                        -> fixed
                        
                        - now it works, model runs, minimisation converges.
        """
        N_CPU = 1
        Nl = 2
        dT = 28*86400 # s
        dt_forcing = 3600 # s
        dt = 60 # model timestep
        if Nl==1:
            vector_k = jnp.asarray([-11.31980127, -10.28525189])
        if Nl==2:
            vector_k = jnp.asarray([-10.76035344, -9.3901326, -10.61707124, -12.66052074])

        
        print('* test jUnstek1D_Kt_spatial '+str(Nl)+' layers')
        #file = 'Croco_Interp_2D_LON-55.0_-50.0_LAT30.0_35.0.nc'
        file = './data_regrid/croco_1h_inst_surf_2006-02-01-2006-02-28_0.1deg_conservative.nc'
        
        forcing2D = Forcing2D(dt_forcing, file, TRUE_WIND_STRESS, LON_bounds, LAT_bounds)
        observations2D = Observation2D(period_obs, dt_OSSE, file, LON_bounds, LAT_bounds)
        model = jUnstek1D_Kt_spatial(dt, Nl, forcing2D, dT)
        var = Variational(model, observations2D)

        # transform into 1D vector
        vector_kt = model.kt_ini(vector_k)
        print('vector_kt',vector_kt)

        t1 = clock.time()
        _, Ca = model.do_forward_jit(vector_kt)

        Ua, Va = np.real(Ca)[0], np.imag(Ca)[0]
        t2 = clock.time()
        print('time, forward model (with compile)',t2-t1)
        
        # _, Ca = model.do_forward_jit(vector_kt)
        # Ua, Va = np.real(Ca)[0], np.imag(Ca)[0]
        # print('time, forward model (no compile)',clock.time()-t2)
        
        # t3 = clock.time()
        # J = var.cost(vector_kt)
        # print('time, cost (with compile)',clock.time()-t3)

        # t4 = clock.time()
        # J = var.cost(vector_kt)
        # print('time, cost (no compile)',clock.time()-t4)

        # t5 = clock.time()
        # dJ = var.grad_cost(vector_kt)
        # print('time, gradcost (with compile)',clock.time()-t5)

        # t6 = clock.time()
        # dJ = var.grad_cost(vector_kt)
        # print('time, gradcost (no compile)',clock.time()-t6)
        
        # Compare vs 1D version
        if False:
            # before looking at minimizing, I need to check if the model runs ok.
            # for this I will compare this model to the 1D version, at the same lon/lat location.
            # for the same vector_k, results should be the same.
            # results: its ok, not exactly the same but this could be bc
            #       - I use the regrid product, which interpolate and so introduces some error
            #       - I compare de 2D at a specific location with the 1D. The 2D model at 'point_loc'
            #               is only the nearest value (regrid grid is 0.1°)
            
            Nl = len(vector_k)//2
            forcing1D = Forcing1D(dt, path_file, TRUE_WIND_STRESS)
            model2 = jUnstek1D_Kt(Nl, forcing=forcing1D, dT=dT)
            var2 = Variational(model, observations)
            
            vector_kt = model.kt_ini(vector_k)
            vector_kt_1D = model.kt_2D_to_1D(vector_kt)
            _, Ca = model2.do_forward_jit(vector_kt_1D)
            Ua2 = np.real(Ca)[0]
            
            # for the 2D version, lets find indexes
            #indx,indy = find_indices(point_loc,forcing2D.data.lon.values,forcing2D.data.lat.values,tree=None)[0]
            indx = nearest(forcing2D.data.lon.values, point_loc[0])
            indy = nearest(forcing2D.data.lat.values, point_loc[1])
            
            fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
            ax.plot(model2.model_time, Ua2, c='b')
            ax.plot(model.forcing_time[1:], Ua[:-1,indy,indx], c='r',ls='--')
            ax.plot(forcing.time,U,c='k')
            
            print('len(model2.model_time), len(model.forcing_time), len(forcing.time)')
            print( len(model2.model_time), len(model.forcing_time), len(forcing.time))
            
            ax.set_xlabel('time')
            ax.set_ylabel('U ageo')
            plt.show()
        
        
         
        res = opt.minimize(var.cost, vector_kt, args=(save_iter),
                        method='L-BFGS-B',
                        jac=var.grad_cost,
                        options={'disp': True, 'maxiter': maxiter})
            
        print_info(var.cost,res)
        vector_kt = res['x']
        _, Ca = model.do_forward_jit(vector_kt)
        Ua, Va = np.real(Ca)[0], np.imag(Ca)[0]
        
        # select at location
        indx = nearest(forcing2D.data.lon.values,point_loc[0])
        indy = nearest(forcing2D.data.lat.values,point_loc[1])
        Ua = Ua[:,indy,indx]
        U = forcing2D.data.U[:,indy,indx].values
        Uo, Vo = observations2D.get_obs()
        Uo = Uo[:,indy,indx]
        
        RMSE = score_RMSE(Ua, U) 
        print('RMSE is',RMSE)
        # PLOT trajectory
        fig, ax = plt.subplots(1,1,figsize = (10,3),constrained_layout=True,dpi=dpi)
        ax.plot(forcing2D.time/86400, U, c='k', lw=2, label='Croco')
        ax.plot(forcing2D.time/86400, Ua, c='g', label='Unstek')
        ax.scatter(observations2D.time_obs/86400,Uo, c='r', label='obs')
        ax.set_ylim([-0.3,0.4])
        ax.set_title('RMSE='+str(np.round(RMSE,4))+' cost='+str(np.round(var.cost(vector_kt),4)))
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Ageo zonal current (m/s)')
        ax.legend(loc=1)
        fig.savefig('JAX_test_junstek1D_kt_spatial_'+str(Nl)+'layers.png')
    
    # execution benchmark   
    if BENCHMARK_ALL:
        print('* Benchmarking ...')       
        Nexec = 20
        dT = 1*86400        
        
        NB_LAYER = [1,2,3]
        
        # 1 layer
        if 1 in NB_LAYER:
            Nl = 1
            pk = dico_pk_solution[SOURCE][str(point_loc)][str(Nl)]['TRUE_TAU']
            Lmodel = [  Unstek1D(Nl, forcing=forcing, observations=observations),
                        jUnstek1D(Nl, forcing=forcing, observations=observations),
                        jUnstek1D_Kt(Nl, forcing=forcing, observations=observations, dT=dT)]
            benchmark_all(pk, Lmodel, observations, Nexec)
        
        # 2 layer
        if 2 in NB_LAYER:
            Nl = 2
            pk = dico_pk_solution[SOURCE][str(point_loc)][str(Nl)]['TRUE_TAU']
            Lmodel = [  Unstek1D(Nl, forcing=forcing, observations=observations),
                        jUnstek1D(Nl, forcing=forcing, observations=observations),
                        jUnstek1D_Kt(Nl, forcing=forcing, observations=observations, dT=dT)]
            benchmark_all(pk, Lmodel, observations, Nexec)
        
        # 3 layer
        if 3 in NB_LAYER:
            Nl = 3
            pk = dico_pk_solution[SOURCE][str(point_loc)][str(Nl)]['TRUE_TAU']
            Lmodel = [  Unstek1D(Nl, forcing=forcing, observations=observations),
                        jUnstek1D(Nl, forcing=forcing, observations=observations),
                        jUnstek1D_Kt(Nl, forcing=forcing, observations=observations, dT=dT)]
            benchmark_all(pk, Lmodel, observations, Nexec)
      
    if REGRID_vs_FINE:
        print('* Comparing regrid and original Croco')
        #dsfine, xgrid = open_croco_sfx_file(['/data2/nobackup/clement/Data/Lionel_coupled_run/croco_1h_inst_surf_2006-02-01-2006-02-28.nc'])
        dsfine, xgrid = open_croco_sfx_file(['/home/jacqhugo/Datlas_2025/DATA_Crocco/croco_1h_inst_surf_2006-02-01-2006-02-28.nc'])
        
        ds1D = xr.open_dataset(path_file)
        dsregridc = xr.open_dataset('./data_regrid/croco_1h_inst_surf_2006-02-01-2006-02-28_0.1deg_conservative.nc')
        dsregridb = xr.open_dataset('./data_regrid/croco_1h_inst_surf_2006-02-01-2006-02-28_0.1deg_bilinear.nc')
        
        indices = find_indices_gridded_latlon(dsfine.lon_rho.values, 
                                              [point_loc[0]-0.1,point_loc[0]+0.1], 
                                              dsfine.lat_rho.values, 
                                              [point_loc[1]-0.1,point_loc[1]+0.1])
        indxmin,indxmax,indymin,indymax = indices
        
        indx = nearest(dsregridc.lon.values,point_loc[0])
        indy = nearest(dsregridc.lat.values,point_loc[1])
        
        fig, ax = plt.subplots(3,1,figsize = (10,7),constrained_layout=True,dpi=dpi)
        ax[0].plot( dsfine.time, dsfine.U.isel(x_u=slice(indxmin,indxmax), 
                                               y_rho=slice(indymin,indymax)).mean(dim=['x_u','y_rho']), label='croco', c='k')
        ax[0].plot( ds1D.time, ds1D.U+ds1D.Ug,label=' 1D file', c='b')
        ax[0].plot( dsregridc.time, (dsregridc.U+dsregridc.Ug).sel(lon=point_loc[0],lat=point_loc[1], method='nearest'), label='regridc', c='g')
        ax[0].plot( dsregridb.time, dsregridb.U.sel(lon=point_loc[0],lat=point_loc[1], method='nearest'), label='regridb', c='orange')
        ax[0].legend()
        ax[0].set_ylabel('U')
        ax[0].set_ylim([-0.3,0.4])
        
        ax[1].plot( dsfine.time, dsfine.SSH.isel(x_rho=slice(indxmin,indxmax), 
                                               y_rho=slice(indymin,indymax)).mean(dim=['x_rho','y_rho']), label='croco', c='k')
        ax[1].plot( ds1D.time, ds1D.SSH,label=' 1D file', c='b')
        ax[1].plot( dsregridc.time, dsregridc.SSH.sel(lon=point_loc[0],lat=point_loc[1], method='nearest'), label='regridc', c='g')
        ax[1].plot( dsregridb.time, dsregridb.SSH.sel(lon=point_loc[0],lat=point_loc[1], method='nearest'), label='regridb', c='orange')
        ax[1].legend()
        ax[1].set_ylabel('SSH')
        
        ax[2].plot( dsfine.time, dsfine.oceTAUX.isel(x_u=slice(indxmin,indxmax), 
                                               y_rho=slice(indymin,indymax)).mean(dim=['x_u','y_rho']), label='croco', c='k')
        ax[2].plot( ds1D.time, ds1D.oceTAUX,label=' 1D file', c='b')
        ax[2].plot( dsregridc.time, dsregridc.oceTAUX.sel(lon=point_loc[0],lat=point_loc[1], method='nearest'), label='regridc', c='g')
        ax[2].plot( dsregridb.time, dsregridb.oceTAUX.sel(lon=point_loc[0],lat=point_loc[1], method='nearest'), label='regridb', c='orange')
        ax[2].legend()
        ax[2].set_ylabel('oceTAUX')
        fig.savefig('comparison_croco_1Dinterp_regridc_regridb.png')
        #ax[2].set_ylim([-0.3,0.4])
        
          
    # SCIENCE TO DOs:
    # - estimation de Uageo : dépendance à la période de filtrage pour estimer Ugeo
    # - Pour le 2 couches : test du point de départ (hypercube) pour trouver un potentiel second minimum
    #       -> le vecteur final est vraiment =/=, mais souvent une des composantes ne bouge pas -> un parametre inutile ?
    # - préparation modèle 2D: modele 1D appliqué à une grille 5°/5°
    #       WIP: 17/02/25 regrid Croco is ok
    # - rotary spectra sur un grand domaine spatial pour meilleur convergence
    # - télécharger fichier janvier Croco (et fichier été ?)
    #       Done
    # - ekman depth vs MLD: comparer (1D, f(time) )
    #       model simple 2 couches avec K obtenu par minimization
    #       model simple 100 couches avec K obtenu par Croco 3D (/3h, à interpoler sur grille fixe)
    #       model 3D: MLD basé sur gradient de densité
    
    # HIGH PRORITY: Compare the model at the PAPA station !!
    
    # TECHNICAL TO DOs
    # test obs : original vs regrid file at 'point_loc'
    #   -> done, its ok. 
    #      note: my 1D interpolation if off, i need to adapt old versions of unstek to use the new regrid file
    #           -> no its not off, i forgot geostrophy, now its ok !
    # merge version of junstek1D_kt and junstek1D_kt_spatial : diff in implementation
    # cleanup junstek1D_kt_spatial
    # use the regrid product even for 1D cases
    
    # Note pour Hugo:
    # A faire: clean up ce fichier, enlever les option pour MITgcm.
    
    end = clock.time()
    print('Total execution time = '+str(np.round(end-start,2))+' s')
    plt.show()