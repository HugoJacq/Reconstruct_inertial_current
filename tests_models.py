"""
This file gather tests for models

python tests_models.py

by: Hugo Jacquet (Datlas)
    hugo.jacquet@datlas.fr
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
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# jax.config.update('jax_platform_name', 'cpu')

# custom imports
# sys.path.append('./src/')
# from src.OSSE import *
# from src.observations import *
# from src.forcing import *
# from src.unstek import *
# from src.junstek import *
# from src.inv import *
# from src.tools import *
# from src.scores import *
# from src.benchmark import *
# from src.classic_NIO_models import *

#from src import OSSE
from src import observations
from src import forcing
#from src import unstek
from src import junstek
from src import inv
from src import tools
#from src import scores
#from src import benchmark
from src import classic_NIO_models

start = clock.time()

###############################################
# PARAMETERS                                  #
###############################################
# //
DASHBOARD   = False     # when using dask
N_CPU       = 8         # when using joblib, if >1 then use // code
JAXIFY      = True      # whether to use JAX or not
ON_HPC      = True      # on HPC

# -> area of interest
# 1D
point_loc = [-50.,35.]
# 2D
# LON_bounds = [-52.5,-47.5]
# LAT_bounds = [32.5,37.5]
# LON_bounds = [-50.2,-49.8]
# LAT_bounds = [34.8,35.2]
R = 0.2 # °, for 5°x5° i need 18Go of vram ...
LON_bounds = [point_loc[0]-R,point_loc[0]+R]
LAT_bounds = [point_loc[1]-R,point_loc[1]+R]

# -> Observations : OSSE
SOURCE              = 'Croco'   # MITgcm Croco
dt                  = 60        # timestep of the model (s) 
dt_OSSE             = 3600      # timestep of the OSSE (s)
period_obs          = 86400      # s, how many second between observations #86400    
Nl                  = 1         # number of layers
dT                  = 3*86400   # how much vectork K changes with time, basis change to exp
dt_forcing          = 3600      # forcing timestep
    
# MINIMIZATION
MINIMIZE            = False      # switch to do the minimisation process
maxiter             = 100       # max number of iteration

# tests
TEST_JUNSTEK1D              = False     # implementing junstek1D
TEST_JUNSTEK1D_KT           = False     # implementing junstek1D_kt
TEST_JUNSTEK1D_KT_SPATIAL   = False     # implementing jUnstek1D_spatial
TEST_CLASSIC_SLAB           = False     # implementing classic_slab1D
TEST_CLASSIC_SLAB_KT        = False     # implementing classic_slab1D_Kt
TEST_CLASSIC_SLAB_EQX       = True      # 
# regrid data
path_regrid = './data_regrid/'
name_regrid = 'croco_1h_inst_surf_2006-02-01-2006-02-28_0.1deg_conservative.nc'

# -> List of files (from Croco)
# Jackzilla
# if ON_HPC:
#     files_dict = {'Croco':{'surface':[#'/data2/nobackup/clement/Data/Lionel_coupled_run/croco_1h_inst_surf_2006-01-01-2006-01-31.nc',
#                                     '/data2/nobackup/clement/Data/Lionel_coupled_run/croco_1h_inst_surf_2006-02-01-2006-02-28.nc'],
#                             '3D':['/data2/nobackup/clement/Data/Lionel_coupled_run/croco_3h_U_aver_2006-02-01-2006-02-28.nc',
#                                 '/data2/nobackup/clement/Data/Lionel_coupled_run/croco_3h_V_aver_2006-02-01-2006-02-28.nc']},}
# # Local
# else:
#     files_dict = {'Croco':{'surface':'/home/jacqhugo/Datlas_2025/DATA_Crocco/croco_1h_inst_surf_2006-02-01-2006-02-28.nc',
#                             '3D':['/home/jacqhugo/Datlas_2025/DATA_Crocco/croco_3h_U_aver_2006-02-01-2006-02-28.nc',
#                                 '/home/jacqhugo/Datlas_2025/DATA_Crocco/croco_3h_V_aver_2006-02-01-2006-02-28.nc']},}    

# -> PLOT
dpi=200
path_save_png = './png_tests_models/'

# END PARAMETERS #################################
##################################################
namesave_loc = str(point_loc[0])+'E_'+str(point_loc[1])+'N'
namesave_loc_area = str(point_loc[0])+'E_'+str(point_loc[1])+'N_R'+str(R)

# MAIN LOOP
# This avoids infinite subprocess creation
if __name__ == "__main__":    


    global client
    client = None
    if DASHBOARD:
        # sometimes dask cluster can cause problems "memoryview is too large"
        # (writing a big netcdf file for eg, hbudget_file)
        cluster = LocalCluster(n_workers=8) # threads_per_worker=1,
        client = Client(cluster)
        print("Dashboard at :",client.dashboard_link)
    
    # regrided file
    # Regriding is done with "regrid_croco.py"
    file = path_regrid+name_regrid 
    forcing1D = forcing.Forcing1D(point_loc, dt_OSSE, file)
    observations1D = observations.Observation1D(point_loc, period_obs, dt_OSSE, file)
    forcing2D = forcing.Forcing2D(dt_forcing, file, LON_bounds, LAT_bounds)
    observations2D = observations.Observation2D(period_obs, dt_OSSE, file, LON_bounds, LAT_bounds)
    
    if Nl==1:
        vector_k = jnp.asarray([-11.31980127, -10.28525189])
    if Nl==2:
        vector_k = jnp.asarray([-10.76035344, -9.3901326, -10.61707124, -12.66052074])
    
    
    if TEST_JUNSTEK1D:
        print('* test jUnstek1D_v2, N='+str(Nl)+' layers')

        model = junstek.jUnstek1D_v2(dt, Nl, forcing1D)
        var = inv.Variational(model, observations1D)    
        
        t1 = clock.time()
        _, Ca = model.do_forward_jit(vector_k)

        Ua, Va = jnp.real(Ca)[0], jnp.imag(Ca)[0]
        t2 = clock.time()
        print('time, forward model (with compile)',t2-t1)
        
        _, Ca = model.do_forward_jit(vector_k)
        Ua, Va = jnp.real(Ca)[0], jnp.imag(Ca)[0]
        print('time, forward model (no compile)',clock.time()-t2)
        
        t3 = clock.time()
        J = var.cost(vector_k)
        print('time, cost (with compile)',clock.time()-t3)

        t4 = clock.time()
        J = var.cost(vector_k)
        print('time, cost (no compile)',clock.time()-t4)

        t5 = clock.time()
        dJ = var.grad_cost(vector_k)
        print('time, gradcost (with compile)',clock.time()-t5)

        t6 = clock.time()
        dJ = var.grad_cost(vector_k)
        print('time, gradcost (no compile)',clock.time()-t6)
    
        if MINIMIZE:
            print('-> minimizing ...')
            res = opt.minimize(var.cost, vector_k,
                            method='L-BFGS-B',
                            jac=var.grad_cost,
                            options={'disp': True, 'maxiter': maxiter})
                
            inv.print_info(var.cost,res)
            vector_k = res['x']
            _, Ca = model.do_forward_jit(vector_k)
            Ua, Va = jnp.real(Ca)[0], jnp.imag(Ca)[0]
    
        # PLOT
        U = forcing1D.data.U.values
        Uo, Vo = observations1D.get_obs()
        
        RMSE = tools.score_RMSE(Ua, U) 
        print('RMSE is',RMSE)
        # PLOT trajectory
        fig, ax = plt.subplots(1,1,figsize = (10,3),constrained_layout=True,dpi=dpi)
        ax.plot(forcing1D.time/86400, U, c='k', lw=2, label='Croco')
        ax.plot(forcing1D.time/86400, Ua, c='g', label='Unstek')
        ax.scatter(observations1D.time_obs/86400,Uo, c='r', label='obs')
        ax.set_ylim([-0.3,0.4])
        ax.set_title('RMSE='+str(jnp.round(RMSE,4))+' cost='+str(jnp.round(var.cost(vector_k),4)))
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Ageo zonal current (m/s)')
        ax.legend(loc=1)
        fig.savefig(path_save_png+'JAX_test_junstek1D_v2_'+str(Nl)+'layers'+namesave_loc+'.png')
        
    if TEST_JUNSTEK1D_KT:
        """
        """
        print('* test jUnstek1D_Kt_v2, N='+str(Nl)+' layers')
            
        model = junstek.jUnstek1D_Kt_v2(dt, Nl, forcing1D, dT)
        var = inv.Variational(model, observations1D)    
        
        # transform into 1D vector
        vector_kt = model.kt_ini(vector_k)
        print('vector_kt',vector_kt)
        
        t1 = clock.time()
        _, Ca = model.do_forward_jit(vector_kt)

        Ua, Va = jnp.real(Ca)[0], jnp.imag(Ca)[0]
        t2 = clock.time()
        print('time, forward model (with compile)',t2-t1)
        
        _, Ca = model.do_forward_jit(vector_kt)
        Ua, Va = jnp.real(Ca)[0], jnp.imag(Ca)[0]
        print('time, forward model (no compile)',clock.time()-t2)
        
        t3 = clock.time()
        J = var.cost(vector_kt)
        print('time, cost (with compile)',clock.time()-t3)

        t4 = clock.time()
        J = var.cost(vector_kt)
        print('time, cost (no compile)',clock.time()-t4)

        t5 = clock.time()
        dJ = var.grad_cost(vector_kt)
        print('time, gradcost (with compile)',clock.time()-t5)

        t6 = clock.time()
        dJ = var.grad_cost(vector_kt)
        print('time, gradcost (no compile)',clock.time()-t6)
    
        if MINIMIZE:
            print('-> minimizing ...')
            res = opt.minimize(var.cost, vector_kt,
                            method='L-BFGS-B',
                            jac=var.grad_cost,
                            options={'disp': True, 'maxiter': maxiter})
                
            inv.print_info(var.cost,res)
            vector_kt = res['x']
            _, Ca = model.do_forward_jit(vector_kt)
            Ua, Va = jnp.real(Ca)[0], jnp.imag(Ca)[0]
    
        # PLOT
        U = forcing1D.data.U.values
        Uo, Vo = observations1D.get_obs()
        
        RMSE = tools.score_RMSE(Ua, U) 
        print('RMSE is',RMSE)
        # PLOT trajectory
        fig, ax = plt.subplots(1,1,figsize = (10,3),constrained_layout=True,dpi=dpi)
        ax.plot(forcing1D.time/86400, U, c='k', lw=2, label='Croco')
        ax.plot(forcing1D.time/86400, Ua, c='g', label='Unstek')
        ax.scatter(observations1D.time_obs/86400,Uo, c='r', label='obs')
        ax.set_ylim([-0.3,0.4])
        ax.set_title('RMSE='+str(jnp.round(RMSE,4))+' cost='+str(jnp.round(var.cost(vector_kt),4)))
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Ageo zonal current (m/s)')
        ax.legend(loc=1)
        fig.savefig(path_save_png+'JAX_test_junstek1D_kt_v2_'+str(Nl)+'layers'+namesave_loc+'.png')
    
    if TEST_JUNSTEK1D_KT_SPATIAL:
        """    
        A note on the result: 
            it doesnt fit as well a single trajectory
            because we inverse the problem to find the best vector K for 
            ALL points defined inside LON_bounds,LAT_bounds. So for a specific
            location, we cant match the performance of a 1D model at the same 
            locatione.
        """        
        print('* test jUnstek1D_Kt_spatial, N='+str(Nl)+' layers')        
        
        model = junstek.jUnstek1D_Kt_spatial(dt, Nl, forcing2D, dT)
        var = inv.Variational(model, observations2D)

        # transform into 1D vector
        vector_kt = model.kt_ini(vector_k)
        print('vector_kt',vector_kt)

        t1 = clock.time()
        _, Ca = model.do_forward_jit(vector_kt)

        Ua, Va = jnp.real(Ca)[0], jnp.imag(Ca)[0]
        t2 = clock.time()
        print('time, forward model (with compile)',t2-t1)
        
        _, Ca = model.do_forward_jit(vector_kt)
        Ua, Va = jnp.real(Ca)[0], jnp.imag(Ca)[0]
        print('time, forward model (no compile)',clock.time()-t2)
        
        t3 = clock.time()
        J = var.cost(vector_kt)
        print('time, cost (with compile)',clock.time()-t3)

        t4 = clock.time()
        J = var.cost(vector_kt)
        print('time, cost (no compile)',clock.time()-t4)

        t5 = clock.time()
        dJ = var.grad_cost(vector_kt)
        print('time, gradcost (with compile)',clock.time()-t5)

        t6 = clock.time()
        dJ = var.grad_cost(vector_kt)
        print('time, gradcost (no compile)',clock.time()-t6)
        
        # Compare vs 1D version
        # see note
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
        
        if MINIMIZE:
            print('-> minimizing ...')
            res = opt.minimize(var.cost, vector_kt,
                            method='L-BFGS-B',
                            jac=var.grad_cost,
                            options={'disp': True, 'maxiter': maxiter})
                
            inv.print_info(var.cost,res)
            vector_kt = res['x']
            _, Ca = model.do_forward_jit(vector_kt)
            Ua, Va = jnp.real(Ca)[0], jnp.imag(Ca)[0]
        
        # select at location
        indx = tools.nearest(forcing2D.data.lon.values,point_loc[0])
        indy = tools.nearest(forcing2D.data.lat.values,point_loc[1])
        Ua = Ua[:,indy,indx]
        U = forcing2D.data.U[:,indy,indx].values
        U_shape = forcing2D.data.U.shape
        every_U = forcing2D.data.U.values.reshape((U_shape[0],U_shape[1]*U_shape[2]))
        Uo, Vo = observations2D.get_obs()
        Uo = Uo[:,indy,indx]
        
        RMSE = tools.score_RMSE(Ua, U) 
        print('RMSE is',RMSE)
        # PLOT trajectory
        fig, ax = plt.subplots(1,1,figsize = (10,3),constrained_layout=True,dpi=dpi)
        ax.plot(forcing2D.time/86400, U, c='k', lw=2, label='Croco')
        ax.plot(forcing2D.time/86400, every_U, c='k', lw=2, alpha=0.1)
        ax.plot(forcing2D.time/86400, Ua, c='g', label='Unstek')
        ax.scatter(observations2D.time_obs/86400,Uo, c='r', label='obs')
        ax.set_ylim([-0.3,0.4])
        ax.set_title('RMSE='+str(jnp.round(RMSE,4))+' cost='+str(jnp.round(var.cost(vector_kt),4)))
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Ageo zonal current (m/s)')
        ax.legend(loc=1)
        fig.savefig(path_save_png+'JAX_test_junstek1D_kt_spatial_'+str(Nl)+'layers'+namesave_loc_area+'.png')
    
    if TEST_CLASSIC_SLAB:
        print('* test classic_slab1D, N='+str(Nl)+' layers')
        
            
        model = classic_NIO_models.classic_slab1D(dt, Nl, forcing1D)
        var = inv.Variational(model, observations1D)    
        
        t1 = clock.time()
        _, Ca = model.do_forward_jit(vector_k)

        Ua, Va = jnp.real(Ca)[0], jnp.imag(Ca)[0]
        t2 = clock.time()
        print('time, forward model (with compile)',t2-t1)
        
        _, Ca = model.do_forward_jit(vector_k)
        Ua, Va = jnp.real(Ca)[0], jnp.imag(Ca)[0]
        print('time, forward model (no compile)',clock.time()-t2)
        
        t3 = clock.time()
        J = var.cost(vector_k)
        print('time, cost (with compile)',clock.time()-t3)

        t4 = clock.time()
        J = var.cost(vector_k)
        print('time, cost (no compile)',clock.time()-t4)

        t5 = clock.time()
        dJ = var.grad_cost(vector_k)
        print('time, gradcost (with compile)',clock.time()-t5)

        t6 = clock.time()
        dJ = var.grad_cost(vector_k)
        print('time, gradcost (no compile)',clock.time()-t6)
    
        if MINIMIZE:
            print('-> minimizing ...')
            res = opt.minimize(var.cost, vector_k,
                            method='L-BFGS-B',
                            jac=var.grad_cost,
                            options={'disp': True, 'maxiter': maxiter})
                
            inv.print_info(var.cost,res)
            vector_k = res['x']
            _, Ca = model.do_forward_jit(vector_k)
            Ua, Va = jnp.real(Ca)[0], jnp.imag(Ca)[0]
    
        # PLOT
        U = forcing1D.data.U.values
        Uo, Vo = observations1D.get_obs()
        
        RMSE = tools.score_RMSE(Ua, U) 
        print('RMSE is',RMSE)
        # PLOT trajectory
        fig, ax = plt.subplots(1,1,figsize = (10,3),constrained_layout=True,dpi=dpi)
        ax.plot(forcing1D.time/86400, U, c='k', lw=2, label='Croco')
        ax.plot(forcing1D.time/86400, Ua, c='g', label='slab')
        ax.scatter(observations1D.time_obs/86400,Uo, c='r', label='obs')
        ax.set_ylim([-0.3,0.4])
        ax.set_title('RMSE='+str(jnp.round(RMSE,4))+' cost='+str(jnp.round(var.cost(vector_k),4)))
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Ageo zonal current (m/s)')
        ax.legend(loc=1)
        fig.savefig(path_save_png+'JAX_test_classic_slab1D_'+str(Nl)+'layers'+namesave_loc+'.png')
        
    if TEST_CLASSIC_SLAB_KT:
        print('* test classic_slab1D_kt, N='+str(Nl)+' layers')
        model = classic_NIO_models.classic_slab1D_Kt(dt, Nl, forcing1D, dT)
        var = inv.Variational(model, observations1D)    
        
        # transform into 1D vector
        vector_kt = model.kt_ini(vector_k)
        print('vector_kt',vector_kt)
        
        t1 = clock.time()
        _, Ca = model.do_forward_jit(vector_kt)

        Ua, Va = jnp.real(Ca)[0], jnp.imag(Ca)[0]
        t2 = clock.time()
        print('time, forward model (with compile)',t2-t1)
        
        _, Ca = model.do_forward_jit(vector_kt)
        Ua, Va = jnp.real(Ca)[0], jnp.imag(Ca)[0]
        print('time, forward model (no compile)',clock.time()-t2)
        
        t3 = clock.time()
        J = var.cost(vector_kt)
        print('time, cost (with compile)',clock.time()-t3)

        t4 = clock.time()
        J = var.cost(vector_kt)
        print('time, cost (no compile)',clock.time()-t4)

        t5 = clock.time()
        dJ = var.grad_cost(vector_kt)
        print('time, gradcost (with compile)',clock.time()-t5)

        t6 = clock.time()
        dJ = var.grad_cost(vector_kt)
        print('time, gradcost (no compile)',clock.time()-t6)
    
        if MINIMIZE:
            print('-> minimizing ...')
            res = opt.minimize(var.cost, vector_kt,
                            method='L-BFGS-B',
                            jac=var.grad_cost,
                            options={'disp': True, 'maxiter': maxiter})
                
            inv.print_info(var.cost,res)
            vector_kt = res['x']
            _, Ca = model.do_forward_jit(vector_kt)
            Ua, Va = jnp.real(Ca)[0], jnp.imag(Ca)[0]
    
        # PLOT
        U = forcing1D.data.U.values
        Uo, Vo = observations1D.get_obs()
        
        RMSE = tools.score_RMSE(Ua, U) 
        print('RMSE is',RMSE)
        # PLOT trajectory
        fig, ax = plt.subplots(1,1,figsize = (10,3),constrained_layout=True,dpi=dpi)
        ax.plot(forcing1D.time/86400, U, c='k', lw=2, label='Croco')
        ax.plot(forcing1D.time/86400, Ua, c='g', label='slab')
        ax.scatter(observations1D.time_obs/86400,Uo, c='r', label='obs')
        ax.set_ylim([-0.3,0.4])
        ax.set_title('RMSE='+str(jnp.round(RMSE,4))+' cost='+str(jnp.round(var.cost(vector_kt),4)))
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Ageo zonal current (m/s)')
        ax.legend(loc=1)
        fig.savefig(path_save_png+'JAX_test_classic_slab1D_Kt_'+str(Nl)+'layers'+namesave_loc+'.png')
        
    if TEST_CLASSIC_SLAB_EQX:
        print('* test classic_slab1D_eqx, N='+str(Nl)+' layers')
        
        
        # variables
        U0 = 0.0
        V0 = 0.0
        # control vector
        pk = jnp.asarray([-11.31980127, -10.28525189])
        # parameters
        TAx = jnp.asarray(forcing1D.TAx)
        TAy = jnp.asarray(forcing1D.TAy)
        fc = jnp.asarray(forcing1D.fc)
        
        t0 = 0.
        t1 = 20*3600. #28*86400.
        dt_forcing=3600.
        dt_saveat = dt_forcing #60.
        
        mymodel = classic_NIO_models.classic_slab1D_eqx(U0, V0, pk, TAx, TAy, fc, dt_forcing, Nl)
        
        time1 = clock.time()
        C = mymodel(t0,t1,dt,SaveAt=dt_saveat)
        print('time, forward model (with compile)',clock.time()-time1)
        
        time2 = clock.time()
        C = mymodel(t0,t1,dt,SaveAt=dt_saveat)
        print('time, forward model (no compile)',clock.time()-time2)
        
        
    end = clock.time()
    print('Total execution time = '+str(jnp.round(end-start,2))+' s')
    plt.show()
