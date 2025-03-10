"""
Main script for JAXathon organized at IGE on 10th to 13th of March 2025

run with:
python main.py
"""
import jax.numpy as jnp
import numpy as np
import time as clock
import scipy.optimize as opt
import matplotlib.pyplot as plt

from forcing import *
from observation import *
from models import *
from inv import *
from tools import *

###############################################
# PARAMETERS                                  #
###############################################
# //
N_CPU       = 8         # when using joblib, if >1 then use // code
JAXIFY      = False      # whether to use JAX or not

# Model definition
Nl                  = 2         # number of layers
dt                  = 60        # timestep of the model (s) 

# OSSE
dt_OSSE             = 3600      # timestep of the OSSE (s)

# Observations for 4Dvar
period_obs          = 86400     # s, how many second between observations

# MINIMIZATION

maxiter             = 100       # max number of iteration


# What do you want to do ?
MINIMIZE                = True     # start a minimisation process
TIME_FORWARD_BACKWARD   = False     # measure the time for forward/backward of model

# regrid data
path_regrid = 'croco_1h_inst_surf_2006-02-01-2006-02-28_cons_-50.0_-35.0.nc'

# PLOT
dpi=200
path_save_png = './'
###############################################


# MAIN LOOP
# This avoids infinite subprocess creation
if __name__ == "__main__": 
    
    point_loc = [-50.,35.]
    
    # How I built the data file
    ds = xr.open_mfdataset('../data_regrid/croco_1h_inst_surf_2006-02-01-2006-02-28_0.1deg_conservative.nc')
    ix = nearest(ds.lon.values,point_loc[0])
    iy = nearest(ds.lat.values,point_loc[1])
    print(ix,iy)
    ds = ds.isel(lat=slice(iy-1,iy+2),lon=slice(ix-1,ix+2))
    ds.to_netcdf(path_regrid)
    raise Exception

    # Initialisation of parameter vector
    if Nl==1:
        vector_k = np.asarray([-11.31980127, -10.28525189])
    elif Nl==2:
        vector_k = np.asarray([-10.76035344, -9.3901326, -10.61707124, -12.66052074])
    else:
        vector_k = np.linspace(-12,-4,Nl*2)        
    
    print('\nNumber of layers Nl =',Nl)
    print('Initial vector_k is:\n'+str(vector_k)+'\n')
    
    # Initialisation of model, forcing, obs and 4Dvar
    forcing1D = Forcing1D(point_loc, dt_OSSE, path_regrid)
    observations1D = Observation1D(point_loc, period_obs, dt_OSSE, path_regrid)
    if JAXIFY:
        vector_k = jnp.asarray(vector_k)
        model = jUnstek1D(dt, Nl, forcing1D)
    else:
        model = Unstek1D(dt, Nl, forcing1D, observations1D)
    var = Variational(model, observations1D)  
    
    
    if TIME_FORWARD_BACKWARD:
        print('* Time deltas for running the model (Nl='+str(Nl)+')')
        
        if JAXIFY:
        
            t1 = clock.time()
            _, Ca = model.do_forward_jit(vector_k)
            Ua, Va = np.real(Ca)[0], np.imag(Ca)[0]
            t2 = clock.time()
            print('time, forward model (with compile)',t2-t1)
        
            _, Ca = model.do_forward_jit(vector_k)
            Ua, Va = np.real(Ca)[0], np.imag(Ca)[0]
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
            
        else:
            
            t1 = clock.time()
            _, Ca = model.do_forward(vector_k)
            Ua, Va = np.real(Ca)[0], np.imag(Ca)[0]
            t2 = clock.time()
            print('time, forward model',t2-t1)
            
            t3 = clock.time()
            J = var.cost(vector_k)
            print('time, cost',clock.time()-t3)

            t5 = clock.time()
            dJ = var.grad_cost(vector_k)
            print('time, gradcost',clock.time()-t5)
            
    if MINIMIZE:
        print('* Minimizing ...')
        res = opt.minimize(var.cost, vector_k,
                        method='L-BFGS-B',
                        jac=var.grad_cost,
                        options={'disp': True, 'maxiter': maxiter})
            
        print_info(var.cost,res)
        vector_k = res['x']
        if JAXIFY:
            _, Ca = model.do_forward_jit(vector_k)
        else:
            _, Ca = model.do_forward(vector_k)
        Ua, Va = np.real(Ca)[0], np.imag(Ca)[0]
    
        # PLOT
        U = forcing1D.data.U.values
        Uo, Vo = observations1D.get_obs()
        
        RMSE = score_RMSE(Ua, U) 
        print('RMSE is',RMSE)
        # PLOT trajectory
        fig, ax = plt.subplots(1,1,figsize = (10,3),constrained_layout=True,dpi=dpi)
        ax.plot(forcing1D.time/86400, U, c='k', lw=2, label='Croco')
        ax.plot(forcing1D.time/86400, Ua, c='g', label='Unstek')
        ax.scatter(observations1D.time_obs/86400,Uo, c='r', label='obs')
        ax.set_ylim([-0.3,0.4])
        ax.set_title('RMSE='+str(np.round(RMSE,4))+' cost='+str(np.round(var.cost(vector_k),4)))
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Ageo zonal current (m/s)')
        ax.legend(loc=1)
        fig.savefig(path_save_png+'minimization_test_'+type(model).__name__+'_'+str(Nl)+'layers.png')