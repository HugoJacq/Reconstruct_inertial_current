import os
import sys
sys.path.insert(0,'..')
from model_unstek import * # old

import matplotlib.pyplot as plt
import scipy.optimize as opt
from optimparallel import minimize_parallel
import time as clock

from unstek import *
from forcing import *

import jax
import jax.numpy as jnp

# initial vector_k
pk = np.array([-3,-12]) # 1 layer
#pk = np.array([-3.61402696, -9.44992617]) # solution of minimization
dt = 60 # s

COMPARE = False
path_file = '../Interp_1D_LON-24.8_LAT45.2.nc'
TRUE_WIND_STRESS = False
period_obs = 86400 # s, how many second between observations

MINIMIZE = True
maxiter = 100
PARALLEL_MINIMIZED = False
N_CPU = 12
PRINT_INFO = False

# Plot
dpi=200

##########
# Normal #
##########
if False:
    t1 = clock.time()
    forcing = Forcing1D(dt, path_file, TRUE_WIND_STRESS)   
    model = Unstek1D(1, forcing)
    Uo,Vo = forcing.get_obs(period_obs)
    U, V = forcing.U, forcing.V
    Ri=Uo*0.+1 
       
    # minimization to find k        
    if MINIMIZE:
        if PARALLEL_MINIMIZED:
            res = minimize_parallel(fun=model.cost, x0=pk, args=(Uo, Vo, Ri),
                            jac=model.grad_cost,
                            parallel={'loginfo': True, 'max_workers':N_CPU,'verbose':PRINT_INFO,'time':True},
                            options={'maxiter': maxiter})
        else:
            res = opt.minimize(model.cost, pk, args=(Uo, Vo, Ri),
                            method='L-BFGS-B',
                            jac=model.grad_cost,
                            options={'disp': True, 'maxiter': maxiter})
            
        if np.isnan(model.cost(res['x'], Uo, Vo, Ri)):
            print('The model has crashed.')
        else:
            print(' vector K solution ('+str(res.nit)+' iterations)',res['x'])
            print(' cost function value with K solution:',model.cost(res['x'], Uo, Vo, Ri))
        pk = res['x']
    
    Ua, Va = model.do_forward(pk)
    RMSE = score_RMSE(Ua, U)     
    
    # PLOT
    title = ''
    for k in range(len(res['x'])):
        title += 'k'+str(k)+','   
    title = title[:-1]+' = '+str(res['x']) + ' ('+str(np.round(RMSE,3))+')'
    plt.figure(figsize=(10,3),dpi=dpi)
    plt.plot(forcing.time/86400,U, c='k', lw=2, label='LLC ref')
    plt.plot(forcing.time/86400,Ua, c='g', label='Unstek')
    plt.scatter(forcing.time/86400,Uo, c='r', label='obs')
    plt.title(title)
    plt.xlabel('Time (days)')
    plt.ylabel('Ageo zonal current (m/s)')
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig('NOJAX_series_reconstructed_long_'+str(model.nl)+'layers.png')
    print('-> time for no jax = ',np.round(clock.time() - t1,2) )

###############
# JAX version #
###############
if True:
    t1 = clock.time()
    print('WIP')
    print('-> time for with jax = ',np.round(clock.time() - t1,2) )
        
plt.show()

###############
# JAX version #
###############
"""
DEBUGING:

- jax.debug.print() for traced (dynamic) array values with jax.jit(), jax.vmap() and others
- print() for static values, such as dtypes and array shapes
"""

