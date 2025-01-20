"""
This file is the main file of the study 'Inertial_current_from_MITgcm'

python reconstruct_intertial.py

by: Hugo Jacquet and Clément Ubelmann

Updates: 
    - CU 6/01/25: creation 
    - HJ 7/01/25: add comments everywhere
    - HJ 9/01/25: add build large scale variable file
"""
import time as clock
import scipy.optimize as opt
from dask.distributed import Client,LocalCluster
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import glob
import os
from optimparallel import minimize_parallel

# custom imports
from build_LS_fields import *
from observations import *
from forcing import *
from unstek import *
from inv import *
from tools import *
from scores import *

start = clock.time()
##############
# PARAMETERS #
##############
# //
DASHBOARD = False   # when using dask
N_CPU = 12          # when using joblib

# -> area of interest
box= [-25, -24, 45., 46, 24000., 24010.] # bottom lat, top lat, left lon, right lon
# -> index for spatial location. We work in 1D
ir=4
jr=4


# observations : OSSE
TRUE_WIND_STRESS = False     # whether to use Cd.U**2 or Tau
dt = 60                     # timestep of the model (s) 
period_obs = 86400 # s, how many second between observations

# LAYER DEFINITION
#        number of values = number of layers
#        values = turbulent diffusion coefficient
vector_k = np.array([-3,-12]) # [-3,-12]         # 1 layers
#vector_k=np.array([-3,-8,-10,-12])    # 2 layers

# -> MINIMIZATION OF THE UNSTEADY EKMAN MODEL
MINIMIZE = True
PRINT_INFO = False          # only if PARALLEL_MINIMIZED
save_iter = False           # save iteration during minimize
maxiter = 100
PARALLEL_MINIMIZED = False # there is a bug with TRUE
N_CPU = 12
PRINT_INFO = False

# -> ANALYSIS   
MAP_1D_LOCATION         = False 
MINIMIZE                = False     # find the vector K starting from 'pk'
PLOT_TRAJECTORY         = False     # plot u(t) for a specific vector_k
ONE_LAYER_COST_MAP      = False      # maps the cost function values
TWO_LAYER_COST_MAP_K1K2 = True     # maps the cost function values, K0 K4 fixed
# note: i need to tweak score_PSD with rotary spectra

# -> PLOT
dpi=200

# -> List of files
filesUV = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/U_V/llc2160_2020-11-*_SSU-SSV.nc"))
filesH = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/SSH/llc2160_2020-11-*_SSH.nc"))
filesW = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/v10m/llc2160_2020-11-*_v10m.nc"))
filesD = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/KPPhbl/llc2160_2020-11-*_KPPhbl.nc"))
filesTau = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/oceTau/llc2160_2020-11-*_oceTAUX-oceTAUY.nc"))

# -> list of save path
path_save_png1D = './png_1D'
path_save_LS = './'
path_save_interp1D = './'

# END PARAMETERS #################################
if TRUE_WIND_STRESS: path_save_png1D = path_save_png1D + '_Tau/'
else: path_save_png1D = path_save_png1D + '_UU/'
    

if __name__ == "__main__":  # This avoids infinite subprocess creation
    
    if not os.path.isdir(path_save_png1D):
        os.system('mkdir '+path_save_png1D)
    
    global client
    client = None
    if DASHBOARD:
        # sometimes dask cluster can cause problems "memoryview is too large"
        # (writing a big netcdf file for eg, hbudget_file)
        cluster = LocalCluster(threads_per_worker=32,n_workers=8)
        
        client = Client(cluster)
        print("Dashboard at :",client.dashboard_link)
    
    # LARGE SCALE FIELDS --------
    print('* Getting large scale motion ...')
    #build_LS_files(filesUV,box, path_save_LS)
    build_LS_files(filesH, box, path_save_LS)
    #build_LS_files(filesW,box, path_save_LS)
    #build_LS_files(filesD,box, path_save_LS)
    dsSSH_LS = xr.open_dataset('LS_fields_SSH_'+str(box[0])+'_'+str(box[1])+'_'+str(box[2])+'_'+str(box[3])+'_'+str(box[4])+'_'+str(box[5])+'.nc')
    # ---------------------------
    
    ## INTERPOLATION 
    # This is interpolation on reconstructed time grid (every dt)
    # this is necessary for the simple model 'unstek'
    print('* interpolation on unsteady ekman model timestep')
    list_files = list(filesUV) + list(filesH) + list(filesW) + list(filesD) + list(filesTau)
    interp_at_model_t_1D(dsSSH_LS, dt, ir, jr, list_files, box, path_save_interp1D)
    
    # observation and forcing
    path_file = 'Interp_1D_LON-24.8_LAT45.2.nc'
    forcing = Forcing1D(dt, path_file, TRUE_WIND_STRESS)   
    observations = Observation1D(period_obs, dt, path_file)
    Uo,Vo = observations.get_obs()
    U, V = forcing.U, forcing.V
    
    # intialisation of the model
    Nl = len(vector_k)//2
    model = Unstek1D(Nl, forcing, observations)
    var = Variational(model, observations)
        
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
        print('tbd')
    
    # minimization procedure of the cost function (n layers)
    if MINIMIZE:
        print('* Minimization with '+str(Nl)+' layers')
        t1 = clock.time()
        _, Ca = model.do_forward(vector_k)
        Ua, Va = jnp.real(Ca), jnp.imag(Ca)
            
        J = var.cost(vector_k)
        print('J',J)
        t2 = clock.time()
        dJ = var.grad_cost(vector_k)
        print('dJ',dJ)
        print('-> time for J = ',np.round(t2 - t1,4) )
        print('-> time for dJ = ',np.round(clock.time() - t2,4) )
        
        if PARALLEL_MINIMIZED:
            # this does not work with Unstek1D class
            res = minimize_parallel(fun=var.cost, x0=vector_k, #  args=(time, fc, TAx, TAy, Uo, Vo, Ri)
                              jac=var.grad_cost,
                              parallel={'loginfo': True, 'max_workers':N_CPU,'verbose':PRINT_INFO,'time':True},
                              options={'maxiter': maxiter})
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
        Ua, Va = np.real(Ca),np.imag(Ca)
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
        plt.savefig(path_save_png1D+'series_reconstructed_long_'+str(Nl)+'layers.png')
        
        # Plot MLD
        fig, ax = plt.subplots(2,1,figsize = (10,6),constrained_layout=True,dpi=dpi)
        ax[0].plot(forcing.time/86400,U, c='k', lw=2, label='LLC ref')
        ax[0].plot(forcing.time/86400,Ua, c='g', label='Unstek')
        ax[0].scatter(observations.time_obs/86400,Uo, c='r', label='obs')
        ax[0].set_title(title)
        ax[0].set_ylabel('Ageo zonal current (m/s)')
        ax[0].legend(loc=1)
        ax[1].plot(forcing.time/86400, - forcing.MLD, c='k')
        ax[1].set_xlabel('Time (days)')
        ax[1].set_ylabel('MLD (m)')
        fig.savefig(path_save_png1D+'series_reconstructed_long_'+str(Nl)+'layers_withMLD.png')
      
    # Plotting trajectories of given vector_k
    # plotting tool for trajectory of 'unstek' for specific vector_k
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
            plt.savefig(path_save_png1D+'series_reconstructed_'+str(len(vector_k)//2)+'layers'+txt_add+'.png')

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
                fig.savefig(path_save_png1D+'Hodograph_from_stepTau_'+str(len(vector_k)//2)+'layers'+txt_add+'.png')    
      
    # looking at the cost function
    #   for a slab model (1 layer)
    if ONE_LAYER_COST_MAP:
        print('* Looking at cost function for k0,k1 in 1 layer model')
    
        PLOT_ITERATIONS = False
        kmin = -15
        kmax = 0
        step = 0.25
        maxiter = 10
        vector_k = np.array([-12,-12]) # initial vector k
        Jmap_cmap = 'terrain'
        
        tested_values = np.arange(kmin,kmax,step)  # -15,0,0.25

        if N_CPU>1:
            J = Parallel(n_jobs=N_CPU)(delayed(
                    var.cost)(np.array([k0,k1]))
                        for k0 in tested_values for k1 in tested_values)
            J = np.transpose(np.reshape(np.array(J),(len(tested_values),len(tested_values)) ))
        else:
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
        plt.savefig(path_save_png1D+'k0k1_J_1layer'+txt_add+'.png')
        
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
        with Cd.U**2:
            vecteur solution :  [ -3.47586165  -9.06189063 -11.22302904 -12.43724667]
            cost function value : 0.28825167461378703
        with true Tau:
            vecteur solution :  [ -9.14721791, -8.79469884, -11.20512638, -12.5794675]
            cost function value : 0.24744266411643656
        """
        PARALLEL = True
        step = 0.25
        Jmap_cmap = 'terrain'
        if TRUE_WIND_STRESS:
            k0 = -9.14721791
            k3 = -12.5794675
            k1_mini = -8.79469884
            k2_mini = -11.20512638
            kmin,kmax = -15, -1
        else:
            k0 = -3.47586165
            k3 = -12.43724667
            k1_mini = -9.06189063
            k2_mini = -11.22302904
            kmin,kmax = -15, -4
        
        tested_values = np.arange(kmin,kmax,step)
        Nl = 2
        model = Unstek1D(Nl, forcing, observations)
        var = Variational(model, observations)
    
        if N_CPU>1:
            J = Parallel(n_jobs=N_CPU)(delayed(
                    var.cost)(np.array([k0,k1,k2,k3]))
                                    for k1 in tested_values for k2 in tested_values)
            J = np.transpose(np.reshape(np.array(J),(len(tested_values),len(tested_values)) ))
        else:
            J = np.zeros((len(tested_values),len(tested_values)))
            for i,k1 in enumerate(tested_values):
                for j,k2 in enumerate(tested_values):
                    vector_k = np.array([k0,k1,k2,k3])
                    J[j,i] = var.cost(vector_k)
        
        # plotting
        cmap = mpl.colormaps.get_cmap(Jmap_cmap)
        cmap.set_bad(color='indianred')
        fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
        s = ax.pcolormesh(tested_values,tested_values,J,cmap=cmap,norm=mpl.colors.LogNorm(0.1,10000)) #,vmin=0.1,vmax=100
        plt.colorbar(s,ax=ax)
        ax.scatter(k1_mini, k2_mini,marker='x',c='g')
        ax.set_xlabel('log(k1)')
        ax.set_ylabel('log(k2)')
        plt.savefig(path_save_png1D+'k1k2_J_2layer.png')

    
    
    end = clock.time()
    print('Total execution time = '+str(np.round(end-start,2))+' s')
    plt.show()