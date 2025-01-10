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
from dask.distributed import Client,LocalCluster
import matplotlib as mpl
import sys

start = clock.time()
##############
# PARAMETERS #
##############
# //
DASHBOARD = False   # when using dask
N_CPU = 12           # when using joblib

# -> area of interest
box= [-25, -24, 45., 46, 24000., 24010.] # bottom lat, top lat, left lon, right lon
# -> index for spatial location. We work in 1D
ir=4
jr=4

# -> MINIMIZATION OF THE UNSTEADY EKMAN MODEL
dt = 60 # timestep of the model (s) 
# LAYER DEFINITION
#        number of values = number of layers
#        values = turbulent diffusion coefficient
pk = np.array([-3,-12]) # [-3,-12]         # 1 layers
#pk=np.array([-3,-8,-10,-12])    # 2 layers
# pk=np.array([-2,-4,-6,-8,-10,-12]) # 3 layers
maxiter=100   # number of iteration max for MINIMIZE

# -> ANALYSIS    
MINIMIZE                = True     # find the vector K starting from 'pk'
PLOT_TRAJECTORY         = True     # plot u(t) for a specific vector_k
ONE_LAYER_COST_MAP      = False      # maps the cost function values
TWO_LAYER_COST_MAP_K1K2 = False     # maps the cost function values, K0 K4 fixed


# -> PLOT
dpi=200

# -> List of files
filesUV = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/U_V/llc2160_2020-11-*_SSU-SSV.nc"))
filesH = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/SSH/llc2160_2020-11-*_SSH.nc"))
filesW = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/v10m/llc2160_2020-11-*_v10m.nc"))
filesD = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/KPPhbl/llc2160_2020-11-*_KPPhbl.nc"))

# -> list of save path
path_save_png1D = './png_1D/'
path_save_LS = './'
path_save_interp1D = './'

# END PARAMETERS #################################


if __name__ == "__main__":  # This avoids infinite subprocess creation
    
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
    #build_LS_files(filesUV,box, my2dfilter, mytimefilter, path_save)
    build_LS_files(filesH, box, path_save_LS)
    #build_LS_files(filesW,box, my2dfilter, mytimefilter, path_save)
    #build_LS_files(filesD,box, my2dfilter, mytimefilter, path_save)
    dsSSH_LS = xr.open_dataset('LS_fields_SSH_'+str(box[0])+'_'+str(box[1])+'_'+str(box[2])+'_'+str(box[3])+'_'+str(box[4])+'_'+str(box[5])+'.nc')
    gUg,gVg = dsSSH_LS['Ug'],dsSSH_LS['Vg']
    # ---------------------------
    
    ## INTERPOLATION 
    # This is interpolation on reconstructed time grid (every dt)
    # this is necessary for the simple model 'unstek'
    print('* interpolation on unsteady ekman model timestep')
    list_files = list(filesUV) + list(filesH) + list(filesW) + list(filesD)
    interp_at_model_t_1D(dsSSH_LS, dt, ir, jr, list_files, box, path_save_interp1D)
    ds1D_i = xr.open_dataset('Interp_1D_LON-24.8_LAT45.2.nc')
    
    U,V,TAx,TAy,MLD = ds1D_i.SSU.values,ds1D_i.SSV.values,ds1D_i.TAx.values,ds1D_i.TAy.values,ds1D_i.MLD
    fc = 2*2*np.pi/86164*np.sin(ds1D_i.lat.values*np.pi/180) # Coriolis value at jr,ir
    nt = len(ds1D_i.time)
    time = np.arange(0,nt*dt,dt)
    # -----------------------------------------------
    
    # OBSERVATIONS from MITgcm, 1 per day
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
    # -----------------------------------------------
    
    # ANALYSIS
    #
    # minimization procedure of the cost function (n layers)
    if MINIMIZE:
        Nlayers = len(pk)//2
        print('* Minimization with '+str(Nlayers)+' layers')
 
        #J = cost(pk, time, fc, TAx.values, TAy.values, Uo.values, Vo.values, Ri.values)
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
        plt.savefig(path_save_png1D+'series_reconstructed_long_'+str(Nlayers)+'layers.png')
        
    # Plotting trajectories of given vector_k
    # plotting tool for trajectory of 'unstek' for specific vector_k
    if PLOT_TRAJECTORY:
        print('* Plotting trajectories/step response for different vector K')
        list_k = [[-3.63133021,  -9.46349552], # 1 layer, minimization from [-3,-12]
                [-5,-12], # 1 layer, hand chosen
                [-2,-7], # 1 layer, hand chosen
                [ -3.47586165, -9.06189063, -11.22302904, -12.43724667], # 2 layer, minimization from [-3,-8,-10,-12]
                ]
        PLOT_MLD = True
                
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
            plt.savefig(path_save_png1D+'series_reconstructed_'+str(len(vector_k)//2)+'layers'+txt_add+'.png')

            # step response
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
    
        PLOT_ITERATIONS = True
        PARALLEL = True
        kmin = -15
        kmax = 0
        step = 0.25
        maxiter = 10
        vector_k = np.array([-12,-12]) # initial vector k
        Jmap_cmap = 'terrain'
        
        tested_values = np.arange(kmin,kmax,step)  # -15,0,0.25

        if PARALLEL:
            J = Parallel(n_jobs=N_CPU)(delayed(
                    cost)(np.array([k0,k1]), time, fc, TAx, TAy, Uo, Vo, Ri)
                                    for k0 in tested_values for k1 in tested_values)
            J = np.transpose(np.reshape(np.array(J),(len(tested_values),len(tested_values)) ))
        else:
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
        vecteur solution :  [ -3.47586165  -9.06189063 -11.22302904 -12.43724667]
        cost function value : 0.28825167461378703
        """
        PARALLEL = True
        k0 = -3.47586165
        k3 = -12.43724667
        kmin = -15
        kmax = -4
        step = 0.25
        Jmap_cmap = 'terrain'
        
        tested_values = np.arange(kmin,kmax,step)
        
        if PARALLEL:
            J = Parallel(n_jobs=N_CPU)(delayed(
                    cost)(np.array([k0,k1,k2,k3]), time, fc, TAx, TAy, Uo, Vo, Ri)
                                    for k1 in tested_values for k2 in tested_values)
            J = np.transpose(np.reshape(np.array(J),(len(tested_values),len(tested_values)) ))
        else:
            J = np.zeros((len(tested_values),len(tested_values)))
            for i,k1 in enumerate(tested_values):
                for j,k2 in enumerate(tested_values):
                    vector_k = np.array([k0,k1,k2,k3])
                    J[j,i] = cost(vector_k, time, fc, TAx, TAy, Uo, Vo, Ri)
        
        # plotting
        cmap = mpl.colormaps.get_cmap(Jmap_cmap)
        cmap.set_bad(color='indianred')
        fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
        s = ax.pcolormesh(tested_values,tested_values,J,cmap=cmap,norm=matplotlib.colors.LogNorm(0.1,1000)) #,vmin=0.1,vmax=100
        plt.colorbar(s,ax=ax)
        ax.scatter(-9.06189063, -11.22302904,marker='x',c='g')
        ax.set_xlabel('log(k1)')
        ax.set_ylabel('log(k2)')
        plt.savefig(path_save_png1D+'k1k2_J_2layer.png')

    
    
    end = clock.time()
    print('Total execution time = '+str(np.round(end-start,2))+' s')
    plt.show()