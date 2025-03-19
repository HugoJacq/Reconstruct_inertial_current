"""
This modules gather the definition of classical models used in the litterature 
to reconstruct inertial current from wind stress
Each model needs a forcing, from the module 'forcing.py'

The models are written in JAX to allow for automatic differentiation

refs:
Wang et al. 2023: https://www.mdpi.com/2072-4292/15/18/4526
Stokes et al. 2024: https://journals.ametsoc.org/view/journals/phoc/54/3/JPO-D-23-0167.1.xml
Pollard 1970: https://linkinghub.elsevier.com/retrieve/pii/0011747170900422
Price et al. 1986 (PWP model): https://doi.org/10.1029/JC091iC07p08411

By: Hugo Jacquet fev 2025
"""
import numpy as np
import xarray as xr
import warnings
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
import equinox as eqx
from diffrax import ODETerm, diffeqsolve, Euler
import diffrax

from .constants import FORCE_CPU

# for jax
cpu_device = jax.devices('cpu')[0]
try:
    gpu_device = jax.devices('gpu')[0]
except:
    gpu_device = cpu_device
if FORCE_CPU:
    jax.config.update('jax_platform_name', 'cpu')


class classic_slab1D:
    """
        Originaly from Pollard (1970), implemented as:
                dU/dt = -ifU + dtau/dz -rU 
            
        1 layer (= "slab model")
                dU/dt = -ifU + Tau_0/H - rU    
            
            parameters = (H,r)
            minimized version:
                dU/dt = -ifU + K0.Tau_0 - K1.U
                
        2 layers
            i=1     dU1/dt = -ifU1 + (tau_0-tau_i)/H1 - rU1
            i=2     dU2/dt = -ifU2 + (tau_i)/H2 - rU2
            
            parameters = (tau_i,H1,H2,r) or (tau_i,H1,H2,r1,r2) if =/= r in between the layers
            minimized version:
                    dU1/dt = -ifU1 + K0.(tau_0 - K1) - K3.U1
                    dU2/dt = -ifU2 + K2.K1 - K3.U2
                    
        3 layers
            i=1     dU1/dt = -ifU1 + (tau_0-tau_i1)/H1 - rU1
            i=2     dU2/dt = -ifU2 + (tau_i1-tau_i2)/H2 - rU2
            i=3     dU3/dt = -ifU3 + (tau_i2)/H3 - rU3
            
            parameters = (tau_i1,tau_i2,H1,H2,H3,r) or (tau_i,H1,H2,r1,r2) if =/= r in between the layers
            minimized version:
                    dU1/dt = -ifU1 + K0.(tau_0 - K1) - K5.U1
                    dU2/dt = -ifU2 + K3.(K1-K2) - K5.U2
                    dU3/dt = -ifU3 + K4.K2 - K5.U3
                    
        """
    def __init__(self, dt, Nl, forcing):
        """
        dt : model time step
        Nl : number of layers
        forcing : class Forcing1D
        """
        # from dataset
        self.nt = len(forcing.time)
        self.dt_forcing = forcing.dt_forcing
        self.dt = int(dt)
        self.ntm = forcing.time[-1]//dt
        self.ntf = forcing.time[-1]//dt
        self.forcing_time = forcing.time
        self.model_time = jnp.arange(0,self.ntm*self.dt,self.dt)
        # forcing
        self.TA = jnp.asarray(forcing.TAx) + 1j*jnp.asarray(forcing.TAy) # wind = windU + j*windV
        self.fc = jnp.asarray(forcing.fc)  
        # for reconstructed 
        self.nl = Nl
        self.isJax = True     
        self.ktrf_method = 'exp'
        
        # JIT compiled functions
        self.do_forward_jit = jit(self.do_forward)
        self.K_transform_jit = jit(self.K_transform)  
    
    def K_transform(self, pk, order=0, function='exp'):
        """
        Transform the K vector to improve minimization process
        
        INPUT:
            - pk : K vector
            - order : 1 is derivative
            - function : string of the name of the function
        OUTPUT:
            - f(k) or f'(k)
        """  
        if function=='exp':
            return jnp.exp(pk)
        elif function=='id':
            return pk
        else:
            raise Exception('K_transform function '+function+' is not available, retry')
    def K_transform_reverse(self, K, function='exp'):
        """
        function is the forward function
        """
        if function=='exp':
            return jnp.log(K)
        elif function=='id':
            return K
        else:
            raise Exception('K_transform_reverse function '+function+' is not available, retry')

    def __Onelayer(self, arg2):
        """
        1 time iteration of unstek model when 1 layer
        """
         
        it, K, TA, U = arg2
        ik = 0
        U = U.at[ik].set( U[ik] + self.dt*( 
                                           -1j*self.fc*U[ik] 
                                           + K[ik]*TA 
                                           - K[-1]*U[ik] ) )
        return it, K, TA, U
        
    def __Nlayer_midlayers_for_scan(self,X0,ik): 
        it, K, U = X0
        U = U.at[ik].set( U[ik] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik] 
                                    - K[ik+1]*(K[ik-1]-K[ik])
                                    - K[-1]*U[ik] ) )
        X = it, K, U
        return X, X
        
    def __Nlayer(self,arg2):
        """
        1 time iteration of unstek model when N layer
        """    

        it, K, TA,  U = arg2
        
        # surface
        ik = 0
        U = U.at[ik].set( U[ik] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik] 
                                    + K[ik]*(TA-K[ik+1])
                                    - K[-1]*U[ik] ) )
        # bottom
        ik = -1
        U = U.at[ik].set(  U[ik] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik] 
                                    + K[ik-1]*K[ik-2]
                                    - K[-1]*U[ik] ) )
        # in between
        X0 = it, K, U
        final, _ = lax.scan(self.__Nlayer_midlayers_for_scan, X0, jnp.arange(1,self.nl-1) )
        _, _, U = final
        return it, K, TA, U    
    
    def __one_step(self, X0, inner_t):
        """
        1 model time step forward (inner loop)
        
        INPUT:
        - itm: current model time step
        - arg0 it, K, U at current model time step
            it: forcing time index
            K : K vector
            U: velocity 
        OUTPUT:
        - U: updated velocity at next time step 
        """
        it, K, U = X0
        
        
        
        nsubsteps = self.dt_forcing // self.dt
        itm = it*nsubsteps + inner_t
        
        # on-the-fly (linear) interpolation of forcing
        aa = jnp.mod(itm,nsubsteps)/nsubsteps
        itsup = lax.select(it+1>=self.nt, -1, it) 
        TA = (1-aa)*self.TA[it-1] + aa*self.TA[itsup]
                    
        # def cond_print(it):
        #     jax.debug.print('it,itf, TA, {}, {}, {}',itm,it,TA)
            
        # jax.lax.cond(it==0, cond_print, lambda x:None, it)    
            
            
        arg2 = itm, K, TA, U
        # loop on layers
        _, _, _, U = lax.cond( self.nl == 1,    # condition, only 1 layer ?
                            self.__Onelayer,    # if 1 layer, ik=0
                            self.__Nlayer,      # else, loop on layers
                            arg2)   
        X0 = it, K, U        
        return X0, X0
   
    
    def __step(self, it, arg0):
        """
        outer loop (forcing time step, 1 hour)
        """
        K, U = arg0
        #Uold = lax.select( it>0, U[:,it-1], U[:,0])
        Uold = U[:,it-1]
        X0 = it, K, Uold
        final, _ = lax.scan(self.__one_step, X0, jnp.arange(0, self.dt_forcing//self.dt))
        _, _, Unext = final
        U = U.at[:,it].set(Unext)

        return K, U
    
    def __step_for_scan(self, X0, it):
        """
        outer loop (forcing time step, 1 hour)
        """
        arg0 = X0
        X0 = self.__step(it, arg0)
        return X0, X0
    
    def do_forward(self, pk):
        U = jnp.zeros( (self.nl, self.nt), dtype='complex')
        
        K = self.K_transform(pk,function=self.ktrf_method)      # optimize inverse problem, for now nothing is done
        should_be_size = self.nl + (self.nl-1) +1   # number of Hi, number of tau_i, r
        actual_size = K.shape[-1]
        if actual_size!=should_be_size:
           raise Exception('Your model is {} layers, but you want to run it with {} layers (k={})'.format(self.nl, actual_size//2, pk))
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # dont show overflow results
            # lax.scan version
            X0 = K, U
            final, _ = lax.scan(self.__step_for_scan, X0, jnp.arange(1,self.nt)) #
            _, U = final
            
        self.Ur_traj = U
        self.Ua, self.Va = jnp.real(U[0,:]), jnp.imag(U[0,:])  # first layer
        
        return pk, U
    
class classic_slab1D_Kt:
    """
    Originaly from Pollard (1970), implemented as:
        dU/dt = -ifU + dtau/dz -rU 
        
    This model is an extension of 'classic_slab1D', parameters are allowed to vary in time.
    """
    
    def __init__(self, dt, Nl, forcing, dT):
        """
        dt : model time step
        Nl : number of layers
        forcing : class Forcing2D (on a different timestep)
        dT : seconds, period of change of K in time
        """
        # from dataset
        self.nt = len(forcing.time)
        self.dt_forcing = forcing.dt_forcing # 1 hour
        self.dt = int(dt) # dt seconds
        self.ntm = forcing.time[-1]//dt
        self.ntf = forcing.time[-1]//dt
        self.forcing_time = forcing.time #//observation.obs_period
        self.model_time = jnp.arange(0,self.ntm*self.dt,self.dt)
        # forcing
        self.TA = jnp.asarray(forcing.TAx) + 1j*jnp.asarray(forcing.TAy) # wind = windU + j*windV
        self.fc = jnp.asarray(forcing.fc)  
        # for reconstructed 
        self.nl = Nl
        self.isJax = True
        # time varying K
        self.dT = int(dT)
        self.NdT = int(forcing.time[-1]//dT + 1)
        
        # JIT compiled functions
        self.do_forward_jit = jit(self.do_forward)
        self.K_transform_jit = jit(self.K_transform)
        self.pkt2Kt_matrix_jit = jit(self.pkt2Kt_matrix, static_argnums=0)
    

    def K_transform(self, pk, order=0, function='exp'):
        """
        Transform the K vector to improve minimization process
        
        INPUT:
            - pk : K vector
            - order : 1 is derivative
            - function : string of the name of the function
        OUTPUT:
            - f(k) or f'(k)
        """  
        if function=='exp':
            return jnp.exp(pk)
        elif function=='id':
            return pk
        else:
            raise Exception('K_transform function '+function+' is not available, retry')
    def K_transform_reverse(self, K, function='exp'):
        """
        function is the forward function
        """
        if function=='exp':
            return jnp.log(K)
        elif function=='id':
            return K
        else:
            raise Exception('K_transform_reverse function '+function+' is not available, retry')

    def kt_ini(self,pk):
        a_2D = jnp.array( [pk]*self.NdT)
        return self.kt_2D_to_1D(a_2D)
    def kt_1D_to_2D(self,vector_kt_1D):
        return vector_kt_1D.reshape((self.NdT,self.nl*2))
    def kt_2D_to_1D(self,vector_kt):
        return vector_kt.flatten()


    def __step_pkt2Kt_matrix(self,ip, arg0):
        gtime,gptime,S,M = arg0
        distt = (gtime-gptime[ip])
        tmp =  jnp.exp(-distt**2/self.dT**2)
        S = lax.add( S, tmp )
        M = M.at[:,ip].set( M[:,ip] + tmp )
        return gtime,gptime,S,M
    def pkt2Kt_matrix(self, gtime):
        """
        original numpy function:
        
            def pkt2Kt_matrix(dT, gtime):
                if dT<gtime[-1]-gtime[0] : gptime = numpy.arange(gtime[0], gtime[-1]+dT,dT)
                else: gptime = numpy.array([gtime[0]])
                nt=len(gtime)
                npt = len(gptime)
                M = numpy.zeros((nt,npt))
                # Ks=numpy.zeros((ny,nx))
                S=numpy.zeros((nt))
                for ip in range(npt):
                    distt = (gtime-gptime[ip])
                    iit = numpy.where((numpy.abs(distt) < (3*dT)))[0]
                    tmp = numpy.exp(-distt[iit]**2/dT**2)
                    S[iit] += tmp
                    M[iit,ip] += tmp
                M = (M.T / S.T).T
                return M
        
        To transform in JAX compatible code, we need to work on array dimensions.
        Short description of what is done in the first 'if' statement:
            if we have a period dT shorter than the total time 'gtime',
                then we create an array with time values spaced by dT.
                last value if 'gtime[-1]'.
            else
                no variation of K along 'gtime' so only 1 vector K
                
        Example with nt = 22, with 1 dot is 1 day, dT = 5 days:
        vector_K :
            X * * * * * * * * * * * * * * * * * * * * X  # 22 values
        vector_Kt :
            X - - - - * - - - - * - - - - * - - - - * X  # 6 values
        """
        step = self.dT//self.dt
        Nstep = len(gtime)//step
        gptime = jnp.zeros( self.ntm//step+1 )        
        for ipt in range(Nstep):
            gptime = gptime.at[ipt].set( gtime[ipt*step] )

        gptime = gptime.at[-1].set( lax.select(Nstep>0, gtime[-1]+self.dT, gtime[0]) )
        gptime = gptime + self.dT
        nt = len(gtime)
        npt = len(gptime)
        M = jnp.zeros((nt,npt))
        S = jnp.zeros((nt))
        
        # loop over each dT
        arg0 = gtime, gptime, S, M
        _, _, S, M = lax.fori_loop(0, npt, self.__step_pkt2Kt_matrix, arg0)
        
        M = (M.T / S.T).T
        return M

    def __Onelayer(self, arg2):
        """
        1 time iteration of unstek model when 1 layer
        """
        it, K, TA, U = arg2
        ik = 0
        U = U.at[ik].set( U[ik] + self.dt*( 
                                           -1j*self.fc*U[ik] 
                                           + K[ik]*TA 
                                           - K[-1]*U[ik] ) )
        return it, K, TA, U
        
    def __Nlayer_midlayers_for_scan(self,X0,ik):
        it, K, U = X0
        U = U.at[ik].set( U[ik] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik] 
                                    - K[ik+1]*(K[ik-1]-K[ik])
                                    - K[-1]*U[ik] ) )
        X = it, K, U
        return X, X
        
    def __Nlayer(self,arg2):
        """
        1 time iteration of unstek model when N layer
        """        
        
        it, K, TA,  U = arg2
        
        # surface
        ik = 0
        U = U.at[ik].set( U[ik] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik] 
                                    + K[ik]*(TA-K[ik+1])
                                    - K[-1]*U[ik] ) )
        # bottom
        ik = -1
        U = U.at[ik].set(  U[ik] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik] 
                                    + K[ik-1]*K[ik-2]
                                    - K[-1]*U[ik] ) )
        # in between
        X0 = it, K, U
        final, _ = lax.scan(self.__Nlayer_midlayers_for_scan, X0, jnp.arange(1,self.nl-1) )
        _, _, U = final
        return it, K, TA, U    
    
    def __one_step(self, X0, inner_t):
        """
        1 model time step forward (inner loop)
        
        INPUT:
        - itm: current model time step
        - arg0 it, K, U at current model time step
            it: forcing time index
            K : K vector
            U: velocity 
        OUTPUT:
        - U: updated velocity at next time step 
        """
        it, Kt, U = X0
        nsubsteps = self.dt_forcing // self.dt
        itm = it*nsubsteps + inner_t
        
        # on-the-fly (linear) interpolation of forcing
        aa = jnp.mod(itm,nsubsteps)/nsubsteps
        itsup = lax.select(it+1>=self.nt, -1, it) 
        TA = (1-aa)*self.TA[it-1] + aa*self.TA[itsup]
        Ktnow = (1-aa)*Kt[it-1] + aa*Kt[itsup]
        
        arg2 = itm, Ktnow, TA, U
        # loop on layers
        _, _, _, U = lax.cond( self.nl == 1,       # condition, only 1 layer ?
                            self.__Onelayer,    # if 1 layer, ik=0
                            self.__Nlayer,      # else, loop on layers
                            arg2)   
        X0 = it, Kt, U        
        return X0, X0    
    
    def __step(self, it, arg0):
        """
        outer loop (forcing time step, 1 hour)
        """
        K, U = arg0
        #jax.debug.print('it = {}',it)
        # start = it
        # end = lax.select((it+1)>=self.nt, self.nt, it+1)
        Uold = lax.select( it>0, U[:,it-1], U[:,0])
        # arg1 = it, K, Uold
        #_, _, Unext = lax.fori_loop(start, end, self.__one_step,arg1)
        
        X0 = it, K, Uold
        final, _ = lax.scan(self.__one_step, X0, jnp.arange(0, self.dt_forcing//self.dt))
        
        _, _, Unext = final
        U = U.at[:,it].set(Unext)

        return K, U
    
    def __step_for_scan(self, X0, it):
        """
        outer loop (forcing time step, 1 hour)
        """
        arg0 = X0
        X0 = self.__step(it, arg0)
        return X0, X0
    
    def do_forward(self, pk):
        """
        Unsteady Ekman model forward model

        for 1 layer: dU/dt = -j*fc*U + K0*Tau - K1*U

        INPUT:
        - pk     : list of boundaries of layer(s)
        OUTPUT:
        - array of surface current

        Note: 
            - The use of 'lax.fori_loop' on time loop greatly reduce exectution speed
            - The use of 'lax.fori_loop' on layer loop has close execution time with not using it, 
                because there is only a few layers
        
        """ 
        #jax.debug.print('pk = {}',pk)
        U = jnp.zeros( (self.nl, self.nt), dtype='complex')

        K = self.K_transform(pk) # optimize inverse problem
        K = self.kt_1D_to_2D(K)
        if K.shape[-1]//2!=self.nl:
           raise Exception('Your model is {} layers, but you want to run it with {} layers (k={})'.format(self.nl, len(pk)//2,pk))

        M = self.pkt2Kt_matrix(gtime=self.forcing_time)
        Kt = jnp.dot(M,K)  
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # dont show overflow results
            # lax.fori_loop
            # arg0 = Kt, U
            # _, U = lax.fori_loop(0, self.nt, self.__step, arg0)
            
            # lax.scan version
            X0 = Kt, U
            #final, _ = lax.scan(self.__step_for_scan, X0, jnp.arange(1,3 )) #self.nt))
            final, _ = lax.scan(self.__step_for_scan, X0, jnp.arange(1,self.nt))
            _, U = final
            
        self.Ur_traj = U
        self.Ua, self.Va = jnp.real(U[0,:]), jnp.imag(U[0,:])  # first layer
        
        return pk, U
    
class classic_slab1D_eqx(eqx.Module):
    """
     Originaly from Pollard (1970), implemented as:
                dU/dt = -ifU + dtau/dz -rU 
            
        1 layer (= "slab model")
                dU/dt = -ifU + Tau_0/H - rU    
            
            parameters = (H,r)
            minimized version:
                dU/dt = -ifU + K0.Tau_0 - K1.U
    """
    # variables
    U0 : np.float64
    V0 : np.float64
    # control vector
    pk : jnp.array
    # parameters
    TA : jnp.array
    fc : jnp.array
    dt_forcing : np.float64
    nl : np.int32
    AD_mode : str = eqx.static_field()
    
    
    @eqx.filter_jit()
    # @partial(jit, static_argnames=['save_traj_at'])
    def __call__(self, call_args, save_traj_at:int):
        t0, t1, dt = call_args
        
        # time saved
        # if type(SaveAt)==list:
        #     if len(SaveAt)>=0:
        #         ns = len(SaveAt)
        # elif type(SaveAt)==np.float64 or type(SaveAt)==np.int32:
        if save_traj_at<=0:
            dt_out = self.dt_forcing
        else:
            dt_out = save_traj_at

        ns = len(self.TA) #jnp.floor_divide((t1-t0),dt_out)
        nsubsteps = jnp.floor_divide(self.dt_forcing,dt) # == self.dt_forcing//dt
        
     
        # initialization
        U = jnp.zeros( (self.nl, ns), dtype='complex') + (self.U0+1j*self.V0)
        
        # optimization transform of control parameters
        K = self.K_transform( self.pk )
        
        # warning on K size
        should_be_size = self.nl + (self.nl-1) +1   # number of Hi, number of tau_i, r
        actual_size = K.shape[-1]
        if actual_size!=should_be_size:
           raise Exception('Your model is {} layers, but you want to run it with {} layers (k={})'.format(self.nl, actual_size//2, self.pk))

        
        # defining loop functions  
        def __Onelayer(arg2):
            """
            1 time iteration of unstek model when 1 layer
            """
            
            it, K, TA, U = arg2
            ik = 0
            U = U.at[ik].set( U[ik] + dt*( 
                                            -1j*self.fc*U[ik] 
                                            + K[ik]*TA 
                                            - K[-1]*U[ik] ) )
            return it, K, TA, U
            
        def __one_step(X0, inner_t):
            """
            1 model time step forward (inner loop)
            
            INPUT:
            - itm: current model time step
            - arg0 it, K, U at current model time step
                it: forcing time index
                K : K vector
                U: velocity 
            OUTPUT:
            - U: updated velocity at next time step 
            """
            it, K, U = X0
    
            # on-the-fly (linear) interpolation of forcing
            itm = it*nsubsteps + inner_t
            itf = jnp.array(it//nsubsteps, int)
            aa = jnp.mod(itm,nsubsteps)/nsubsteps
            itsup = jnp.where(itf+1>=len(self.TA), -1, itf) 
            TA = (1-aa)*self.TA[itf-1] + aa*self.TA[itsup]
            def cond_print(it):
                jax.debug.print('it,itm,itf, {}, {}, {}',it, itm,itf)
            
            cond = jnp.logical_and(it==0,itm<=10)
            jax.lax.cond(cond, cond_print, lambda x:None, it)
            
            arg2 = itm, K, TA, U
            _, _, _, U = __Onelayer(arg2)
            X0 = it, K, U  
                  
            return X0, X0
     
        
        # def __step(it, arg0):
        #     """
        #     outer loop
        #     """
        #     K, U = arg0
        #     Uold = lax.select( it>0, U[:,it-1], U[:,0])
            
        #     X0 = it, K, Uold
        #     final, _ = lax.scan(__one_step, X0, jnp.arange(0, nsubsteps))
        #     _, _, Unext = final
        #     U = U.at[:,it].set(Unext)

        #     return K, U
        def __step(arg0):
            """
            outer loop
            """
            it, K, U = arg0
            Uold = lax.select( it>0, U[:,it-1], U[:,0])
            
            X0 = it, K, Uold
            final, _ = lax.scan(__one_step, X0, jnp.arange(0, nsubsteps))
            _, _, Unext = final
            U = U.at[:,it].set(Unext)
            it = it +1
            return it, K, U
        
        
        # time loop
        # arg0 = K, U
        # _, U = lax.fori_loop(0, ns, __step, arg0)
        
        it=0
        arg0 = it, K, U
        def cond(arg):
            it, _, _ = arg
            return it<ns
                
        _, _, U = lax.while_loop(cond, __step,arg0)        
        return jnp.real(U), jnp.imag(U)
        
    def K_transform(self, pk, order=0, function='exp'):
        """
        Transform the K vector to improve minimization process
        
        INPUT:
            - pk : K vector
            - order : 1 is derivative
            - function : string of the name of the function
        OUTPUT:
            - f(k) or f'(k)
        """  
        if function=='exp':
            return jnp.exp(pk)
        elif function=='id':
            return pk
        else:
            raise Exception('K_transform function '+function+' is not available, retry')
        
class classic_slab1D_eqx_difx(eqx.Module):
    # variables
    # U0 : np.float64
    # V0 : np.float64
    # control vector
    pk : jnp.array
    # parameters
    TAx : jnp.array
    TAy : jnp.array
    fc : jnp.array
    dt_forcing : np.int32
    nl : np.int32
    AD_mode : str
    
    @eqx.filter_jit
    def __call__(self, call_args, save_traj_at = None):
        t0, t1, dt = call_args
        nsubsteps = self.dt_forcing // dt
        def vector_field(t, C, args):
            U,V = C
            fc, K, TAx, TAy = args
            
            # on the fly interpolation
            it = jnp.array(t//dt, int)
            itf = jnp.array(it//nsubsteps, int)
            
            aa = jnp.mod(it,nsubsteps)/nsubsteps
            itsup = lax.select(itf+1>=len(TAx), -1, itf+1) 
            TAx = (1-aa)*TAx[itf] + aa*TAx[itsup]
            TAy = (1-aa)*TAy[itf] + aa*TAy[itsup]
            # def cond_print(it):
            #     jax.debug.print('it,itf, TA, {}, {}, {}',it,itf,(TAx,TAy))
            
            # jax.lax.cond(it<=10, cond_print, lambda x:None, it)
            
            # physic
            d_U = fc*V + K[0]*TAx - K[1]*U
            d_V = -fc*U + K[0]*TAy - K[1]*V
            d_y = d_U,d_V
            return d_y
        
        term = ODETerm(vector_field)
        
        solver = Euler()
        # Auto-diff mode
        if self.AD_mode=='F':
            adjoint = diffrax.ForwardMode()
        else:
            adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=10)

        y0 = 0.0,0.0 # self.U0,self.V0
        # control
        K = jnp.exp( jnp.asarray(self.pk) )
  
        args = self.fc, K, self.TAx, self.TAy
        
        if save_traj_at is None:
            saveat = diffrax.SaveAt(steps=True)
        else:
            saveat = diffrax.SaveAt(ts=jnp.arange(t0,t1,save_traj_at)) # slower than above (no idea why)
            #saveat = diffrax.SaveAt(ts=save_traj_at)
        
        maxstep = int((t1-t0)//dt) +1 
        
        return diffeqsolve(term, 
                           solver, 
                           t0=t0, 
                           t1=t1, 
                           y0=y0, 
                           args=args, 
                           dt0=dt, #dt, None
                           saveat=saveat,
                           #stepsize_controller=diffrax.StepTo(jnp.arange(t0, t1+dt, dt)),
                           adjoint=adjoint,
                           max_steps=maxstep,
                           made_jump=False) # here this is needed to be able to forward AD
    