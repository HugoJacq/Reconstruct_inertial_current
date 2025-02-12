import numpy as np
import xarray as xr
import warnings
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit,grad, lax, jvp, vjp
from functools import partial


class jUnstek1D:
    """
    JAX version, Unsteady Ekman Model 1D, with N layers 
    
    See : https://doi.org/10.3390/rs15184526
    """
    def __init__(self, Nl, forcing, observations):
        # from dataset
        self.nt = len(forcing.time)
        self.dt = forcing.time[1] - forcing.time[0]
        # forcing
        self.TA = jnp.asarray(forcing.TAx) + 1j*jnp.asarray(forcing.TAy) # wind = windU + j*windV
        self.fc = jnp.asarray(forcing.fc)  
        # obs
        Co = observations.get_obs()
        self.Uo, self.Vo = Co[0], Co[1]
        self.Ri = self.Uo*0.+1
        # for reconstructed 
        self.nl = Nl
        self.isJax = True

        # JIT compiled functions
        self.do_forward_jit = jit(self.do_forward)
        self.K_transform_jit = jit(self.K_transform)
        self.__one_step_jit = jit(self.__one_step)
        
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
            return jnp.exp(pk) # .astype('complex')
        else:
            raise Exception('K_transform function '+function+' is not available, retry')

    def __Onelayer(self, arg1):
        """
        1 time iteration of unstek model when 1 layer
        """
        it, K, U = arg1
        ik = 0
        U = U.at[ik,it+1].set( U[ik, it] + self.dt*( -1j*self.fc*U[ik, it] +K[2*ik]*self.TA[it] - K[2*ik+1]*(U[ik, it]) ) )
        return it, K, U
        
    def __Nlayer_midlayers_for_scan(self,X0,ik):
        it, K, U = X0
        U = U.at[ik,it+1].set( U[ik,it] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik, it] 
                                    - K[2*ik]*(U[ik, it]-U[ik-1, it]) 
                                    - K[2*ik+1]*(U[ik, it]-U[ik+1, it]) ) )
        X = it, K, U
        return X, X
        
    def __Nlayer(self,arg1):
        """
        1 time iteration of unstek model when N layer
        """        
        
        it, K, U = arg1
        # _, _, U = lax.fori_loop(0,self.nl,self.__do_layer_k,arg1)
        
        # surface
        ik = 0
        U = U.at[ik, it+1].set( U[ik, it] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik, it] 
                                    + K[2*ik]*self.TA[it] 
                                    - K[2*ik+1]*(U[ik, it]-U[ik+1, it]) ) )
        # bottom
        ik = -1
        U = U.at[ik,it+1].set(  U[ik, it] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik, it] 
                                    - K[2*ik]*(U[ik, it]-U[ik-1, it]) 
                                    - K[2*ik+1]*U[ik, it] ) )
        # in between
        X0 = it, K, U
        final, result = lax.scan(self.__Nlayer_midlayers_for_scan, X0, jnp.arange(1,self.nl-1) )
        it, K, U = final
        return it, K, U    
    
    def __one_step(self, it, arg0):
        """
        1 time step advance

        
        INPUT:
        - pk : K vector
        - it: current time step
        - U: velocity at current time step 
        OUTPUT:
        - U: updated velocity at next time step 
        """
        pk ,U = arg0
        K = self.K_transform(pk)

        arg1 = it, K, U
        # loop on layers
        _, _, U = lax.cond( self.nl == 1,       # condition, only 1 layer ?
                            self.__Onelayer,    # if 1 layer, ik=0
                            self.__Nlayer,      # else, loop on layers
                            arg1)   
           
        # faire un ouput avec 1 valeur tous les jours.   
            
        # old, work but slow (same as no jax version)
        # performance is comparable to not using __Nlayer because there is a few layers
        # forward model is on par with no jax (~0.5s) but gradient is much slower (~5s)
        # for ik in range(self.nl):
        #     if ((ik==0)&(ik==self.nl-1)): 
        #         U = U.at[ik,it+1].set( U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] +K[2*ik]*self.TA[it] - K[2*ik+1]*(U[ik][it]) ) )
        #     else:
        #         if ik==0: U = U.at[ik, it+1].set( U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] +K[2*ik]*self.TA[it] - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) ) )
        #         elif ik==self.nl-1:  U = U.at[ik,it+1].set( U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*U[ik][it] ) )
        #         else: U = U.at[ik,it+1].set( U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) ) )
        
        return pk, U


    def do_forward(self, pk):
        """
        Unsteady Ekman model forward model

        for 1 layer: dU/dt = -j*fc*U + K0*Tau - K1*U

        INPUT:
        - pk     : list of boundaries of layer(s)
        - U0    : initial value of complex current
        OUTPUT:
        - array of surface current

        Note: 
            - The use of 'lax.fori_loop' on time loop greatly reduce exectution speed
            - The use of 'lax.fori_loop' on layer loop has close execution time with not using it, 
                because there is only a few layers
        """ 
        #jax.debug.print('pk = {}',pk)
        if len(pk)//2!=self.nl:
            raise Exception('Your model is {} layers, but you want to run it with {} layers (k={})'.format(self.nl, len(pk)//2,pk))

        # starting from nul current
        U = jnp.zeros( (self.nl,self.nt), dtype='complex')

        arg0 = pk, U
        X0 = pk, U
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # dont show overflow results
            _, U = lax.fori_loop(0,self.nt,self.__one_step_jit,arg0)
            
        self.Ur_traj = U
        self.Ua, self.Va = jnp.real(U[0,:]), jnp.imag(U[0,:])  # first layer

        return pk, U
    
    
    def tgl(self, k0):
        U0 = jnp.zeros((self.nl,self.nt), dtype='complex')
        dU0 = jnp.zeros((self.nl,self.nt), dtype='complex')
        _, dU = jvp(partial(self.do_forward_jit, pk=k0), (U0,), (dU0,) )
        return dU

    def adjoint(self, pk0, d):
        """
        Computes the adjoint of the vector K for 'unstek'

        INPUT:
            - pk0 : K vector
            - d  : innovation vector, observation forcing for [zonal,meridional] currents
        OUTPUT:
            - returns: - adjoint of vectors K
            
        Note: this function works with jax arrays
        # this function is old and should not be used
        """
        
        ad_U0 = jnp.zeros((self.nl,self.nt), dtype='complex')            
        ad_U0 = ad_U0.at[:,:].add( d[0] + 1j*d[1] )
        
        ad_K0 = jnp.zeros(len(pk0)).astype('complex')
        #jax.debug.print("jax.debug.print(ad_K0) -> {x}", x=ad_K0)
        U0 = jnp.zeros((self.nl,self.nt), dtype='complex')
        
                   
        _, adf = vjp( self.do_forward, pk0, U0)
        ad_K = adf( (ad_K0, ad_U0) )[0]
        return jnp.real(ad_K)

class jUnstek1D_Kt:
    """
    Unsteady Ekman Model 1D, with N layers 
    Vector K change with slowly with time.
    
    Written in JAX formalism.
    """
    def __init__(self, Nl, forcing, dT):
        """
        Nl : number of layers
        forcing : class Forcing1D
        observations : class Observation1D
        dT : seconds, period of change of K in time
        """
        # from dataset
        self.nt = len(forcing.time)
        self.dt = forcing.time[1] - forcing.time[0]
        self.forcing_time = forcing.time
        self.model_time = jnp.arange(0,self.nt*self.dt,self.dt)
        # forcing
        self.TA = jnp.asarray(forcing.TAx) + 1j*jnp.asarray(forcing.TAy) # wind = windU + j*windV
        self.fc = jnp.asarray(forcing.fc)  
        # obs
        # Co = observations.get_obs()
        # self.Uo, self.Vo = Co[0], Co[1]
        # self.Ri = self.Uo*0.+1
        # for reconstructed 
        self.nl = Nl
        self.isJax = True
        # time varying K
        self.dT = dT
        self.NdT = forcing.time[-1]//dT + 1

        # JIT compiled functions
        self.do_forward_jit = jit(self.do_forward)
        self.K_transform_jit = jit(self.K_transform)
        self.__one_step_jit = jit(self.__one_step)
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
            return jnp.exp(pk) # .astype('complex')
        else:
            raise Exception('K_transform function '+function+' is not available, retry')

    def K_transform_reverse(self, K, function='exp'):
        """
        function is the forward function
        """
        if function=='exp':
            return jnp.log(K) # .astype('complex')
        else:
            raise Exception('K_transform_reverse function '+function+' is not available, retry')

    def kt_ini(self,pk):
        return jnp.array( [pk]*self.NdT)

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
        
        step = (self.dT//self.dt).astype(int)
        Nstep = self.nt//step
        gptime = jnp.zeros( self.nt//step+1 )        
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

    def __Onelayer(self, arg1):
        """
        1 time iteration of unstek model when 1 layer
        """
        it, K, U = arg1
        ik = 0
        U = U.at[ik,it+1].set( U[ik, it] + self.dt*( -1j*self.fc*U[ik, it] +K[it,2*ik]*self.TA[it] - K[it,2*ik+1]*(U[ik, it]) ) )
        return it, K, U
        
    def __Nlayer_midlayers_for_scan(self,X0,ik):
        it, K, U = X0
        U = U.at[ik,it+1].set( U[ik,it] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik, it] 
                                    - K[it,2*ik]*(U[ik, it]-U[ik-1, it]) 
                                    - K[it,2*ik+1]*(U[ik, it]-U[ik+1, it]) ) )
        X = it, K, U
        return X, X
        
    def __Nlayer(self,arg1):
        """
        1 time iteration of unstek model when N layer
        """        
        
        it, K, U = arg1
        # _, _, U = lax.fori_loop(0,self.nl,self.__do_layer_k,arg1)
        
        # surface
        ik = 0
        U = U.at[ik, it+1].set( U[ik, it] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik, it] 
                                    + K[it,2*ik]*self.TA[it] 
                                    - K[it,2*ik+1]*(U[ik, it]-U[ik+1, it]) ) )
        # bottom
        ik = -1
        U = U.at[ik,it+1].set(  U[ik, it] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik, it] 
                                    - K[it,2*ik]*(U[ik, it]-U[ik-1, it]) 
                                    - K[it,2*ik+1]*U[ik, it] ) )
        # in between
        X0 = it, K, U
        final, result = lax.scan(self.__Nlayer_midlayers_for_scan, X0, jnp.arange(1,self.nl-1) )
        it, K, U = final
        return it, K, U    
    
    def __one_step(self, it, arg0):
        """
        1 time step advance

        
        INPUT:
        - pk : K vector
        - it: current time step
        - U: velocity at current time step 
        OUTPUT:
        - U: updated velocity at next time step 
        """
        K, U = arg0
        

        arg1 = it, K, U
        # loop on layers
        _, _, U = lax.cond( self.nl == 1,       # condition, only 1 layer ?
                            self.__Onelayer,    # if 1 layer, ik=0
                            self.__Nlayer,      # else, loop on layers
                            arg1)   
           
        # faire un ouput avec 1 valeur tous les jours.   
            
        # old, work but slow (same as no jax version)
        # performance is comparable to not using __Nlayer because there is a few layers
        # forward model is on par with no jax (~0.5s) but gradient is much slower (~5s)
        # for ik in range(self.nl):
        #     if ((ik==0)&(ik==self.nl-1)): 
        #         U = U.at[ik,it+1].set( U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] +K[2*ik]*self.TA[it] - K[2*ik+1]*(U[ik][it]) ) )
        #     else:
        #         if ik==0: U = U.at[ik, it+1].set( U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] +K[2*ik]*self.TA[it] - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) ) )
        #         elif ik==self.nl-1:  U = U.at[ik,it+1].set( U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*U[ik][it] ) )
        #         else: U = U.at[ik,it+1].set( U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) ) )
        
        return K, U

    def do_forward(self, pk):
        """
        Unsteady Ekman model forward model

        for 1 layer: dU/dt = -j*fc*U + K0*Tau - K1*U

        INPUT:
        - pk     : list of boundaries of layer(s)
        - U0    : initial value of complex current
        OUTPUT:
        - array of surface current

        Note: 
            - The use of 'lax.fori_loop' on time loop greatly reduce exectution speed
            - The use of 'lax.fori_loop' on layer loop has close execution time with not using it, 
                because there is only a few layers
        """ 
        #jax.debug.print('pk = {}',pk)

        # starting from nul current
        U = jnp.zeros( (self.nl,self.nt), dtype='complex')
        K = self.K_transform(pk) # optimize inverse problem
        K = K.reshape((self.NdT,self.nl*2))
        if K.shape[-1]//2!=self.nl:
           raise Exception('Your model is {} layers, but you want to run it with {} layers (k={})'.format(self.nl, len(pk)//2,pk))
        gtime = jnp.asarray(self.forcing_time)
        M = self.pkt2Kt_matrix(gtime=gtime)
        Kt = jnp.dot(M,K)  
        arg0 = Kt, U
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # dont show overflow results
            _, U = lax.fori_loop(0,self.nt,self.__one_step_jit,arg0)
            
        self.Ur_traj = U
        self.Ua, self.Va = jnp.real(U[0,:]), jnp.imag(U[0,:])  # first layer
        
        return pk, U

class jUnstek1D_Kt_spatial:
    """
    Unsteady Ekman Model 1D, with N layers 
    Vector K changes slowly with time.
    This is still 1D because the K is the same at all spatial location, it only varies with time.
    
    Written in JAX formalism.
    """
    def __init__(self, dt, Nl, forcing, observation, dT):
        """
        dt : model time step
        Nl : number of layers
        forcing : class Forcing1D (on a different timestep)
        dT : seconds, period of change of K in time
        """
        # from dataset
        self.nt = len(forcing.time)
        self.nx = forcing.data.sizes['x_rho']
        self.ny = forcing.data.sizes['y_rho']
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
        self.__one_step_jit = jit(self.__one_step)
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
        else:
            raise Exception('K_transform function '+function+' is not available, retry')
    def K_transform_reverse(self, K, function='exp'):
        """
        function is the forward function
        """
        if function=='exp':
            return jnp.log(K)
        else:
            raise Exception('K_transform_reverse function '+function+' is not available, retry')
    def kt_ini(self,pk):
        return jnp.array( [pk]*self.NdT)
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
        Nstep = self.ntm//step
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
        #U = U.at[ik,it+1].set( U[ik, it] + self.dt*( -1j*self.fc*U[ik, it] +K[it,2*ik]*TA - K[it,2*ik+1]*(U[ik, it]) ) )
        U = U.at[ik,:,:].set( U[ik,:,:] + self.dt*( -1j*self.fc*U[ik,:,:] +K[it,2*ik]*TA - K[it,2*ik+1]*(U[ik,:,:]) ) )
        return it, K, TA, U
        
    def __Nlayer_midlayers_for_scan(self,X0,ik):
        it, K, U = X0
        # U = U.at[ik,it+1].set( U[ik][it] + 
        #                         self.dt*( 
        #                             - 1j*self.fc*U[ik, it] 
        #                             - K[it,2*ik]*(U[ik, it]-U[ik-1, it]) 
        #                             - K[it,2*ik+1]*(U[ik, it]-U[ik+1, it]) ) )
        U = U.at[ik,:,:].set( U[ik,:,:] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik,:,:] 
                                    - K[it,2*ik]*(U[ik,:,:]-U[ik-1,:,:]) 
                                    - K[it,2*ik+1]*(U[ik,:,:]-U[ik+1,:,:]) ) )
        X = it, K, U
        return X, X
        
    def __Nlayer(self,arg2):
        """
        1 time iteration of unstek model when N layer
        """        
        
        it, K, TA,  U = arg2
        # _, _, U = lax.fori_loop(0,self.nl,self.__do_layer_k,arg1)
        
        # surface
        ik = 0
        # U = U.at[ik, it+1].set( U[ik, it] + 
        #                         self.dt*( 
        #                             - 1j*self.fc*U[ik, it] 
        #                             + K[it,2*ik]*TA 
        #                             - K[it,2*ik+1]*(U[ik, it]-U[ik+1, it]) ) )
        U = U.at[ik,:,:].set( U[ik,:,:] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik,:,:] 
                                    + K[it,2*ik]*TA 
                                    - K[it,2*ik+1]*(U[ik,:,:]-U[ik+1,:,:]) ) )
        # bottom
        ik = -1
        # U = U.at[ik,it+1].set(  U[ik, it] + 
        #                         self.dt*( 
        #                             - 1j*self.fc*U[ik, it] 
        #                             - K[it,2*ik]*(U[ik, it]-U[ik-1, it]) 
        #                             - K[it,2*ik+1]*U[ik, it] ) )
        U = U.at[ik,:,:].set(  U[ik,:,:] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik,:,:] 
                                    - K[it,2*ik]*(U[ik,:,:]-U[ik-1,:,:]) 
                                    - K[it,2*ik+1]*U[ik,:,:] ) )
        # in between
        X0 = it, K, U
        final, _ = lax.scan(self.__Nlayer_midlayers_for_scan, X0, jnp.arange(1,self.nl-1) )
        _, _, U = final
        return it, K, TA, U    
    
    #@partial(jax.checkpoint,
    #     policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
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
        #jax.debug.print('   inner_t = {}',inner_t)
        itm = it*nsubsteps + inner_t
        #jax.debug.print('   itm = {}',itm)
        
        # on-the-fly (linear) interpolation of forcing
        aa = jnp.mod(itm,nsubsteps)/nsubsteps
        #jax.debug.print('it = {}, itm = {}, aa = {}',it,itm,aa)
        #print('it,itm,aa = {}, {}, {}',it,itm,aa)
        #it = lax.select(aa==0,it+1,it)                      # if aa==0: it += 1
        itsup = lax.select(it+1>=self.nt, -1, it+1)         # if itsup>=self.nt: itsup=-1, else: it+1
        TA = (1-aa)*self.TA[it,:,:] + aa*self.TA[itsup,:,:] # TA is stress at current model time step
        
        arg2 = itm, Kt, TA, U
        # loop on layers
        _, _, _, U = lax.cond( self.nl == 1,       # condition, only 1 layer ?
                            self.__Onelayer,    # if 1 layer, ik=0
                            self.__Nlayer,      # else, loop on layers
                            arg2)   
        X0 = it, Kt, U        
        return X0, X0
    
    # https://docs.jax.dev/en/latest/notebooks/autodiff_remat.html#fundamentals-of-jax-checkpoint
    
    
    def __step(self, it, arg0):
        """
        outer loop (forcing time step, 1 hour)
        """
        K, U = arg0
        #jax.debug.print('it = {}',it)
        # start = it
        # end = lax.select((it+1)>=self.nt, self.nt, it+1)
        Uold = lax.select( it>0, U[:,it-1,:,:], U[:,0,:,:])
        # arg1 = it, K, Uold
        #_, _, Unext = lax.fori_loop(start, end, self.__one_step,arg1)
        
        X0 = it, K, Uold
        final, _ = lax.scan(self.__one_step, X0, jnp.arange(0, self.dt_forcing//self.dt))
        
        _, _, Unext = final
        U = U.at[:,it,:,:].set(Unext)

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
        
        TBD : add option to save at every dt
        """ 
        #jax.debug.print('pk = {}',pk)

        # starting from nul current
        #U = jnp.zeros( (self.nl, self.ntm, self.ny, self.nx), dtype='complex')
        # ici pas besoin de save tous les pas de temps pour la fonction cout !
        # (pour obtenir une trajectoire si, quand le K est final)
        # save every hour (model OSSE)
        U = jnp.zeros( (self.nl, self.nt, self.ny, self.nx), dtype='complex')
        
        print(U.shape)
        print(U.dtype)
        
        # i need to fill in U only at
        #   dt if save_all_dt=True
        #   obs_dt if save_all_dt=False
        
        # -> voir la fonction step (obs_dt) et one_step (dt) dans MSHH de Florian
        # https://github.com/leguillf/MASSH/blob/main/mapping/src/tools_4Dvar.py
        # autre possibilité : 
        # https://docs.kidger.site/diffrax/
        self.Kt = self.kt_2D_to_1D(self.kt_ini(pk))
        
        
        K = self.K_transform(self.Kt) # optimize inverse problem
        K = K.reshape((self.NdT,self.nl*2))
        if K.shape[-1]//2!=self.nl:
           raise Exception('Your model is {} layers, but you want to run it with {} layers (k={})'.format(self.nl, len(pk)//2,pk))

        #gtime = jnp.asarray(self.forcing_time)
        M = self.pkt2Kt_matrix(gtime=self.model_time)
        Kt = jnp.dot(M,K)  
        
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # dont show overflow results
            # lax.fori_loop
            # arg0 = Kt, U
            # _, U = lax.fori_loop(0, self.nt, self.__step, arg0)
            
            # lax.scan version
            X0 = Kt, U
            #final, _ = lax.scan(self.__step_for_scan, X0, jnp.arange(1,3 )) #self.nt))
            final, _ = lax.scan(self.__step_for_scan, X0, jnp.arange(0,self.nt))
            _, U = final
            
        self.Ur_traj = U
        self.Ua, self.Va = jnp.real(U[0,:]), jnp.imag(U[0,:])  # first layer
        
        return pk, U