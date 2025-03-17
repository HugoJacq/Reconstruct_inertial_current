
"""
This modules gather models using unsteady ekman equation to reconstruct inertial current from wind stress
Each model needs a forcing, from the module 'forcing.py'.

The models are written in numpy and JAX to allow for a direct comparison.

refs:
Wang et al. 2023: https://www.mdpi.com/2072-4292/15/18/4526

By: Hugo Jacquet march 2025
"""
import warnings
import numpy as np

import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax

class Unstek1D:
    """
    Unsteady Ekman Model 1D, with N layers 
    
    See : https://doi.org/10.3390/rs15184526
    """
    
    def __init__(self, dt, Nl, forcing, observations):
        # from dataset
        self.dt = dt
        self.dt_forcing = forcing.dt_forcing
        self.nt = int(len(forcing.time) * self.dt_forcing / self.dt)
        self.time = np.arange(0,self.nt,dt)
        # forcing
        self.forcing = forcing
        self.TA = forcing.TAx + 1j*forcing.TAy # wind = windU + j*windV
        self.fc = forcing.fc  
        # observations
        self.Uo, self.Vo = observations.get_obs()
        self    .Ri = self.Uo*0.+1
        # for reconstructed 
        self.nl = Nl
        self.isJax = False
      
      
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
            return np.exp(pk)
        else:
            raise Exception('K_transform function '+function+' is not available, retry')
            
                      
    def step(self, it, arg0):
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
        
        # on-the-fly (linear) interpolation of forcing
        nsubsteps = self.dt_forcing // self.dt
        aa = np.mod(it,nsubsteps)/nsubsteps
        itf = it // (self.dt_forcing//self.dt)
        if itf+1>=self.forcing.nt:
            itsup = -1
        else:
            itsup = itf+1
            
        #print("it, itf",it,itf)
        TA = (1-aa)*self.TA[itf] + aa*self.TA[itsup]
        
        if it==60:
            print("it, itf",it,itf)
            print("TA",TA)
            #print("fc",self.fc)
            print("aa",aa)
            
        for ik in range(self.nl):
            if ((ik==0)&(ik==self.nl-1)): 
                U[ik,it+1] = U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] +K[2*ik]*TA - K[2*ik+1]*(U[ik][it]) )
            else:
                if ik==0: U[ik, it+1] = U[ik, it] + self.dt*( -1j*self.fc*U[ik, it] +K[2*ik]*TA - K[2*ik+1]*(U[ik, it]-U[ik+1, it]) ) 
                elif ik==self.nl-1:  U[ik,it+1] = U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*U[ik][it] ) 
                else: U[ik,it+1] = U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) ) 
        return pk, U
        
    def do_forward(self, pk):
        """
        Unsteady Ekman model forward model
        
        for 1 layer: dU/dt = -j*fc*U + K0*Tau - K1*U

        INPUT:
            - pk     : list of boundaries of layer(s)
            - return_traj : if True, return current as complex number
        OUTPUT:
            - array of surface current
            
        Note: this function works with numpy arrays
        """ 
        U = np.zeros((self.nl,self.nt), dtype='complex')
        if len(pk)//2!=self.nl:
            raise Exception('Your model is {} layers, but you want to run it with {} layers (k={})'.format(self.nl, len(pk)//2,pk))
        
        arg0 = pk, U
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # dont show overflow results
            for it in range(0,self.nt-1):
                _, U = self.step(it, arg0)
                
        self.Ur_traj = U
        self.Ua, self.Va = np.real(U[0,:]), np.imag(U[0,:])  # first layer        
        return pk, U
         
    def step_tgl(self, K, dK, it, U, dU):
        """
        1 time step of tgl
        """
        for ik in range(self.nl):
            if ((ik==0)&(ik==self.nl-1)): 
                U[ik][it+1] = U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] +K[2*ik]*self.TA[it] - K[2*ik+1]*(U[ik][it]) )
                dU[ik][it+1] = dU[ik][it] + self.dt*( -1j*self.fc*dU[ik][it] +dK[2*ik]*self.TA[it] - dK[2*ik+1]*(U[ik][it]) - K[2*ik+1]*(dU[ik][it]) )
            else:
                if ik==0: 
                    U[ik][it+1] = U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] +K[2*ik]*self.TA[it] - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) )
                    dU[ik][it+1] = dU[ik][it] + self.dt*( -1j*self.fc*dU[ik][it] +dK[2*ik]*self.TA[it] - dK[2*ik+1]*(U[ik][it]-U[ik+1][it]) - K[2*ik+1]*(dU[ik][it]-dU[ik+1][it]))
                elif ik==self.nl-1: 
                    U[ik][it+1] = U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*U[ik][it] )
                    dU[ik][it+1] = dU[ik][it] + self.dt*( -1j*self.fc*dU[ik][it] -dK[2*ik]*(U[ik][it]-U[ik-1][it]) -K[2*ik]*(dU[ik][it]-dU[ik-1][it])    - dK[2*ik+1]*U[ik][it] - K[2*ik+1]*dU[ik][it])
                else: 
                    U[ik][it+1] = U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) )
                    dU[ik][it+1] = dU[ik][it] + self.dt*( -1j*self.fc*dU[ik][it] -dK[2*ik]*(U[ik][it]-U[ik-1][it]) -K[2*ik]*(dU[ik][it]-dU[ik-1][it]) - dK[2*ik+1]*(U[ik][it]-U[ik+1][it]) - K[2*ik+1]*(dU[ik][it]-dU[ik+1][it]) )

        return  dU
        
    def tgl(self, k, dk):
        """
        Computes the Tangent Linear model for 'unstek'.
        This is a linear approximation of the 'unstek' model derived with control theory.

        INPUT:
            - time  : array of time
            - k     : list of boundaries of layer(s)
            - dk    : perturbation of K
        OUTPUT:
            - First layer values of the TGL for zonal/meridional current, along the time dimension
            
        Note: this function works with numpy arrays
        """    
        K = self.K_transform(k)
        dK = self.K_transform(k, order=1)*dk
        time = self.time
        nl = int(len(K)//2)
        U = [None]*nl
        dU = [None]*nl
        for ik in range(nl):
            U[ik]=np.zeros((len(time)), dtype='complex')
            dU[ik]=np.zeros((len(time)), dtype='complex')

        for it in range(len(time)-1):
            dU = self.step_tgl(K, dK, it, U, dU)

        return dU # np.real(dU[0]), np.imag(dU[0])   

        
    def step_adj(self, K, it, ad_U, ad_K, U):
        """
        1 time step advance for adjoint
        
        INPUT:
            - pk  : K vector
            - it: current time step
            - ad_U: U adjoint at current step
            - ad_K: K adjoint at current step
            - U    : current at the current step (after a forward pass)
        OUTPUT:
            - ad_U: U adjoint at next step
            - ad_K: K adjoint at next step
        """
        
        # on-the-fly (linear) interpolation of forcing
        nsubsteps = self.dt_forcing // self.dt
        aa = np.mod(it,nsubsteps)/nsubsteps
        itf = it // (self.dt_forcing//self.dt)
        if itf+1>=self.forcing.nt:
            itsup = -1
        else:
            itsup = itf+1
        TA = (1-aa)*self.TA[itf] + aa*self.TA[itsup]
        
        for ik in range(self.nl):
            if ((ik==0)&(ik==self.nl-1)): 
                ad_U[ik,it] += ad_U[ik,it+1]*np.conjugate(1+self.dt*( -1j*self.fc- K[2*ik+1]))
                ad_K[2*ik] += self.dt*(ad_U[ik,it+1]*np.conjugate(TA))
                ad_K[2*ik+1] += self.dt*(-ad_U[ik,it+1]*np.conjugate(U[ik,it]))
                
            else:
                if ik==0: 
                    ad_U[ik,it] += ad_U[ik,it+1]*np.conjugate(1+self.dt*( -1j*self.fc- K[2*ik+1])) 
                    ad_U[ik+1,it] += ad_U[ik,it+1]*np.conjugate(self.dt*K[2*ik+1])
                    ad_K[2*ik] += self.dt*(ad_U[ik,it+1]*np.conjugate(TA)) 
                    ad_K[2*ik+1] += self.dt*(-ad_U[ik,it+1]*np.conjugate(U[ik,it]-U[ik+1,it]))

                elif ik==self.nl-1: 
                    ad_U[ik,it] += ad_U[ik,it+1]*np.conjugate(1+self.dt*( -1j*self.fc- K[2*ik]- K[2*ik+1]))
                    ad_U[ik-1,it] += ad_U[ik,it+1]*np.conjugate(self.dt*K[2*ik])
                    ad_K[2*ik] += ad_U[ik,it+1]*(-self.dt*np.conjugate(U[ik,it]-U[ik-1,it]))
                    ad_K[2*ik+1] += ad_U[ik,it+1]*self.dt*np.conjugate(-U[ik,it])

                else: 
                    ad_U[ik,it] += ad_U[ik,it+1]*np.conjugate(1+self.dt*( -1j*self.fc- K[2*ik]- K[2*ik+1]))
                    ad_U[ik+1,it] += ad_U[ik,it+1]*np.conjugate(self.dt*K[2*ik+1])
                    ad_U[ik-1,it] += ad_U[ik,it+1]*np.conjugate(self.dt*K[2*ik])
                    ad_K[2*ik] += ad_U[ik,it+1]*np.conjugate(-self.dt*(U[ik,it]-U[ik-1,it]))
                    ad_K[2*ik+1] += self.dt*(-ad_U[ik,it+1]*np.conjugate(U[ik,it]-U[ik+1,it]))
        return ad_K
                   
    def adjoint(self, pk, d):
        """
        Computes the adjoint of the vector K for 'unstek'

        INPUT:
            - pk : K vector
            - d  : innovation vector, observation forcing for [zonal,meridional] currents
        OUTPUT:
            - returns: - adjoint of vectors K
            
        Note: this function works with numpy arrays
        """
        _, U = self.do_forward(pk)
            
        ad_U = np.zeros((self.nl,self.nt), dtype='complex')
        ad_U[0,:] = d[0] + 1j*d[1] 

        K = self.K_transform(pk)
        ad_K = np.zeros(len(pk), dtype='complex')

        for it in np.arange(self.nt - 1)[::-1]:
            ad_K = self.step_adj(K, it, ad_U, ad_K, U)

        ad_k = self.K_transform(pk, order=1, function='exp')*ad_K # watch out if K_transform is different from exp !

        return np.real(ad_k)



class jUnstek1D:
    """
    JAX version, Unsteady Ekman Model 1D, with N layers 
       
                dU/dt = -ifU + dtau/dz  with tau(z) = K(z)dU/dz
    
    +> 1 layer:
                dU/dt = -ifU + (tau_0 - Kz.U1/H)/H
        
        minimized form is:
                dU/dt = -ifU + K0.tau_0 - K1.U1
        
        
    -> 2 layers:
        i=1     dU1/dt = -i.f.U1 + (tau_0 - Kz.(U1-U2)/(0.5(H1+H2))/H1
        i=2     dU2/dt = -i.f.U2 + (Kz.(U2-U1)/(0.5(H1+H2) - Kz.(U2)/H2)/H2

        minimized form is:
                dU1/dt = -i.f.U1 + K0.tau_0 - K1.(U1-U2)
                dU2/dt = -i.f.U2 + K2.(U2-U1) - K3.U2        
    """
    def __init__(self, dt, Nl, forcing):
        """
        dt : model time step
        Nl : number of layers
        forcing : class Forcing1D (on a different timestep)
        """
        # from dataset
        self.nt = len(forcing.time)
        self.dt_forcing = forcing.dt_forcing
        self.dt = int(dt)
        self.ntm = int(len(forcing.time) * self.dt_forcing / self.dt) # forcing.time[-1]//dt
        self.forcing_time = forcing.time
        self.model_time = jnp.arange(0,self.ntm*self.dt,self.dt)
        # forcing
        self.TA = jnp.asarray(forcing.TAx) + 1j*jnp.asarray(forcing.TAy) # wind = windU + j*windV
        self.fc = jnp.asarray(forcing.fc)  
        # for reconstructed 
        self.nl = Nl
        self.isJax = True     
        
        # JIT compiled functions
        self.do_forward_jit = jit(self.do_forward)
        #self.do_forward_jit = self.do_forward
    
    
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

    def __Onelayer(self, arg2):
        """
        1 time iteration of unstek model when 1 layer
        """
        K, TA, U = arg2
        ik = 0
        U = U.at[ik].set( U[ik] + self.dt*( -1j*self.fc*U[ik] +K[2*ik]*TA - K[2*ik+1]*(U[ik]) ) )
        return K, TA, U
        
    def __Nlayer_midlayers_for_scan(self,X0,ik):
        K, U = X0
        U = U.at[ik].set( U[ik] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik] 
                                    - K[2*ik]*(U[ik]-U[ik-1]) 
                                    - K[2*ik+1]*(U[ik]-U[ik+1]) ) )
        X = K, U
        return X, X
        
    def __Nlayer(self,arg2):
        """
        1 time iteration of unstek model when N layer
        """        
        
        K, TA,  U = arg2
        
        # surface
        ik = 0
        U = U.at[ik].set( U[ik] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik] 
                                    + K[2*ik]*TA 
                                    - K[2*ik+1]*(U[ik]-U[ik+1]) ) )
        # bottom
        ik = -1
        U = U.at[ik].set(  U[ik] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik] 
                                    - K[2*ik]*(U[ik]-U[ik-1]) 
                                    - K[2*ik+1]*U[ik] ) )
        # in between
        X0 = K, U
        final, _ = lax.scan(self.__Nlayer_midlayers_for_scan, X0, jnp.arange(1,self.nl-1) )
        _, U = final
        return K, TA, U    
    
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
        itm = it*nsubsteps + inner_t # + 1 # here the +1 is due to Python starting at 0
        
        # on-the-fly (linear) interpolation of forcing
        aa = jnp.mod(itm,nsubsteps)/nsubsteps
        itsup = lax.select(it+1>=self.nt, -1, it+1) 
        TA = (1-aa)*self.TA[it] + aa*self.TA[itsup]
        
        
        def print_jax():
            jax.debug.print('itm = {}',itm)
            jax.debug.print('itf = {}',it)
            jax.debug.print('TA = {}',TA)
            #jax.debug.print('fc = {}',self.fc)
            jax.debug.print('aa = {}',aa)
            return None
        def print_none():
            return None
        lax.cond(itm==60,print_jax,print_none)

            
        arg2 = K, TA, U
        # loop on layers
        _, _, U = lax.cond( self.nl == 1,    # condition, only 1 layer ?
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
        Uold = U[:,it]
        X0 = it, K, Uold
        final, _ = lax.scan(self.__one_step, X0, jnp.arange(0, self.dt_forcing//self.dt))
        _, _, Unext = final
        U = U.at[:,it+1].set(Unext)

        return K, U

    def __step2(self,itm,arg0):
        K, U = arg0
        Uold = U[:,itm]
        it = itm // (self.dt_forcing//self.dt)
        X0 = it, K, Uold
        final, _ = self.__one_step(X0, itm)
        _, _, Unext = final
        U = U.at[:,itm+1].set(Unext)
        return K, U
    
    def __step_for_scan(self, X0, it):
        """
        outer loop (forcing time step, 1 hour)
        """
        arg0 = X0
        X0 = self.__step(it, arg0)
        return X0, X0

    def __step2_for_scan(self, X0, it):
        """
        outer loop (forcing time step, 1 hour)
        """
        arg0 = X0
        X0 = self.__step2(it, arg0)
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
        U = jnp.zeros( (self.nl, self.nt), dtype='complex')
        #U = jnp.zeros( (self.nl, self.ntm), dtype='complex')
        
        K = self.K_transform(pk) # optimize inverse problem
        if K.shape[-1]//2!=self.nl:
           raise Exception('Your model is {} layers, but you want to run it with {} layers (k={})'.format(self.nl, len(pk)//2,pk))
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # dont show overflow results
            
            # lax.scan version
            X0 = K, U
            final, _ = lax.scan(self.__step_for_scan, X0, jnp.arange(0,self.nt-1))
            #final, _ = lax.scan(self.__step2_for_scan, X0, jnp.arange(0,self.ntm-1))
            _, U = final
            
        self.Ur_traj = U
        self.Ua, self.Va = jnp.real(U[0,:]), jnp.imag(U[0,:])  # first layer
        
        return pk, U
