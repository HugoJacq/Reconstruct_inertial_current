import numpy as np
import xarray as xr
import warnings

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
            return jnp.exp(pk).astype('complex')
        else:
            raise Exception('K_transform function '+function+' is not available, retry')

    def __Onelayer(self, arg1):
        """
        1 time iteration of unstek model when 1 layer
        """
        it, K, U = arg1
        ik = 0
        U = U.at[ik,it+1].set( U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] +K[2*ik]*self.TA[it] - K[2*ik+1]*(U[ik][it]) ) )
        return it, K, U
        
    def __Nlayer_midlayers_for_scan(self,X0,ik):
        it, K, U = X0
        U = U.at[ik,it+1].set( U[ik][it] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik][it] 
                                    - K[2*ik]*(U[ik][it]-U[ik-1][it]) 
                                    - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) ) )
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
        U = U.at[ik, it+1].set( U[ik][it] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik][it] 
                                    + K[2*ik]*self.TA[it] 
                                    - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) ) )
        # bottom
        ik = -1
        U = U.at[ik,it+1].set(  U[ik][it] + 
                                self.dt*( 
                                    - 1j*self.fc*U[ik][it] 
                                    - K[2*ik]*(U[ik][it]-U[ik-1][it]) 
                                    - K[2*ik+1]*U[ik][it] ) )
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
        if len(pk)//2!=self.nl:
            raise Exception('Your model is {} layers, but you want to run it with {} layers (k={})'.format(self.nl, len(pk)//2,pk))

        # starting from nul current
        U = jnp.zeros( (self.nl,self.nt), dtype='complex')

        arg0 = pk, U
        X0 = pk, U
        with warnings.catch_warnings(action="ignore"): # dont show overflow results
            _, U = lax.fori_loop(1,self.nt,self.__one_step,arg0)
            # X, _ = lax.   scan(jit(self.__one_step_for_scan), X0, jnp.arange(1,self.nt) )
            # _, U = X
            
        self.Ur_traj = U
        self.Ua, self.Va = jnp.real(U[0,:]), jnp.imag(U[0,:])  # first layer

        return pk, U
    

    # def tgl(self, k0, dk0):
    #     U0 = jnp.zeros((self.nl,self.nt), dtype='complex')
    #     dU0 = jnp.zeros((self.nl,self.nt), dtype='complex')
    #     K0 = self.K_transform(k0)
    #     dK0 = self.K_transform(k0, order=1)*dk0
    #     _, (dK,dU) = jvp(self.do_forward_jit, (K0, U0), (dK0, dU0) )
    #     return dK,dU

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
