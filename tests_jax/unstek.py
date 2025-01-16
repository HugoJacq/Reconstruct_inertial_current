import numpy as np
import xarray as xr
import warnings

class Unstek1D:
    """
    Unsteady Ekman Model 1D, with N layers 
    
    See : https://doi.org/10.3390/rs15184526
    """
    
    def __init__(self, Nl, forcing, observations):
        # from dataset
        self.nt = len(forcing.time)
        self.dt = forcing.time[1] - forcing.time[0]
        # forcing
        self.TA = forcing.TAx + 1j*forcing.TAy # wind = windU + j*windV
        self.fc = forcing.fc  
        # observations
        self.Uo, self.Vo = observations.get_obs()
        self.Ri = self.Uo*0.+1
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
            
                      
    def step(self, pk, it,U):
        """
        1 time step advance
        
        INPUT:
            - pk : K vector
            - it: current time step
            - U: velocity at current time step 
        OUTPUT:
            - U: updated velocity at next time step 
        """
        K = self.K_transform(pk)
        
        for ik in range(self.nl):
            if ((ik==0)&(ik==self.nl-1)): 
                U[ik][it+1] = U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] +K[2*ik]*self.TA[it] - K[2*ik+1]*(U[ik][it]) )
            else:
                if ik==0: U[ik][it+1] = U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] +K[2*ik]*self.TA[it] - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) )
                elif ik==self.nl-1: U[ik][it+1] = U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*U[ik][it] )
                else: U[ik][it+1] = U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) )
        return U
        
    def do_forward(self, pk, return_traj=False):
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
        if len(pk)//2!=self.nl:
            raise Exception('Your model is {} layers, but you want to run it with {} layers (k={})'.format(self.nl, len(pk)//2,pk))
        else:
            K = self.K_transform(pk)
            
            
        U0 = np.zeros((self.nl,self.nt), dtype='complex')
        with warnings.catch_warnings(action="ignore"): # dont show overflow results
            for it in range(self.nt-1):
                U0 = self.step(pk, it,U0)
        if return_traj: 
            self.Ur_traj = U0
            return U0
        else: 
            self.Ua,self.Va = np.real(U0[0,:]), np.imag(U0[0,:])
            return self.Ua,self.Va
       
       
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
    
    def tgl(self, time, k, dk):
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

        nl = int(len(K)//2)
        U = [None]*nl
        dU = [None]*nl
        for ik in range(nl):
            U[ik]=np.zeros((len(time)), dtype='complex')
            dU[ik]=np.zeros((len(time)), dtype='complex')

        for it in range(len(time)-1):
            U[ik][:], dU[ik][:] = self.step_tgl(K, dK, it, U, dU)

        return np.real(dU[0]), np.imag(dU[0])   

        
    def step_adj(self, pk, it, ad_U, ad_K, U):
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
        K = self.K_transform(pk)
        for ik in range(self.nl):
            if ((ik==0)&(ik==self.nl-1)): 
                ad_U[ik][it] += ad_U[ik][it+1]*np.conjugate(1+self.dt*( -1j*self.fc- K[2*ik+1]))
                ad_K[2*ik] += self.dt*(ad_U[ik][it+1]*np.conjugate(self.TA[it]))
                ad_K[2*ik+1] += self.dt*(-ad_U[ik][it+1]*np.conjugate(U[ik][it]))
                
            else:
                if ik==0: 
                    ad_U[ik][it] += ad_U[ik][it+1]*np.conjugate(1+self.dt*( -1j*self.fc- K[2*ik+1])) 
                    ad_U[ik+1][it] += ad_U[ik][it+1]*np.conjugate(self.dt*self.K[2*ik+1])
                    ad_K[2*ik] += self.dt*(ad_U[ik][it+1]*np.conjugate(self.TA[it])) 
                    ad_K[2*ik+1] += self.dt*(-ad_U[ik][it+1]*np.conjugate(U[ik][it]-U[ik+1][it]))

                elif ik==self.nl-1: 
                    ad_U[ik][it] += ad_U[ik][it+1]*np.conjugate(1+self.dt*( -1j*self.fc- K[2*ik]- K[2*ik+1]))
                    ad_U[ik-1][it] += ad_U[ik][it+1]*np.conjugate(self.dt*self.K[2*ik])
                    ad_K[2*ik] += ad_U[ik][it+1]*(-self.dt*np.conjugate(U[ik][it]-U[ik-1][it]))
                    ad_K[2*ik+1] += ad_U[ik][it+1]*self.dt*np.conjugate(-U[ik][it])

                else: 
                    ad_U[ik][it] += ad_U[ik][it+1]*np.conjugate(1+self.dt*( -1j*self.fc- K[2*ik]- K[2*ik+1]))
                    ad_U[ik+1][it] += ad_U[ik][it+1]*np.conjugate(self.dt*self.K[2*ik+1])
                    ad_U[ik-1][it] += ad_U[ik][it+1]*np.conjugate(self.dt*self.K[2*ik])
                    ad_K[2*ik] += ad_U[ik][it+1]*np.conjugate(-self.dt*(U[ik][it]-U[ik-1][it]))
                    ad_K[2*ik+1] += self.dt*(-ad_U[ik][it+1]*np.conjugate(U[ik][it]-U[ik+1][it]))
        return ad_U, ad_K
                   
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
        U = self.do_forward(pk, return_traj=True)
        
        ad_U = [None]*self.nl
        for ik in range(self.nl):
            ad_U[ik]=np.zeros(self.nt, dtype='complex')
        ad_U[0] = d[0] + 1j*d[1]

        ad_K = np.zeros(len(pk), dtype='complex')

        for it in np.arange(self.nt - 1)[::-1]:
            ad_U, ad_K = self.step_adj(pk, it, ad_U, ad_K, U)

        ad_k = self.K_transform(pk, order=1, function='exp')*ad_K # watch out if K_transform is different from exp !

        return np.real(ad_k)


    def cost(self, pk):
        """
        Computes the cost function of reconstructed current vs observations

        INPUT:
            - pk     : K vector
        OUTPUT:
            - scalar cost
            
        Note: this function works with numpy arrays
        """
        U, V = self.do_forward(pk)
        with warnings.catch_warnings(action="ignore"): # dont show overflow results
            J = 0.5 * np.nansum( ((self.Uo - U)*self.Ri)**2 + ((self.Vo - V)*self.Ri)**2 )
        if np.sum( np.isnan(U) ) + np.sum(np.isnan(V))>0:
            # some nan have been detected, the model has crashed with 'pk'
            # so J is nan.
            J = np.nan
        return J

    def grad_cost(self, pk):
        """
        Computes the gradient of the cost function

        INPUT:
            - pk    : K vector
        OUTPUT:
            - gradient of the cost function
            
        Note: this function works with numpy arrays
        """
        U, V = self.do_forward(pk)

        # distance to observations (innovation)
        # this is used in the adjoint to add a forcing where obs is available
        d_U = (self.Uo - U)*self.Ri
        d_V = (self.Vo - V)*self.Ri
        #   = 0 where no observation available
        d_U[np.isnan(d_U)]=0.
        d_V[np.isnan(d_V)]=0.
        # computing the gradient of cost function with TGL
        dJ_pk = self.adjoint(pk, [d_U,d_V])

        return -dJ_pk

