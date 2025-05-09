"""
4Dvar module
"""
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit,lax


# print infos if scipy.minimize
def print_info(cost,res):
    if np.isnan(cost(res['x'])): # , Uo, Vo, Ri
        print('The model has crashed.')
    else:
        print(' vector K solution ('+str(res.nit)+' iterations)',res['x'])
        print(' cost function value with K solution:',cost(res['x']))

class Variational:
    """
    """
    def __init__(self, model, observations):
        self.inJax = model.isJax
        Co = observations.get_obs()
        self.Uo = Co[0]
        self.Vo = Co[1]
        self.Ri = self.Uo*0.+1
        self.observations = observations
        self.obs_period = observations.obs_period
        self.model_dt = model.dt
        self.dt_forcing = model.dt_forcing
        self.model = model
        self.param = []
        self.J = []
        self.G = []

        # JAX
        self.jax_grad_cost_jit = jit(self.jax_grad_cost)
        self.jax_cost_jit = jit(self.jax_cost)
        self.__step_jax_cost_vect_jit_gpu = jit(self.__step_jax_cost_vect) 
        self.__step_jax_cost_vect_jit_cpu = jit(self.__step_jax_cost_vect) 
        
        #self.run_opt_jit = jit(self.run_opt)   
        #self.my_opt_jit = jit(self.my_opt)

  
    # NO JAX
    def nojax_cost(self, pk):
        """
        Computes the cost function of reconstructed current vs observations

        INPUT:
            - pk     : K vector
        OUTPUT:
            - scalar cost
            
        Note: this function works with numpy arrays
        """
        _, C = self.model.do_forward(pk)
        U, V = np.real(C)[0],np.imag(C)[0]
        A = U[::self.obs_period//self.model_dt]
        B = V[::self.obs_period//self.model_dt]
        J = 0.5 * np.sum( ((self.observations.Uo - A)*self.Ri)**2 + ((self.observations.Vo - B)*self.Ri)**2 )
        return J
 
    def nojax_grad_cost(self, pk):
        """
        Computes the gradient of the cost function

        INPUT:
            - pk    : K vector
        OUTPUT:
            - gradient of the cost function with respect to pk
            
        Note: this function works with numpy arrays
        """
        _, C = self.model.do_forward(pk)
        U, V = np.real(C)[0],np.imag(C)[0]

        # distance to observations (innovation)
        # this is used in the adjoint to add a forcing where obs is available
        d_U, d_V = np.zeros(len(U)), np.zeros(len(V))
        d_U[::self.obs_period//self.model_dt] = (self.observations.Uo - U[::self.obs_period//self.model_dt])*self.Ri
        d_V[::self.obs_period//self.model_dt] = (self.observations.Vo - V[::self.obs_period//self.model_dt])*self.Ri
        # computing the gradient of cost function with TGL
        dJ_pk = self.adjoint(pk, [d_U,d_V])

        return - dJ_pk
    
    # JAX
    def jax_cost(self, pk):
        """
        Computes the cost function of reconstructed current vs observations

        INPUT:
            - pk     : K vector
        OUTPUT:
            - scalar cost
            
        Note: this function works with numpy arrays
        """
        
        _, C = self.model.do_forward_jit(pk) # _, C = self.model.do_forward_jit(pk)
        U, V = jnp.real(C)[0], jnp.imag(C)[0]
        
        
        A = jnp.zeros( self.Uo.shape, dtype='float64')
        B = jnp.zeros( self.Uo.shape, dtype='float64')
        A = A.at[:].set(U[::self.obs_period//self.dt_forcing])
        B = B.at[:].set(V[::self.obs_period//self.dt_forcing])
        #A = A.at[:].set(U[::self.obs_period//self.model_dt])
        #B = B.at[:].set(V[::self.obs_period//self.model_dt])
        J = 0.5 * jnp.sum( ((self.observations.Uo - A)*self.Ri)**2 + ((self.observations.Vo - B)*self.Ri)**2 )
        return J 
 
    def jax_grad_cost(self, pk):        
        """
        Using grad from JAX
        """
        #return grad(self.jax_cost_jit)(pk) # this is much slower than jax.jacfwd
        return jax.jacfwd(self.jax_cost)(pk) # this is much faster than jax.grad
        #return jnp.real( grad(self.jax_cost_jit)(pk) )        

    def __step_jax_cost_vect(self,ik,arg0):
            array_pk, indexes, J = arg0
            vector_k = array_pk[ik]
            i,j = indexes[ik][0],indexes[ik][1]
            #jax.debug.print('ik, vector: {}, {}',ik,vector_k)
            J = J.at[i,j].set( self.jax_cost(vector_k) )
            return array_pk, indexes, J  
        
    def jax_cost_vect(self, array_pk, indexes):
        """
        Vectorized version of 'jax_cost'

        INPUT:
            - array_pk : A 2D array with:
                    axis=0 the number of pk vector tested
                    axis=1 the individual pk vector, can be N dimension (tested with 2)
                    
                    example:
                    
                    array_pk = jnp.asarray( [ [k0,k1] for k0 in jtested_values for k1 in jtested_values] )
                    
                    if you want to test values for a bigger than len(vector_k)=2, you can only change 2 values 
                    to be plotable...
        OUTPUT:
            -        
        """
        N, Nl2 = jnp.shape(array_pk) # squared number of vector k, size of vector k
        Nchange = 2 # number of component of vector k that change
        Nl = Nl2//2
        
        sizeN = jnp.sqrt(array_pk.shape[0]).astype(int)
        Jshape = tuple([sizeN]*Nchange)
        J = jnp.zeros( Jshape )
        
        arg0 = array_pk, indexes, J
        if Nl==1:
            _, _, J = lax.fori_loop(0,N,self.__step_jax_cost_vect_jit_cpu,arg0)
        else:
            _, _, J = lax.fori_loop(0,N,self.__step_jax_cost_vect_jit_gpu,arg0)
        return J.transpose()
    
    # COMMON WRAPPER
    def cost(self, pk, save_iter=False):
        if self.inJax:
            J = self.jax_cost_jit(pk)
        else:
            J = self.nojax_cost(pk)
        if save_iter:
            self.J.append(J)
        return J.astype(float)   
    
    def grad_cost(self, pk, save_iter=False):
        if self.inJax:
            G = self.jax_grad_cost_jit(pk)
            #print(jax.make_jaxpr(self.jax_grad_cost)(pk))
        else:
            self.adjoint = self.model.adjoint
            self.tgl = self.model.tgl
            G =  self.nojax_grad_cost(pk)
        if save_iter:
            self.G.append(G)
        return G.astype(float)   
    