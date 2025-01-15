import numpy as np
import xarray as xr
import warnings

import jax
import jax.numpy as jnp

class jUnstek1D:
    """
    JAX version, Unsteady Ekman Model 1D, with N layers 
    
    See : https://doi.org/10.3390/rs15184526
    """
    def __init__(self, Nl, forcing):
        self.nl = Nl