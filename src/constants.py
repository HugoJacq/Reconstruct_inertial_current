"""
Constants used in the current folder

to be used with 'reconstruct_inertial.py'
"""
import jax

grav = 10 # m/s2
rho = 1000 # kg/m3
distance_1deg_equator = 111000.0 # m
oneday = 86400 # s

# jax
FORCE_CPU = False
cpu_device = jax.devices('cpu')[0]
try:
    gpu_device = jax.devices('gpu')[0]
except:
    gpu_device = cpu_device
if FORCE_CPU:
    jax.config.update('jax_platform_name', 'cpu')