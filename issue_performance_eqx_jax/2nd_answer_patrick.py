import numpy as np
import os
import timeit

os.environ["EQX_ON_ERROR"] = "nan"
import jax
import jax.numpy as jnp
from jax import lax
import equinox as eqx
import diffrax
from diffrax import Euler, diffeqsolve, ODETerm

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')

TAx = jnp.array(0.2)
TAy = jnp.array(0.0)
fc = jnp.array(1e-4)
dt_forcing = 3600
t0 = 0
t1 = 28 * 86400
nt = 28 * 86400 // 60
dt = 60


@jax.jit
def classic_slab1D_eqx(pk):
    K = jnp.exp(pk)
    U = jnp.zeros(nt, dtype="complex")

    def __one_step(X0, it):
        K, U = X0
        TA = TAx + 1j * TAy
        U = U.at[it + 1].set(
            U[it] + dt * (-1j * fc * U[it] + K[0] * TA - K[1] * U[it])
        )
        X0 = K, U
        return X0, X0

    X0 = K, U
    final, _ = lax.scan(
        lambda carry, y: __one_step(carry, y), X0, jnp.arange(0, nt - 1)
    )
    _, U = final
    return jnp.real(U), jnp.imag(U)


@jax.jit
def classic_slab1D_dfx(pk):
    K = jnp.exp(pk)

    def vector_field(t, C, args):
        U, V = C
        fc, K, TAx, TAy = args
        # physic
        d_U = fc * V + K[0] * TAx - K[1] * U
        d_V = -fc * U + K[0] * TAy - K[1] * V
        d_y = d_U, d_V
        return d_y

    sol = diffeqsolve(
        terms=ODETerm(vector_field),
        solver=Euler(),
        t0=t0,
        t1=t1,
        y0=(0.0, 0.0),  # ,
        args=(fc, K, TAx, TAy),
        dt0=None,  # dt,
        stepsize_controller=diffrax.StepTo(
            jnp.arange(t0, t1 + dt, dt)
        ),
        saveat=diffrax.SaveAt(steps=True),
        adjoint=diffrax.ForwardMode(),
        max_steps=nt,
    ).ys

    return sol[0], sol[1]


def benchmark(func, N=100):
    pk = jnp.asarray([-8.0, -13.0])
    func(pk)
    L = np.array(timeit.repeat(lambda: func(pk), number=1, repeat=N))
    print(func.__name__, f"min={L.min()}", f"mean={L.mean()}", f"std={L.std()}")


benchmark(classic_slab1D_eqx)
benchmark(classic_slab1D_dfx)