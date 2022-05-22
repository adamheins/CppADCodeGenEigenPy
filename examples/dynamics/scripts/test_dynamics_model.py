import os
from functools import partial
import time
import timeit

import numpy as np
import jax
import jax.numpy as jnp

from CppADCodeGenEigenPy import CompiledModel

import util


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
LIB_DIR = SCRIPT_DIR + "/../lib"
DYNAMICS_MODEL_NAME = "DynamicsModel"
ROLLOUT_COST_MODEL_NAME = "RolloutCostModel"

STATE_DIM = 7 + 6
INPUT_DIM = 6


# It is convenient to create a wrapper around the C++ bindings to do things
# like build the parameter vector.
class CppDynamicsModelWrapper:
    """Wrapper around C++ bindings for DynamicsModel."""

    def __init__(self, mass, inertia):
        lib_path = os.path.join(LIB_DIR, "lib" + DYNAMICS_MODEL_NAME)
        self._model = CompiledModel(DYNAMICS_MODEL_NAME, lib_path)

        self.mass = mass
        self.inertia = inertia

    def _params(self):
        """Create the parameter vector."""
        return np.concatenate([[self.mass], self.inertia.reshape(9)])

    def evaluate(self, x, u):
        """Compute acceleration using forward dynamics."""
        inp = np.concatenate((x, u))
        return self._model.evaluate(inp, self._params())

    def jacobians(self, x, u):
        """Compute derivatives of forward dynamics w.r.t. state x and force input u."""
        inp = np.concatenate((x, u))
        J = self._model.jacobian(inp, self._params())
        dfdx = J[:, :STATE_DIM]
        dfdu = J[:, STATE_DIM:]
        return dfdx, dfdu


class JaxDynamicsModel:
    """Equivalent rigid body dynamics model in Python with JAX."""

    def __init__(self, mass, inertia):
        self.mass = mass
        self.inertia = inertia
        self.inertia_inv = jnp.linalg.inv(inertia)

        self.dfdx = jax.jit(jax.jacfwd(self.evaluate, argnums=0))
        self.dfdu = jax.jit(jax.jacfwd(self.evaluate, argnums=1))

    @partial(jax.jit, static_argnums=(0,))
    def evaluate(self, x, u):
        """Compute forward dynamics.

        Returns the acceleration based on the provided state x and force input
        u.
        """
        f, τ = u[:3], u[3:]
        _, q, v, ω = util.decompose_state(x)

        C_wo = util.quaternion_to_rotation_matrix(q)
        Iw = C_wo @ self.inertia @ C_wo.T
        Iw_inv = C_wo @ self.inertia_inv @ C_wo.T

        a = f / self.mass
        α = Iw_inv @ (τ - jnp.cross(ω, Iw @ ω))

        return jnp.concatenate((a, α))

    def jacobians(self, x, u):
        """Compute Jacobian of forward dynamics wrt x and u."""
        return self.dfdx(x, u), self.dfdu(x, u)


def main():
    # model parameters
    mass = 1.0
    inertia = np.eye(3)

    # state and input
    x = util.zero_state()
    u = np.ones(INPUT_DIM)

    # C++-based model which we bind to and load from a shared lib
    t = time.time()
    cpp_model = CppDynamicsModelWrapper(mass, inertia)
    dfdx_cpp, dfdu_cpp = cpp_model.jacobians(x, u)
    print(f"Time to load C++ model = {time.time() - t}")

    # jax-based model which is computed just in time
    t = time.time()
    jax_model = JaxDynamicsModel(mass, inertia)
    dfdx_jax, dfdu_jax = jax_model.jacobians(x, u)
    print(f"Time to load Jax model = {time.time() - t}")

    acc_cpp = cpp_model.evaluate(x, u)
    acc_jax = jax_model.evaluate(x, u)

    # check that both models actually get the same results
    assert np.isclose(
        acc_cpp, acc_jax
    ).all(), "Forward dynamics result not the same between models"
    assert np.isclose(dfdx_cpp, dfdx_jax).all(), "dfdx is not the same between models."
    assert np.isclose(dfdu_cpp, dfdu_jax).all(), "dfdu is not the same between models."

    # compare runtime evaluation time of Jacobians
    n = 100000
    cpp_time = timeit.timeit("cpp_model.jacobians(x, u)", number=n, globals=locals())
    jax_time = timeit.timeit("jax_model.jacobians(x, u)", number=n, globals=locals())
    print(f"Time to evaluate C++ model jacobians {n} times = {cpp_time} sec")
    print(f"Time to evaluate JAX model jacobians {n} times = {jax_time} sec")


if __name__ == "__main__":
    main()
