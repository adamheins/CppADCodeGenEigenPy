import os
from functools import partial
import time
import timeit

import numpy as np
import jax
import jax.numpy as jnp

from CppADCodeGenEigenPy import CompiledModel

import util

# use 64-bit floating point numbers with jax to match the C++ model
# otherwise, small numerical precision errors can cause model mismatch
jax.config.update("jax_enable_x64", True)


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
LIB_DIR = SCRIPT_DIR + "/../lib"
ROLLOUT_COST_MODEL_NAME = "RolloutCostModel"

STATE_DIM = 7 + 6
INPUT_DIM = 6

# NOTE these are currently fixed on the C++ side, and need to be set to the
# same values here
NUM_TIME_STEPS = 10
TIMESTEP = 0.1


class CppRolloutCostModelWrapper:
    """Wrapper around C++ bindings for DynamicsModel."""

    def __init__(self, mass, inertia):
        lib_path = os.path.join(LIB_DIR, "lib" + ROLLOUT_COST_MODEL_NAME)
        self._model = CompiledModel(ROLLOUT_COST_MODEL_NAME, lib_path)

        self.mass = mass
        self.inertia = inertia

    def _params(self, xds):
        """Create the parameter vector."""
        return np.concatenate([[self.mass], self.inertia.flatten(), xds.flatten()])

    def _input(self, x0, us):
        return np.concatenate((x0, us.flatten()))

    def evaluate(self, x0, us, xds):
        """Compute acceleration using forward dynamics."""
        return self._model.evaluate(self._input(x0, us), self._params(xds))

    def jacobians(self, x0, us, xds):
        """Compute derivatives of forward dynamics w.r.t. state x and force input u."""
        J = self._model.jacobian(self._input(x0, us), self._params(xds))
        dfdx0 = J[:, :STATE_DIM]
        dfdus = J[:, STATE_DIM:]
        return dfdx0, dfdus


def orientation_error(q, qd):
    """Error between two quaternions."""
    # This is the vector portion of qd.inverse() * q
    return qd[3] * q[:3] - q[3] * qd[:3] - jnp.cross(qd[:3], q[:3])


def state_error(x, xd):
    """Error between actual state x and desired state xd."""
    r, q, v, ω = util.decompose_state(x)
    rd, qd, vd, ωd = util.decompose_state(xd)

    r_err = rd - r
    q_err = orientation_error(q, qd)
    v_err = vd - v
    ω_err = ωd - ω

    return jnp.concatenate((r_err, q_err, v_err, ω_err))


def integrate_state(x0, A, dt):
    """Integrate state x0 given acceleration A over timestep dt."""
    r0, q0, v0, ω0 = util.decompose_state(x0)
    a, α = A[:3], A[3:]

    r1 = r0 + dt * v0 + 0.5 * dt ** 2 * a
    v1 = v0 + dt * a

    ω1 = ω0 + dt * α

    aa = 0.5 * dt * (ω0 + ω1)
    angle = jnp.linalg.norm(aa)
    axis = aa / angle

    c = jnp.cos(0.5 * angle)
    s = jnp.sin(0.5 * angle)
    qw = jnp.append(s * axis, c)
    q1 = util.quaternion_multiply(qw, q0)

    return jnp.concatenate((r1, q1, v1, ω1))


class JaxRolloutCostModel:
    """Equivalent cost model in Python with JAX."""

    def __init__(self, mass, inertia):
        self.mass = mass
        self.inertia = inertia
        self.inertia_inv = jnp.linalg.inv(inertia)

        self.dfdx0 = jax.jit(jax.jacfwd(self.evaluate, argnums=0))
        self.dfdus = jax.jit(jax.jacfwd(self.evaluate, argnums=1))

    @partial(jax.jit, static_argnums=(0,))
    def forward_dynamics(self, x, u):
        f, τ = u[:3], u[3:]
        _, q, v, ω = util.decompose_state(x)

        C_wo = util.quaternion_to_rotation_matrix(q)
        Iw = C_wo @ self.inertia @ C_wo.T
        Iw_inv = C_wo @ self.inertia_inv @ C_wo.T

        a = f / self.mass
        α = Iw_inv @ (τ - jnp.cross(ω, Iw @ ω))

        return jnp.concatenate((a, α))

    def evaluate(self, x0, us, xds):
        """Compute the cost."""

        def state_func(x, u):
            A = self.forward_dynamics(x, u)
            x = integrate_state(x, A, TIMESTEP)
            return x, x

        _, xs = jax.lax.scan(state_func, x0, us)

        def cost_func(cost, datum):
            x, xd, u = (
                datum[:STATE_DIM],
                datum[STATE_DIM : 2 * STATE_DIM],
                datum[-INPUT_DIM:],
            )
            e = state_error(x, xd)
            cost = cost + 0.5 * (e @ e + 0.1 * u @ u)
            return cost, datum

        data = jnp.hstack((xs, xds, us))
        cost, _ = jax.lax.scan(cost_func, 0, data)

        return cost

    def jacobians(self, x0, us, xds):
        """Compute Jacobians of cost wrt initial state x0 and inputs us."""
        return self.dfdx0(x0, us, xds), self.dfdus(x0, us, xds).flatten()


def main():
    np.random.seed(0)

    # model parameters
    mass = 1.0
    inertia = np.eye(3)

    # initial state
    x0 = util.zero_state()

    # force/torque inputs
    us = np.random.random((NUM_TIME_STEPS, INPUT_DIM))

    # desired states
    xd = util.zero_state()
    xd[:3] = [1, 1, 1]  # want body to move position
    xds = np.tile(xd, (NUM_TIME_STEPS, 1))

    # C++-based model which we bind to and load from a shared lib
    t = time.time()
    cpp_model = CppRolloutCostModelWrapper(mass, inertia)
    dfdx0_cpp, dfdus_cpp = cpp_model.jacobians(x0, us, xds)
    print(f"Time to load C++ model jacobians = {time.time() - t} sec")

    # jax-based model which is computed just in time
    t = time.time()
    jax_model = JaxRolloutCostModel(mass, inertia)
    dfdx0_jax, dfdus_jax = jax_model.jacobians(x0, us, xds)
    print(f"Time to load JAX model jacobians = {time.time() - t} sec")

    cost_cpp = cpp_model.evaluate(x0, us, xds)
    cost_jax = jax_model.evaluate(x0, us, xds)

    # check that both models actually get the same results
    assert np.isclose(cost_cpp, cost_jax), "Cost is not the same between models."
    assert np.isclose(
        dfdx0_cpp, dfdx0_jax
    ).all(), "dfdx0 is not the same between models."
    assert np.isclose(
        dfdus_cpp, dfdus_jax
    ).all(), "dfdus is not the same between models."

    # compare runtime evaluation time of Jacobians
    n = 100000
    cpp_time = timeit.timeit(
        "cpp_model.jacobians(x0, us, xds)", number=n, globals=locals()
    )
    jax_time = timeit.timeit(
        "jax_model.jacobians(x0, us, xds)", number=n, globals=locals()
    )
    print(f"Time to evaluate C++ model jacobians {n} times = {cpp_time} sec")
    print(f"Time to evaluate JAX model jacobians {n} times = {jax_time} sec")


if __name__ == "__main__":
    main()
