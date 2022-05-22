import numpy as np
import jax.numpy as jnp


def skew(v):
    """Convert 3-dimensional array to skew-symmetric matrix."""
    return jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def quaternion_multiply(q0, q1):
    """Hamilton product of two quaternions, stored (x, y, z, w)."""
    v0, w0 = q0[:3], q0[3]
    v1, w1 = q1[:3], q1[3]
    return jnp.append(w0 * v1 + w1 * v0 + jnp.cross(v0, v1), w0 * w1 - v0 @ v1)


def quaternion_to_rotation_matrix(q):
    """Convert a quaternion to a rotation matrix."""
    # v, w = q[:3], q[3]
    # C = (w ** 2 - v @ v) * jnp.eye(3) + 2 * w * skew(v) + 2 * jnp.outer(v, v)

    # This is the algorithm used by Eigen. It actually produces slightly
    # different derivative values than the commented-out lines above, so I had
    # to use this version to match the Eigen model.
    x, y, z, w = q

    tx = 2 * x
    ty = 2 * y
    tz = 2 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = tx * y
    txz = tx * z
    tyy = ty * y
    tyz = ty * z
    tzz = tz * z

    return jnp.array([
        [1 - (tyy + tzz), txy - twz, txz + twy],
        [txy + twz, 1 - (txx + tzz), tyz - twx],
        [txz - twy, tyz + twx, 1 - (txx + tyy)]])


def decompose_state(x):
    """Decompose state into position, orientation, linear and angular velocity."""
    r = x[:3]
    q = x[3:7]
    v = x[7:10]
    ω = x[10:]
    return r, q, v, ω


def zero_state():
    x = np.zeros(7 + 6)
    x[6] = 1  # for quaternion
    return x

