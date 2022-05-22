import os

import numpy as np
import pinocchio

from CppADCodeGenEigenPy import CompiledModel


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")

ROBOT_NAME = "kinova"  # can be either "kinova" or "ur5"
LIB_DIR = os.path.join(ROOT_DIR, "lib", ROBOT_NAME)
AD_MODEL_NAME = "InverseDynamicsModel"
URDF_PATH = os.path.join(ROOT_DIR, "urdf", ROBOT_NAME + ".urdf")


class CppInverseDynamicsModelWrapper:
    """Wrapper around C++ bindings for InverseDynamicsModel."""

    def __init__(self, nq, nv):
        lib_path = os.path.join(LIB_DIR, "lib" + AD_MODEL_NAME)
        self._model = CompiledModel(AD_MODEL_NAME, lib_path)
        self.nq = nq
        self.nv = nv

    def evaluate(self, q, v, a):
        """Compute torques corresponding to position q, velocity v, acceleration a."""
        inp = np.concatenate((q, v, a))
        return self._model.evaluate(inp)

    def jacobians(self, q, v, a):
        """Compute derivatives of forward dynamics w.r.t. state x and force input u."""
        inp = np.concatenate((q, v, a))
        J = self._model.jacobian(inp)
        dτdq = J[:, : self.nq]
        dτdv = J[:, self.nq : self.nq + self.nv]
        dτda = J[:, -self.nv :]
        return dτdq, dτdv, dτda


def main():
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(0)

    # load pinocchio model
    pin_model = pinocchio.buildModelFromUrdf(URDF_PATH)
    pin_data = pin_model.createData()

    # determine the joint indices that actually correpond to the robot's
    # joints; i.e., the joints which pinocchio computes inverse dynamics for
    idx_qs = []
    for joint in pin_model.joints:
        if joint.idx_q >= 0:
            idx_qs.append(joint.idx_q)

    # load our auto-diffed model
    ad_model = CppInverseDynamicsModelWrapper(pin_model.nq, pin_model.nv)

    # random joint position, velocity, acceleration
    q = pinocchio.randomConfiguration(pin_model)
    v = np.random.random(pin_model.nv) - 0.5
    a = np.random.random(pin_model.nv) - 0.5

    # compute inverse dynamics and derivatives using built-in pinocchio
    # functions
    τ_pin = pinocchio.rnea(pin_model, pin_data, q, v, a)
    dτdq_pin, dτdv_pin, dτda_pin = pinocchio.computeRNEADerivatives(
        pin_model, pin_data, q, v, a
    )

    # compute inverse dynamics and derivatives using our compiled model
    τ_ad = ad_model.evaluate(q, v, a)
    dτdq_ad_augmented, dτdv_ad, dτda_ad = ad_model.jacobians(q, v, a)

    # a continuous joint q_i is parameterized using cos(q_i) and sin(q_i)
    # (i.e., two entries in the q vector). Below, we detect this case (we
    # assume nq == 2 implies a continuous joint), and compute the actual
    # derivative dτ/dq_i (like Pinocchio) using the derivatives dτ/d cos(q_i)
    # and dτ/d sin(q_i) provided by auto-diff.
    dτdq_ad = np.zeros_like(dτdq_pin)
    idx = 0
    for joint in pin_model.joints:
        if joint.idx_q < 0:
            continue

        if joint.nq == 1:
            # normal revolute joint
            dτdq_ad[:, idx] = dτdq_ad_augmented[:, joint.idx_q]
        elif joint.nq == 2:
            # continuous joint
            c = q[joint.idx_q]
            s = q[joint.idx_q + 1]
            dτdc = dτdq_ad_augmented[:, joint.idx_q]
            dτds = dτdq_ad_augmented[:, joint.idx_q + 1]
            dτdq_ad[:, idx] = -dτdc * s + dτds * c
        else:
            raise ValueError("I don't know how to handle a joint with nq > 2.")
        idx += 1

    # check that the built-in Pinocchio model and our auto-diff model give
    # the same results
    assert np.isclose(τ_ad, τ_pin).all(), "τ not the same between models"
    assert np.isclose(dτdq_ad, dτdq_pin).all(), "dτdq not the same between models"
    assert np.isclose(dτdv_ad, dτdv_pin).all(), "dτdv not the same between models"
    assert np.isclose(dτda_ad, dτda_pin).all(), "dτda not the same between models"

    print(f"Auto-diffed and built-in Pinocchio models are equal for {ROBOT_NAME} robot.")


if __name__ == "__main__":
    main()
