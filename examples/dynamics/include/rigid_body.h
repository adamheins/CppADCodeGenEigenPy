#pragma once

#include <Eigen/Eigen>

#include "types.h"

template <typename Scalar>
struct RigidBody {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RigidBody(const Scalar mass, const Mat3<Scalar>& inertia)
        : mass(mass), inertia(inertia), inertia_inv(inertia.inverse()) {}

    // Compute forward dynamics for the rigid body. Force inputs are assumed to
    // be applied in the inertial (world) frame.
    Vec6<Scalar> forward_dynamics(const StateVec<Scalar>& x,
                                  const InputVec<Scalar>& u) const {
        // Compute inertia matrix rotated into the fixed world frame.
        Mat3<Scalar> C_wo = orientation(x).toRotationMatrix();
        Mat3<Scalar> Iw = C_wo * inertia * C_wo.transpose();
        Mat3<Scalar> Iw_inv = C_wo * inertia_inv * C_wo.transpose();

        Vec3<Scalar> force = u.head(3);
        Vec3<Scalar> torque = u.tail(3);
        Vec3<Scalar> v = linear_velocity(x);
        Vec3<Scalar> omega = angular_velocity(x);

        // Find acceleration from Newton-Euler equations
        Vec3<Scalar> a = force / mass;
        Vec3<Scalar> alpha = Iw_inv * (torque - omega.cross(Iw * omega));

        Vec6<Scalar> A;
        A << a, alpha;
        return A;
    }

    // Roll out the forward dynamics over num_steps time steps, each spaced dt
    // seconds apart.
    std::vector<StateVec<Scalar>> rollout(
        const StateVec<Scalar>& x0, const std::vector<InputVec<Scalar>>& us,
        const Scalar dt, const size_t num_steps) const {
        StateVec<Scalar> x = x0;
        std::vector<StateVec<Scalar>> xs;
        for (InputVec<Scalar> u : us) {
            Vec6<Scalar> A = forward_dynamics(x, u);

            // Integrate linear portion of state
            Vec3<Scalar> r0 = position(x);
            Vec3<Scalar> v0 = linear_velocity(x);
            Vec3<Scalar> a = A.head(3);

            Vec3<Scalar> r1 = r0 + dt * v0 + 0.5 * dt * dt * a;
            Vec3<Scalar> v1 = v0 + dt * a;

            // Integrate rotation portion
            Eigen::Quaternion<Scalar> q0 = orientation(x);
            Vec3<Scalar> omega0 = angular_velocity(x);
            Vec3<Scalar> alpha = A.tail(3);
            Vec3<Scalar> omega1 = omega0 + dt * alpha;

            // First term of the Magnus expansion
            Vec3<Scalar> aa_vec = 0.5 * dt * (omega0 + omega1);

            // Map to a quaternion via exponential map (note this
            // implementation is not robust against numerical problems as angle
            // -> 0)
            Scalar angle = aa_vec.norm();
            Vec3<Scalar> axis = aa_vec / angle;
            Scalar c = cos(0.5 * angle);
            Scalar s = sin(0.5 * angle);

            Eigen::Quaternion<Scalar> qw;
            qw.coeffs() << s * axis, c;
            Eigen::Quaternion<Scalar> q1 = qw * q0;

            x << r1, q1.coeffs(), v1, omega1;
            xs.push_back(x);
        }
        return xs;
    }

    static StateVec<Scalar> zero_state() {
        StateVec<Scalar> x = StateVec<Scalar>::Zero();
        x(6) = 1;  // w of quaternion
        return x;
    }

    static Vec3<Scalar> orientation_error(const Eigen::Quaternion<Scalar>& q,
                                          const Eigen::Quaternion<Scalar>& qd) {
        // Vector part of qd.inverse() * q. We implement it manually here to
        // avoid code branches (that check numerical stability) that AD cannot
        // deal with.
        return qd.w() * q.vec() - q.w() * qd.vec() - qd.vec().cross(q.vec());
    }

    static StateErrorVec<Scalar> state_error(const StateVec<Scalar>& x,
                                             const StateVec<Scalar>& xd) {
        Eigen::Quaternion<Scalar> q = orientation(x);
        Eigen::Quaternion<Scalar> qd = orientation(xd);

        StateErrorVec<Scalar> e;
        e << position(xd) - position(x), orientation_error(q, qd),
            linear_velocity(xd) - linear_velocity(x),
            angular_velocity(xd) - angular_velocity(x);
        return e;
    }

    static Vec3<Scalar> position(const StateVec<Scalar>& x) {
        return x.head(3);
    }

    static Eigen::Quaternion<Scalar> orientation(const StateVec<Scalar>& x) {
        Eigen::Quaternion<Scalar> q;
        q.coeffs() << x.segment(3, 4);
        return q;
    }

    // Get linear velocity component of the state
    static Vec3<Scalar> linear_velocity(const StateVec<Scalar>& x) {
        return x.segment(7, 3);
    }

    // Get angular velocity component of the state
    static Vec3<Scalar> angular_velocity(const StateVec<Scalar>& x) {
        return x.segment(10, 3);
    }

    Scalar mass;
    Mat3<Scalar> inertia;
    Mat3<Scalar> inertia_inv;
};
