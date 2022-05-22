#pragma once

#include <Eigen/Eigen>

#include <CppADCodeGenEigenPy/ADModel.h>

#include "rigid_body.h"
#include "types.h"

namespace ad = CppADCodeGenEigenPy;

const double TIMESTEP = 0.1;
const size_t NUM_TIME_STEPS = 10;
const size_t NUM_INPUT = INPUT_DIM * NUM_TIME_STEPS;

template <typename Scalar>
struct RolloutCostModel : public ad::ADModel<Scalar> {
    using typename ad::ADModel<Scalar>::ADScalar;
    using typename ad::ADModel<Scalar>::ADVector;
    using typename ad::ADModel<Scalar>::ADMatrix;

    ADVector input() const override {
        // Input is the initial state and the force and torque at each timestep
        ADVector inp = ADVector::Ones(STATE_DIM + NUM_INPUT);
        inp.head(STATE_DIM) = RigidBody<ADScalar>::zero_state();
        return inp;
    }

    ADVector parameters() const override {
        // Parameters consist of mass (1), inertia (9), and desired states
        // (STATE_DIM * NUM_TIME_STEPS)
        ADScalar mass(1.0);
        Mat3<ADScalar> inertia = Mat3<ADScalar>::Identity();

        ADVector p(1 + 9 + STATE_DIM * NUM_TIME_STEPS);

        p(0) = mass;
        p.segment(1, 9) = Eigen::Map<ADVector>(inertia.data(), 9, 1);

        StateVec<ADScalar> x0 = RigidBody<ADScalar>::zero_state();
        for (int i = 0; i < NUM_TIME_STEPS; ++i) {
            p.segment(10 + i * STATE_DIM, STATE_DIM) = x0;
        }
        return p;
    }

    static ADScalar get_mass(const ADVector& parameters) {
        return parameters(0);
    }

    static Mat3<ADScalar> get_inertia(const ADVector& parameters) {
        ADVector inertia_vec = parameters.segment(1, 3 * 3);
        Eigen::Map<Mat3<ADScalar>> inertia(inertia_vec.data(), 3, 3);
        return inertia;
    }

    static std::vector<StateVec<ADScalar>> get_desired_states(
        const ADVector& parameters) {
        const size_t start_idx = 10;
        std::vector<StateVec<ADScalar>> xds;
        for (size_t i = 0; i < NUM_TIME_STEPS; ++i) {
            xds.push_back(
                parameters.segment(start_idx + i * STATE_DIM, STATE_DIM));
        }
        return xds;
    }

    static StateVec<ADScalar> get_initial_state(const ADVector& input) {
        return input.head(STATE_DIM);
    }

    static std::vector<InputVec<ADScalar>> get_wrenches(const ADVector& input) {
        const size_t start_idx = STATE_DIM;
        std::vector<InputVec<ADScalar>> us;
        for (size_t i = 0; i < NUM_TIME_STEPS; ++i) {
            us.push_back(
                input.segment(start_idx + i * INPUT_DIM, INPUT_DIM));
        }
        return us;
    }

    /**
     * Compute cost of the rollout.
     */
    ADVector function(const ADVector& input,
                      const ADVector& parameters) const override {
        ADScalar mass = get_mass(parameters);
        Mat3<ADScalar> inertia = get_inertia(parameters);

        std::vector<StateVec<ADScalar>> xds = get_desired_states(parameters);
        StateVec<ADScalar> x0 = get_initial_state(input);
        std::vector<InputVec<ADScalar>> us = get_wrenches(input);

        // Do the rollout
        RigidBody<ADScalar> body(mass, inertia);
        std::vector<StateVec<ADScalar>> xs =
            body.rollout(x0, us, ADScalar(TIMESTEP), NUM_TIME_STEPS);

        // Compute cost
        ADVector cost = ADVector::Zero(1);
        for (int i = 0; i < NUM_TIME_STEPS; ++i) {
            StateErrorVec<ADScalar> x_err =
                RigidBody<ADScalar>::state_error(xs[i], xds[i]);

            ADScalar step_cost =
                ADScalar(0.5) * (x_err.dot(x_err) +
                                 ADScalar(0.1) * us[i].dot(us[i]));
            cost(0) += step_cost;
        }
        return cost;
    }
};
