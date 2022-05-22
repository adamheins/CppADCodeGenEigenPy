#pragma once

#include <Eigen/Eigen>

#include <CppADCodeGenEigenPy/ADModel.h>

#include "rigid_body.h"
#include "types.h"

namespace ad = CppADCodeGenEigenPy;

template <typename Scalar>
struct DynamicsModel : public ad::ADModel<Scalar> {
    using typename ad::ADModel<Scalar>::ADScalar;
    using typename ad::ADModel<Scalar>::ADVector;
    using typename ad::ADModel<Scalar>::ADMatrix;

    // Generate the input used when differentiating the function
    ADVector input() const override {
        return ADVector::Ones(STATE_DIM + INPUT_DIM);
    }

    // Generate parameters used when differentiating the function
    ADVector parameters() const override {
        ADScalar mass(1.0);
        Mat3<ADScalar> inertia = Mat3<ADScalar>::Identity();

        ADVector p(1 + inertia.size());
        p << mass, Eigen::Map<ADVector>(inertia.data(), inertia.size());
        return p;
    }

    ADScalar get_mass(const ADVector& parameters) const {
        return parameters(0);
    }

    Mat3<ADScalar> get_inertia(const ADVector& parameters) const {
        ADVector inertia_vec = parameters.tail(3 * 3);
        Eigen::Map<Mat3<ADScalar>> inertia(inertia_vec.data(), 3, 3);
        return inertia;
    }

    /**
     * Forward dynamics: compute acceleration from current state and force
     * input.
     */
    ADVector function(const ADVector& input,
                      const ADVector& parameters) const override {
        // Parameters are mass and inertia matrix of the body
        ADScalar mass = get_mass(parameters);
        Mat3<ADScalar> inertia = get_inertia(parameters);

        // Input is (state, system input) = (x, u)
        ADVector x = input.head(STATE_DIM);
        ADVector u = input.tail(INPUT_DIM);

        RigidBody<ADScalar> body(mass, inertia);
        return body.forward_dynamics(x, u);
    }
};
