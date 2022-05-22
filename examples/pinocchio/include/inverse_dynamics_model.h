#pragma once

#include <iostream>

// This has to be included before other CppAD-related headers. It defines some
// things required for auto-diff types to work with Pinocchio.
#include <pinocchio/codegen/cppadcg.hpp>

#include <CppADCodeGenEigenPy/ADModel.h>
#include <Eigen/Eigen>

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>

namespace ad = CppADCodeGenEigenPy;

template <typename Scalar>
struct InverseDynamicsModel : public ad::ADModel<Scalar> {
    using typename ad::ADModel<Scalar>::ADScalar;
    using typename ad::ADModel<Scalar>::ADVector;

    // Template Pinocchio types for auto-diff.
    using Model =
        pinocchio::ModelTpl<ADScalar, 0, pinocchio::JointCollectionDefaultTpl>;
    using Data =
        pinocchio::DataTpl<ADScalar, 0, pinocchio::JointCollectionDefaultTpl>;

    InverseDynamicsModel(const pinocchio::Model& model)
        : model_(model.template cast<ADScalar>()),
          ad::ADModel<Scalar>() {
    }

    // Generate the input used when differentiating the function
    ADVector input() const override {
        ADVector input(model_.nq + 2 * model_.nv);
        input << pinocchio::neutral(model_), ADVector::Zero(2 * model_.nv);
        return input;
    }

    /**
     * Inverse dynamics: compute torques that achieve desired acceleration
     * given position and velocity.
     */
    ADVector function(const ADVector& input) const override {
        Data data(model_);

        ADVector q = input.head(model_.nq);
        ADVector v = input.segment(model_.nq, model_.nv);
        ADVector a = input.tail(model_.nv);

        return pinocchio::rnea(model_, data, q, v, a);
    }

    Model model_;
};
