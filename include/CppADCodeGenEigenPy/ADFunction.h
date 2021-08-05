#pragma once

#include <Eigen/Eigen>
#include <cppad/cg.hpp>
#include <cppad/example/cppad_eigen.hpp>
#include <iostream>
#include <string>
#include <typeinfo>

template <typename Scalar>
class ADFunction {
   public:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    ADFunction(const std::string& model_name,
               const std::string& library_generic_path) {
        lib_.reset(new CppAD::cg::LinuxDynamicLib<Scalar>(
            library_generic_path +
            CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION));
        model_ = lib_->model(model_name);
        input_size_ = model_->Domain();
        output_size_ = model_->Range();
    }

    Vector evaluate(const Eigen::Ref<const Vector>& input) const {
        check_input_size(input.size());

        // We need to explicitly tell the compiler what the template parameter
        // is (Vector), since otherwise it will deduce it as Eigen::Ref, which
        // won't work. See <https://stackoverflow.com/a/3505738/5145874>.
        return model_->template ForwardZero<Vector>(input);
    }

    Vector evaluate(const Eigen::Ref<const Vector>& input,
                    const Eigen::Ref<const Vector>& parameters) const {
        check_input_size_with_params(input.size(), parameters.size());

        Vector xp(input.size() + parameters.size());
        xp << input, parameters;
        return evaluate(xp);
    }

    Matrix jacobian(const Eigen::Ref<const Vector>& input) const {
        if (!model_->isJacobianAvailable()) {
            throw std::runtime_error(
                "Jacobian is not available: compiled model must be at least "
                "first-order.");
        }
        check_input_size(input.size());

        assert(input.rows() == input_size_);
        Vector J_vec = model_->template Jacobian<Vector>(input);
        Eigen::Map<Matrix> J(J_vec.data(), output_size_, input_size_);
        assert(J.allFinite());
        return J;
    }

    Matrix jacobian(const Eigen::Ref<const Vector>& input,
                    const Eigen::Ref<const Vector>& parameters) const {
        check_input_size_with_params(input.size(), parameters.size());

        Vector xp(input.size() + parameters.size());
        xp << input, parameters;
        return jacobian(xp).leftCols(input.rows());
    }

    Matrix hessian(const Eigen::Ref<const Vector>& input,
                   size_t output_dim) const {
        if (!model_->isHessianAvailable()) {
            throw std::runtime_error(
                "Hessian is not available: compiled model must be "
                "second-order.");
        }
        check_input_size(input.size());

        Vector H_vec = model_->template Hessian<Vector>(input, output_dim);
        Eigen::Map<Matrix> H(H_vec.data(), input_size_, input_size_);
        assert(H.allFinite());
        return H;
    }

    Matrix hessian(const Eigen::Ref<const Vector>& input) const {
        return hessian(input, 0);
    }

    Matrix hessian(const Eigen::Ref<const Vector>& input,
                   const Eigen::Ref<const Vector>& parameters,
                   size_t output_dim) const {
        check_input_size_with_params(input.size(), parameters.size());

        Vector xp(input.size() + parameters.size());
        xp << input, parameters;
        return hessian(xp, output_dim).leftCols(input.rows());
    }

    Matrix hessian(const Eigen::Ref<const Vector>& input,
                   const Eigen::Ref<const Vector>& parameters) const {
        return hessian(input, parameters, 0);
    }

    size_t get_input_size() const { return input_size_; }

    size_t get_output_size() const { return output_size_; }

   protected:
    std::unique_ptr<CppAD::cg::DynamicLib<Scalar>> lib_;
    std::unique_ptr<CppAD::cg::GenericModel<Scalar>> model_;

    size_t input_size_;
    size_t output_size_;

    // Error if input size is wrong. If it is too small,the user may have
    // meant to pass parameters as well.
    void check_input_size(size_t size) const {
        if (size < input_size_) {
            throw std::runtime_error(
                "Model domain is " + std::to_string(input_size_) +
                ", but input is of size " + std::to_string(size) +
                ". Did you mean to pass parameters, too?");
        } else if (size > input_size_) {
            throw std::runtime_error(
                "Model domain is " + std::to_string(input_size_) +
                ", but input is of size " + std::to_string(size));
        }
    }

    // Error if input + parameters vector is too large, which may mean the
    // user shouldn't have passed parameters
    void check_input_size_with_params(size_t input_size,
                                      size_t param_size) const {
        size_t total_size = input_size + param_size;
        if (total_size > input_size_) {
            throw std::runtime_error(
                "Input size is " + std::to_string(input_size) +
                " and parameter size is " + std::to_string(param_size) +
                ". The total is " + std::to_string(total_size) +
                ", which larger than the model domain " +
                std::to_string(input_size_) +
                ". Maybe you meant not to pass parameters?");
        }
    }
};
