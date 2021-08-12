#pragma once

#include <Eigen/Eigen>
#include <cppad/cg.hpp>
#include <string>

namespace CppADCodeGenEigenPy {

template <typename Scalar>
class Model {
   public:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    Model(const std::string& model_name,
          const std::string& library_generic_path);

    Vector evaluate(const Eigen::Ref<const Vector>& input) const;

    Vector evaluate(const Eigen::Ref<const Vector>& input,
                    const Eigen::Ref<const Vector>& parameters) const;

    Matrix jacobian(const Eigen::Ref<const Vector>& input) const;

    Matrix jacobian(const Eigen::Ref<const Vector>& input,
                    const Eigen::Ref<const Vector>& parameters) const;

    Matrix hessian(const Eigen::Ref<const Vector>& input,
                   size_t output_dim = 0) const;

    Matrix hessian(const Eigen::Ref<const Vector>& input,
                   const Eigen::Ref<const Vector>& parameters,
                   size_t output_dim = 0) const;

    size_t get_input_size() const;

    size_t get_output_size() const;

   protected:
    std::unique_ptr<CppAD::cg::DynamicLib<Scalar>> lib_;
    std::unique_ptr<CppAD::cg::GenericModel<Scalar>> model_;

    size_t input_size_;
    size_t output_size_;

    // Error if input size is wrong. If it is too small,the user may have
    // meant to pass parameters as well.
    void check_input_size(size_t size) const;

    // Error if input + parameters vector is too large, which may mean the
    // user shouldn't have passed parameters
    void check_input_size_with_params(size_t input_size,
                                      size_t param_size) const;
};  // class Model

#include "impl/Model.tpp"

}  // namespace CppADCodeGenEigenPy
