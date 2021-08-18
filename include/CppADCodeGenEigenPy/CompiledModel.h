#pragma once

#include <Eigen/Eigen>
#include <cppad/cg.hpp>
#include <string>

#include <CppADCodeGenEigenPy/Util.h>

namespace CppADCodeGenEigenPy {

/** A CompiledModel wraps a dynamic library that has been compiled from derived
 * class
 *  of ADModel. It provides methods for evaluating the function and available
 *  derivatives (Jacobian, Hessian).
 *
 * @tparam Scalar  The scalar type to use. Typically float or double.
 */
template <typename Scalar>
class CompiledModel {
   public:
    /** Dynamic vector type. */
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    /** Dynamic matrix type. */
    using Matrix =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    /** Constructor.
     *
     * @param[in] model_name  The name of the model being loaded from the
     *                        dynamic library.
     * @param[in] library_generic_path  The full path to the dynamic library,
     *                                  without the file extension.
     */
    CompiledModel(const std::string& model_name,
                  const std::string& library_generic_path);

    // ~CompiledModel() = default;

    /** Evaluate the function. This overload should be called if the modelled
     *  function has no parameters.
     *
     * @param[in] input  The input at which to evaluate the function.
     *
     * @throws std::runtime_error if the provided input size does not match
     * that of the model.
     *
     * @returns The function output.
     */
    Vector evaluate(const Eigen::Ref<const Vector>& input) const;

    /** Evaluate the function. This overload should be called if the modelled
     *  function has parameters.
     *
     * @param[in] input       The input at which to evaluate the function.
     * @param[in] parameters  The parameters for the function.
     *
     * @throws std::runtime_error if the combined size of the provided input
     * and parameters does not match the input size of the model.
     *
     * @returns The function output.
     */
    Vector evaluate(const Eigen::Ref<const Vector>& input,
                    const Eigen::Ref<const Vector>& parameters) const;

    /** Compute the function's Jacobian (first-order derivative). This overload
     *  should be called if the modelled function has no parameters.
     *
     * @param[in] input  The input at which to evaluate the Jacobian.
     *
     * @throws std::runtime_error if the provided input size does not match
     * that of the model.
     * @throws std::runtime_error if the order of the model is not at least
     * one.
     *
     * @returns The Jacobian matrix.
     */
    Matrix jacobian(const Eigen::Ref<const Vector>& input) const;

    /** Compute the function's Jacobian (first-order derivative). This overload
     *  should be called if the modelled function has parameters.
     *
     * @param[in] input       The input at which to evaluate the Jacobian.
     * @param[in] parameters  The parameters for the function.
     *
     * @throws std::runtime_error if the combined size of the provided input
     * and parameters does not match the input size of the model.
     * @throws std::runtime_error if the order of the model is not at least
     * one.
     *
     * @returns The Jacobian matrix.
     */
    Matrix jacobian(const Eigen::Ref<const Vector>& input,
                    const Eigen::Ref<const Vector>& parameters) const;

    /** Compute the function's Hessian (second-order derivative) for a given
     *  output dimension. This overload should be called if the modelled
     *  function has no parameters.
     *
     * @param[in] input       The input at which to evaluate the Hessian.
     * @param[in] output_dim  The output dimension for which to evaluate the
     *                        Hessian.
     *
     * @throws std::runtime_error if the provided input size does not match
     * that of the model.
     * @throws std::runtime_error if the order of the model is not at least
     * two.
     *
     * @returns The Hessian matrix.
     */
    Matrix hessian(const Eigen::Ref<const Vector>& input,
                   size_t output_dim = 0) const;

    /** Compute the function's Hessian (second-order derivative) for a given
     *  output dimension. This overload should be called if the modelled
     *  function has parameters.
     *
     * @param[in] input       The input at which to evaluate the Hessian.
     * @param[in] parameters  The parameters for the function.
     * @param[in] output_dim  The output dimension for which to evaluate the
     *                        Hessian.
     *
     * @throws std::runtime_error if the combined size of the provided input
     * and parameters does not match the input size of the model.
     * @throws std::runtime_error if the order of the model is not at least
     * two.
     *
     * @returns The Hessian matrix.
     */
    Matrix hessian(const Eigen::Ref<const Vector>& input,
                   const Eigen::Ref<const Vector>& parameters,
                   size_t output_dim = 0) const;

    /** Get the input size of the model. Note that this includes both normal
     *  inputs and parameters.
     *
     * @returns The combined size of the model input and parameters.
     */
    size_t get_input_size() const;

    /** Get the output size of the model.
     *
     * @returns The size of the model output.
     */
    size_t get_output_size() const;

   private:
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
};  // class CompiledModel

#include "impl/CompiledModel.tpp"

}  // namespace CppADCodeGenEigenPy
