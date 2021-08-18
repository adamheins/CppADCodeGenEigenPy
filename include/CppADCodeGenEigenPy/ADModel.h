#pragma once

#include <Eigen/Eigen>
#include <cppad/cg.hpp>
#include <iostream>

#include <CppADCodeGenEigenPy/CompiledModel.h>
#include <CppADCodeGenEigenPy/Util.h>

namespace CppADCodeGenEigenPy {

/** Derivative order */
enum class DerivativeOrder { Zero, First, Second };

/** Abstract base class for a function to be auto-differentiated and then
 *  compiled.
 *
 * This class should be subclassed to define the function and its inputs and
 * (optional) parameters. It can then be compiled to a dynamic library and
 * loaded for use with the `Model` class.
 *
 * @tparam Scalar     The scalar type to use. Typically float or double.
 */
template <typename Scalar>
class ADModel {
   public:
    using ADScalarBase = CppAD::cg::CG<Scalar>;

    /** Auto-diff scalar type. */
    using ADScalar = CppAD::AD<ADScalarBase>;

    /** Auto-diff dynamic vector type. */
    using ADVector = Eigen::Matrix<ADScalar, Eigen::Dynamic, 1>;

    /** Auto-diff dynamic matrix type. */
    // Needs to be row major to interface with numpy
    using ADMatrix = Eigen::Matrix<ADScalar, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>;

    /** Constructor. */
    ADModel() {}

    /** Compile the library into a dynamic library.
     *
     * @param[in] model_name     Name of the compiled model.
     * @param[in] directory_path Path of directory where model library should
     *                           go. The direcotory must exist; it will not be
     *                           created automatically.
     * @param[in] verbose        Print additional information.
     * @param[in] compile_flags  Flags to pass to the compiler to compile the
     *                           library.
     *
     * @returns The compiled model.
     */
    CompiledModel<Scalar> compile(
        const std::string& model_name, const std::string& directory_path,
        DerivativeOrder order = DerivativeOrder::Second, bool verbose = false,
        std::vector<std::string> compile_flags = {
            "-O3", "-march=native", "-mtune=native", "-ffast-math"}) const;

   protected:
    /** Defines this model's (parameterless) function.
     *
     * This should be overridden by derived classes implementing a model with
     * no parameters. Only one of the overloads of `function` should be
     * overridden by a single derived class.
     *
     * @param[in] x  The input to the function.
     *
     * @returns The function output.
     */
    virtual ADVector function(const ADVector& x) const;

    /** Defines this model's (parameterized) function.
     *
     * This should be overridden by derived classes implementing a model with
     * parameters. Parameters differ from inputs only in that derivatives are
     * not taken with respect to parameters. Only one of the overloads of
     * `function` should be overridden by a single derived class.
     *
     * @param[in] x  The input to the function.
     * @param[in] p  The function parameters.
     *
     * @returns The function output.
     */
    virtual ADVector function(const ADVector& x, const ADVector& p) const;

    /** Generates the input that should be used when recording the operation
     *  sequence for auto-differentiation. Also defines the shape of the input.
     *  Must be defined by the derived class.
     *
     * When in doubt, a good choice is often to generate a vector of ones. A
     * vector of zeros is often more liable to produce divisions by zero when
     * differentiating, but it depends on the particular function.
     *
     * @returns The input vector to use for auto-differentiation.
     */
    virtual ADVector input() const = 0;

    /** Generates the parameter vector that should be used when recording the
     *  operation sequence for auto-differentiation. Also defines the shape of
     *  the parameter vector. Need only be overridden by the derived class if
     *  the modelled function takes parameters.
     *
     * @returns The parameter vector to use for auto-differentiation
     */
    virtual ADVector parameters() const;
};  // class ADModel

#include "impl/ADModel.tpp"

}  // namespace CppADCodeGenEigenPy
