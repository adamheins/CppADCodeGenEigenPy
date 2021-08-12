#pragma once

#include <Eigen/Eigen>
#include <boost/filesystem.hpp>
#include <cppad/cg.hpp>

#include <iostream>

namespace CppADCodeGenEigenPy {

/** Derivative order */
enum class ADOrder { Zero, First, Second };

/** Abstract base class for a function to be auto-differentiated and compiled.
 *
 * This class should be subclassed to define the function and its inputs and
 * (optional) parameters. It can then be compiled to a dynamic library and
 * loaded for use with the `Model` class.
 *
 * @tparam Scalar  The scalar type to use. Typically float or double.
 */
template <typename Scalar>
class ADModel {
   public:
    using ADBase = CppAD::cg::CG<Scalar>;

    /** Auto-diff scalar type. */
    using ADScalar = CppAD::AD<ADBase>;

    /** Auto-diff vector type. */
    using ADVector = Eigen::Matrix<ADScalar, Eigen::Dynamic, 1>;

    /** Auto-diff matrix type. */
    // Needs to be row major to interface with numpy
    using ADMatrix = Eigen::Matrix<ADScalar, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>;

    /** Constructor.
     *
     * @param[in] model_name      The name of this model.
     * @param[in] directory_name  The directory path where the dynamic library
     *                            should go once compiled.
     * @param[in] order           Order of derivatives to be taken. Zero
     *                            indicates no derivatives are computed.
     */
    ADModel(std::string model_name, std::string directory_name,
            ADOrder order = ADOrder::First);

    /** Check if the dynamic library currently exists at the directory path
     *  given in the constructor.
     *
     * @returns True if the library exists, false otherwise.
     */
    bool library_exists() const;

    /** Get the full path of the dynamic library (even if it doesn't currently
     *  exist), including the (platform-dependent) extension.
     *
     * @returns The library path including the extension.
     */
    std::string get_library_real_path() const;

    /** Get the full path of the dynamic library (even if it doesn't currently
     *  exist), excluding the extension.
     *
     * @returns The library path excluding the extension.
     */
    std::string get_library_generic_path() const;

    /** Get the name of the model.
     *
     * @returns The name of the model.
     */
    std::string get_model_name() const;

    /** Compile the library into a dynamic library.
     *
     * @param[in] verbose        Print additional information.
     * @param[in] compile_flags  Flags to pass to the compiler to compile the
     *                           library.
     */
    void compile(bool verbose = false,
                 std::vector<std::string> compile_flags = {
                     "-O3", "-march=native", "-mtune=native",
                     "-ffast-math"}) const;

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

   private:
    std::string model_name_;
    std::string directory_name_;
    std::string library_generic_path_;
    ADOrder order_;
};  // class ADModel

#include "impl/ADModel.tpp"

}  // namespace CppADCodeGenEigenPy
