#pragma once

#include <Eigen/Eigen>
#include <boost/filesystem.hpp>
#include <cppad/cg.hpp>
#include <cppad/example/cppad_eigen.hpp>

#include <iostream>

namespace CppADCodeGenEigenPy {

// Derivative order
enum class ADOrder { Zero, First, Second };

// Compiles the model
template <typename Scalar>
class ADModel {
   public:
    using ADBase = CppAD::cg::CG<Scalar>;
    using ADScalar = CppAD::AD<ADBase>;
    using ADVector = Eigen::Matrix<ADScalar, Eigen::Dynamic, 1>;

    // Matrix needs to be row major to interface with numpy
    using ADMatrix = Eigen::Matrix<ADScalar, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>;

    ADModel(std::string model_name, std::string folder_name,
            ADOrder order = ADOrder::First);

    ~ADModel() = default;

    bool library_exists() const;

    std::string get_library_real_path() const;

    std::string get_library_generic_path() const;

    std::string get_model_name() const;

    void compile(bool verbose = false,
                 std::vector<std::string> compile_flags = {
                     "-O3", "-march=native", "-mtune=native",
                     "-ffast-math"}) const;

   protected:
    virtual ADVector function(const ADVector& x) const;

    virtual ADVector function(const ADVector& x, const ADVector& p) const;

    virtual ADVector input() const = 0;

    virtual ADVector parameters() const;

    std::string model_name_;
    std::string folder_name_;
    std::string library_generic_path_;
    ADOrder order_;
};  // class ADModel

#include "impl/ADModel.tpp"

}  // namespace CppADCodeGenEigenPy
