#include <iostream>

#include <CppADCodeGenEigenPy/ADModel.h>

#include "example/Defs.h"

// template<typename Scalar>
// using ADScalar = CppAD::AD<CppAD::cg::CG<Scalar>>;
//
// template<typename Scalar>
// using ADMatrix =
//     Eigen::Matrix<ADScalar<Scalar>, Eigen::Dynamic, Eigen::Dynamic,
//     Eigen::RowMajor>;
//
// template<typename Scalar>
// ADMatrix<Scalar> operator*(const ADMatrix<Scalar>& lhs, const Scalar& rhs) {
//     return lhs * ADScalar<Scalar>(rhs);
// }

// Instantiate the template function
// template ADMatrix<double> operator*<double>(const ADMatrix<double>&, const
// double&);

// ADMatrix<double> operator*(const ADMatrix<double>& lhs, const double& rhs) {
//     return lhs * ADScalar<double>(rhs);
// }
//
// ADMatrix<double> operator*(const double& lhs, const ADMatrix<double>& rhs) {
//     return ADScalar<double>(lhs) * rhs;
// }

template <typename Scalar>
struct MyADModel : public ADModel<Scalar> {
    using typename ADModel<Scalar>::ADScalar;
    using typename ADModel<Scalar>::ADVector;

    MyADModel(const std::string& model_name, const std::string& folder_name)
        : ADModel<Scalar>(model_name, folder_name){};

    // Generate the input to the function
    ADVector input() override { return ADVector::Ones(NUM_INPUT); }

    // Evaluate the function
    ADVector function(const ADVector& input) override {
        ADVector output = input * ADScalar(2.);
        output(0) += input(1);
        return output;
    }
};

int main() {
    MyADModel<double> model(MODEL_NAME, FOLDER_NAME);
    model.compile();
}
