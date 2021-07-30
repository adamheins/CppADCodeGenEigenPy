#include <cppad/cg.hpp>
#include <cppad/example/cppad_eigen.hpp>

// using ADBase = CppAD::cg::CG<Scalar>;
// using ADScalar = CppAD::AD<ADBase>;

template <typename Scalar, typename Derived>
CppAD::AD<CppAD::cg::CG<Scalar>> operator*(
    const MatrixBase<Derived>& lhs,
    const CppAD::AD<CppAD::cg::CG<Scalar>>& rhs) {

    return lhs * + rhs;
}
