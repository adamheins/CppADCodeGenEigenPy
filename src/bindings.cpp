#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <Eigen/Eigen>

#include "CppADCodeGenEigenPy/ADFunction.h"

namespace py = pybind11;

// Basic Eigen binding example; can be called in Python with:
// import example
// example.vec_func(np.ndarray)
// if array is not of type float64 (double), it will be copied
Eigen::VectorXd vec_func(const Eigen::Ref<const Eigen::VectorXd> x) {
    return 2 * x;
}

double add(double a, double b) { return a + b; }

PYBIND11_MODULE(CppADCodeGenEigenPy, m) {
    using Scalar = double;
    using Vector = ADFunction<Scalar>::Vector;
    using Matrix = ADFunction<Scalar>::Matrix;

    m.doc() = "Bindings for code auto-differentiated using CppADCodeGen.";

    // m.def("add", &add, "A function which adds two numbers");
    // m.def("vec_func", &vec_func, "Vector function",
    // py::return_value_policy::reference_internal);

    // TODO I've hardcoded double into this for now
    py::class_<ADFunction<Scalar>>(m, "ADFunction")
        .def(py::init<const std::string&, const std::string&>())
        // .def("evaluate", &ADFunction<Scalar>::evaluate)
        // .def("jacobian", &ADFunction<double>::jacobian);
        .def("evaluate", static_cast<Vector (ADFunction<Scalar>::*)(
                             const Eigen::Ref<const Vector>&) const>(
                             &ADFunction<Scalar>::evaluate),
             "Evaluate function with no parameters.")
        .def("evaluate", static_cast<Vector (ADFunction<Scalar>::*)(
                             const Eigen::Ref<const Vector>&,
                             const Eigen::Ref<const Vector>&) const>(
                             &ADFunction<Scalar>::evaluate),
             "Evaluate function with parameters.")
        .def("jacobian", static_cast<Matrix (ADFunction<Scalar>::*)(
                             const Eigen::Ref<const Vector>&) const>(
                             &ADFunction<Scalar>::jacobian),
             "Evaluate Jacobian with no parameters.")
        .def("jacobian", static_cast<Matrix (ADFunction<Scalar>::*)(
                             const Eigen::Ref<const Vector>&,
                             const Eigen::Ref<const Vector>&) const>(
                             &ADFunction<Scalar>::jacobian),
             "Evaluate Jacobian with parameters.");
}
