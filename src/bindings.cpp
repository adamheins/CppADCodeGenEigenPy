#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <Eigen/Eigen>

#include "CppADCodeGenEigenPy/ADFunction.h"

namespace py = pybind11;
namespace ad = CppADCodeGenEigenPy;

PYBIND11_MODULE(CppADCodeGenEigenPy, m) {
    using Scalar = double;
    using Vector = ad::ADFunction<Scalar>::Vector;
    using Matrix = ad::ADFunction<Scalar>::Matrix;

    m.doc() = "Bindings for code auto-differentiated using CppADCodeGen.";

    // TODO I've hardcoded double into this for now---not sure if I can
    // actually get away from this...
    py::class_<ad::ADFunction<Scalar>>(m, "ADFunction")
        .def(py::init<const std::string&, const std::string&>())
        .def("evaluate", static_cast<Vector (ad::ADFunction<Scalar>::*)(
                             const Eigen::Ref<const Vector>&) const>(
                             &ad::ADFunction<Scalar>::evaluate),
             "Evaluate function with no parameters.")
        .def("evaluate", static_cast<Vector (ad::ADFunction<Scalar>::*)(
                             const Eigen::Ref<const Vector>&,
                             const Eigen::Ref<const Vector>&) const>(
                             &ad::ADFunction<Scalar>::evaluate),
             "Evaluate function with parameters.")
        .def("jacobian", static_cast<Matrix (ad::ADFunction<Scalar>::*)(
                             const Eigen::Ref<const Vector>&) const>(
                             &ad::ADFunction<Scalar>::jacobian),
             "Evaluate Jacobian with no parameters.")
        .def("jacobian", static_cast<Matrix (ad::ADFunction<Scalar>::*)(
                             const Eigen::Ref<const Vector>&,
                             const Eigen::Ref<const Vector>&) const>(
                             &ad::ADFunction<Scalar>::jacobian),
             "Evaluate Jacobian with parameters.")
        .def("hessian", static_cast<Matrix (ad::ADFunction<Scalar>::*)(
                            const Eigen::Ref<const Vector>&, size_t) const>(
                            &ad::ADFunction<Scalar>::hessian),
             "Evaluate Hessian with no parameters.")
        .def("hessian", static_cast<Matrix (ad::ADFunction<Scalar>::*)(
                            const Eigen::Ref<const Vector>&,
                            const Eigen::Ref<const Vector>&, size_t) const>(
                            &ad::ADFunction<Scalar>::hessian),
             "Evaluate Hessian with parameters.")
        .def_property_readonly("input_size",
                               &ad::ADFunction<Scalar>::get_input_size)
        .def_property_readonly("output_size",
                               &ad::ADFunction<Scalar>::get_output_size);
}
