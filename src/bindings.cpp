#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <Eigen/Eigen>

#include <CppADCodeGenEigenPy/CompiledModel.h>

namespace py = pybind11;
namespace ad = CppADCodeGenEigenPy;

PYBIND11_MODULE(CppADCodeGenEigenPy, m) {
    using Scalar = double;
    using Vector = ad::CompiledModel<Scalar>::Vector;
    using Matrix = ad::CompiledModel<Scalar>::Matrix;

    m.doc() = "Bindings for code auto-differentiated using CppADCodeGen.";

    // TODO I've hardcoded double into this for now---not sure if I can
    // actually get away from this...
    py::class_<ad::CompiledModel<Scalar>>(m, "CompiledModel")
        .def(py::init<const std::string&, const std::string&>())
        .def("evaluate", static_cast<Vector (ad::CompiledModel<Scalar>::*)(
                             const Eigen::Ref<const Vector>&) const>(
                             &ad::CompiledModel<Scalar>::evaluate),
             "Evaluate function with no parameters.")
        .def("evaluate", static_cast<Vector (ad::CompiledModel<Scalar>::*)(
                             const Eigen::Ref<const Vector>&,
                             const Eigen::Ref<const Vector>&) const>(
                             &ad::CompiledModel<Scalar>::evaluate),
             "Evaluate function with parameters.")
        .def("jacobian", static_cast<Matrix (ad::CompiledModel<Scalar>::*)(
                             const Eigen::Ref<const Vector>&) const>(
                             &ad::CompiledModel<Scalar>::jacobian),
             "Evaluate Jacobian with no parameters.")
        .def("jacobian", static_cast<Matrix (ad::CompiledModel<Scalar>::*)(
                             const Eigen::Ref<const Vector>&,
                             const Eigen::Ref<const Vector>&) const>(
                             &ad::CompiledModel<Scalar>::jacobian),
             "Evaluate Jacobian with parameters.")
        .def("hessian", static_cast<Matrix (ad::CompiledModel<Scalar>::*)(
                            const Eigen::Ref<const Vector>&, size_t) const>(
                            &ad::CompiledModel<Scalar>::hessian),
             "Evaluate Hessian with no parameters.")
        .def("hessian", static_cast<Matrix (ad::CompiledModel<Scalar>::*)(
                            const Eigen::Ref<const Vector>&,
                            const Eigen::Ref<const Vector>&, size_t) const>(
                            &ad::CompiledModel<Scalar>::hessian),
             "Evaluate Hessian with parameters.")
        .def_property_readonly("input_size",
                               &ad::CompiledModel<Scalar>::get_input_size)
        .def_property_readonly("output_size",
                               &ad::CompiledModel<Scalar>::get_output_size);
}
