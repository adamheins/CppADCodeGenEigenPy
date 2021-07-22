#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

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


double add(double a, double b) {
    return a + b;
}

PYBIND11_MODULE(CppADCodeGenEigenPy, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
    m.def("vec_func", &vec_func, "Vector function", py::return_value_policy::reference_internal);

    // TODO I've hardcoded double into this for now
    py::class_<ADFunction<double>>(m, "ADFunction")
        .def(py::init<const std::string&, const std::string&>())
        .def("evaluate", &ADFunction<double>::evaluate)
        .def("jacobian", &ADFunction<double>::jacobian);
}
