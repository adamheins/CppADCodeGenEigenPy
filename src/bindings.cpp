#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Eigen>

namespace py = pybind11;

// using VectorX = Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::RowMajor>;

Eigen::VectorXd vec_func(const Eigen::Ref<const Eigen::VectorXd> x) {
    return 2 * x;
}


double add(double a, double b) {
    return a + b;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
    m.def("vec_func", &vec_func, "Vector function", py::return_value_policy::reference_internal);
}
