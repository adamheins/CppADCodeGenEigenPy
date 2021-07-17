#include <Eigen/Eigen>
#include <cppad/cg.hpp>
#include <cppad/example/cppad_eigen.hpp>

template <typename Scalar>
class ADFunction {
   public:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    ADFunction(const std::string& model_name, const std::string& library_name,
               size_t input_dim, size_t output_dim)
        : input_dim(input_dim), output_dim(output_dim) {
        lib.reset(new CppAD::cg::LinuxDynamicLib<Scalar>(
            library_name +
            CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION));
        model = lib->model(model_name);
    }

    ADFunction(const std::string& model_name,
               std::unique_ptr<CppAD::cg::DynamicLib<Scalar>> lib,
               size_t input_dim, size_t output_dim)
        : input_dim(input_dim), output_dim(output_dim), lib(std::move(lib)) {
        model = this->lib->model(model_name);
    }

    Vector evaluate(const Vector& x) {
        Vector y(output_dim);
        model->ForwardZero(x, y);
        return y;
    }

    Matrix jacobian(const Vector& x) {
        Vector J_vec = model->Jacobian(x);
        Eigen::Map<Matrix> J(J_vec.data(), output_dim, input_dim);
        assert(J.allFinite());
        return J;
    }

   protected:
    std::unique_ptr<CppAD::cg::DynamicLib<Scalar>> lib;
    std::unique_ptr<CppAD::cg::GenericModel<Scalar>> model;

    // TODO may template these at some point, for speed
    size_t input_dim;  // TODO not really required
    size_t output_dim;
};
