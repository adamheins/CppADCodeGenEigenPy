#include <Eigen/Eigen>
#include <cppad/cg.hpp>
#include <cppad/example/cppad_eigen.hpp>
#include <iostream>
#include <typeinfo>

template <typename Scalar>
class ADFunction {
   public:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    ADFunction(const std::string& model_name, const std::string& library_name) {
        lib.reset(new CppAD::cg::LinuxDynamicLib<Scalar>(
            library_name +
            CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION));
        model = lib->model(model_name);
    }

    Vector evaluate(const Eigen::Ref<const Vector>& input) {
        // Warn if input size is too small, which may mean the user should have
        // passed parameters.
        if (input.size() < model->Domain()) {
            std::cerr << "Model domain is " << model->Domain()
                      << ", but input is of size " << input.size()
                      << ". Did you mean to pass parameters, too?" << std::endl;
        }
        assert(input.rows() == model->Domain());

        // We need to explicitly tell the compiler what the template parameter
        // is (Vector), since otherwise it will deduce it as Eigen::Ref, which
        // won't work. See <https://stackoverflow.com/a/3505738/5145874>.
        return model->template ForwardZero<Vector>(input);
    }

    Vector evaluate(const Eigen::Ref<const Vector>& input,
                    const Eigen::Ref<const Vector>& parameters) {
        Vector xp(input.size() + parameters.size());
        xp << input, parameters;

        // Warn if input + parameters vector is too large, which may mean the
        // user shouldn't have passed parameters
        if (xp.size() > model->Domain()) {
            std::cerr << "Input size is " << input.size()
                      << " and parameter size is " << parameters.size()
                      << ". The total is " << xp.size()
                      << ", which larger than the model domain "
                      << model->Domain()
                      << ". Maybe you meant not to pass parameters?"
                      << std::endl;
        }
        return evaluate(xp);
    }

    Matrix jacobian(const Eigen::Ref<const Vector>& input) {
        assert(input.rows() == model->Domain());
        Vector J_vec = model->template Jacobian<Vector>(input);
        Eigen::Map<Matrix> J(J_vec.data(), model->Range(), model->Domain());
        assert(J.allFinite());
        return J;
    }

    Matrix jacobian(const Eigen::Ref<const Vector>& input,
                    const Eigen::Ref<const Vector>& parameters) {
        Vector xp(input.size() + parameters.size());
        xp << input, parameters;
        return jacobian(xp).leftCols(input.rows());
    }

   protected:
    std::unique_ptr<CppAD::cg::DynamicLib<Scalar>> lib;
    std::unique_ptr<CppAD::cg::GenericModel<Scalar>> model;
};
