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
        // We need to explicitly tell the compiler what the template parameter
        // is (Vector), since otherwise it will deduce it as Eigen::Ref, which
        // won't work. See <https://stackoverflow.com/a/3505738/5145874>.
        if (input.rows() != model->Domain()) {
            assert(false);
            std::cout << "Model domain is " << model->Domain()
                      << ", but input has " << input.rows() << " rows."
                      << std::endl;
        }
        // assert(input.rows() ==
        //        model->Domain());  // TODO should we have such checks?
        return model->template ForwardZero<Vector>(input);
    }

    Vector evaluate(const Eigen::Ref<const Vector>& input,
                    const Eigen::Ref<const Vector>& parameters) {
        Vector xp(input.size() + parameters.size());
        xp << input, parameters;
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
