// #include <Eigen/Eigen>
// #include <boost/filesystem.hpp>
// #include <cppad/cg.hpp>
// #include <cppad/example/cppad_eigen.hpp>
//
// #include "CppADCodeGenEigenPy/ADFunction.h"

#define BEGIN_AD_MODEL(Model, Scalar)              \
    struct Model : ADModel<Scalar> {               \
        Model(size_t input_dim, size_t output_dim) \
            : ADModel("Model", input_dim, output_dim){};

#define END_AD_MODEL };
