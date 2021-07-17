#include <Eigen/Eigen>
#include <boost/filesystem.hpp>
#include <cppad/cg.hpp>
#include <cppad/example/cppad_eigen.hpp>

#include "CppADCodeGenEigenPy/ADFunction.h"


// Compiles the model
template <typename Scalar>
class ADModel {
   public:
    using ADBase = CppAD::cg::CG<Scalar>;
    using ADScalar = CppAD::AD<ADBase>;
    using ADVector = Eigen::Matrix<ADScalar, Eigen::Dynamic, 1>;

    // Matrix needs to be row major to interface with numpy
    using ADMatrix = Eigen::Matrix<ADScalar, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>;

    ADModel(std::string model_name, size_t input_dim, size_t output_dim,
            std::string folder_name = "/tmp/cppad")
        : model_name(model_name),
          input_dim(input_dim),
          output_dim(output_dim),
          folder_name(folder_name) {
        library_name = folder_name + "/lib" + model_name;
    }

    ~ADModel() = default;

    bool library_exists() {
        return boost::filesystem::exists(
            library_name +
            CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION);
    }

    // TODO
    ADFunction<Scalar> load(bool verbose = false) {
        return ADFunction<Scalar>(model_name, library_name, input_dim,
                                  output_dim);
    }

    ADFunction<Scalar> compile(bool verbose = false,
                               std::vector<std::string> compile_flags = {
                                   "-O3", "-march=native", "-mtune=native",
                                   "-ffast-math"}) {
        boost::filesystem::create_directories(folder_name);

        ADVector x = input();
        CppAD::Independent(x);

        // Apply the model function to get output
        ADVector y = function(x);

        // Record the relationship for AD
        CppAD::ADFun<ADBase> ad_func(x, y);

        // Optimize the operation sequence
        ad_func.optimize();

        // generates source code
        // TODO optionally support Hessian
        // TODO support sparse Jacobian/Hessian
        CppAD::cg::ModelCSourceGen<Scalar> source_gen(ad_func, model_name);
        source_gen.setCreateJacobian(true);

        // Compiler objects, compile to temporary shared library file to avoid
        // interference between processes
        CppAD::cg::ModelLibraryCSourceGen<Scalar> lib_source_gen(source_gen);
        CppAD::cg::GccCompiler<Scalar> compiler;
        CppAD::cg::DynamicModelLibraryProcessor<Scalar> lib_processor(
            lib_source_gen, library_name);

        compiler.setCompileLibFlags(compile_flags);
        compiler.addCompileLibFlag("-shared");
        compiler.addCompileLibFlag("-rdynamic");

        // Compile the library
        std::unique_ptr<CppAD::cg::DynamicLib<Scalar>> lib =
            lib_processor.createDynamicLibrary(compiler);
        std::cout << "Compiled library to " << library_name << std::endl;
        return ADFunction<Scalar>(model_name, std::move(lib), input_dim,
                                  output_dim);
    }

   protected:
    virtual ADVector function(const ADVector& x) = 0;

    virtual ADVector input() { return ADVector::Ones(input_dim); }

    size_t input_dim;
    size_t output_dim;

    std::string model_name;
    std::string folder_name;
    std::string library_name;  // NOTE: does not include the extension
};

#define BEGIN_AD_MODEL(Model, Scalar)              \
    struct Model : ADModel<Scalar> {               \
        Model(size_t input_dim, size_t output_dim) \
            : ADModel("Model", input_dim, output_dim){};

#define END_AD_MODEL };
