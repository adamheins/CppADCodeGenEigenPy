#pragma once

std::string make_library_generic_path(const std::string& directory_name,
                                      const std::string& model_name) {
    return directory_name + "/lib" + model_name;
}

std::string make_library_real_path(const std::string& directory_name,
                                   const std::string& model_name) {
    std::string ext = CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION;
    return make_library_generic_path(directory_name, model_name) + ext;
}

// TODO why don't I let the user handle this? because its not easy to get the
// path any more
// template <typename Scalar>
// void ADModel<Scalar>::compile_unless_exists(
//     const std::string& model_name, const std::string& directory_name,
//     bool verbose, std::vector<std::string> compile_flags) const {
//     if (!boost::filesystem::exists(make_library_real_path())) {
//         compile(model_name, directory_name, verbose, compile_flags);
//     }
//     if (verbose) {
//         std::cout << "Library already exists. Skipping compile." << std::endl;
//     }
// }

template <typename Scalar>
void ADModel<Scalar>::compile(const std::string& model_name,
                              const std::string& directory_name, bool verbose,
                              std::vector<std::string> compile_flags) const {
    boost::filesystem::create_directories(directory_name);

    ADVector x = input();
    ADVector p = parameters();
    ADVector xp(x.size() + p.size());
    xp << x, p;

    CppAD::Independent(xp);

    // Apply the model function to get output
    ADVector y = function(xp.head(x.size()), xp.tail(p.size()));

    // Record the relationship for AD
    CppAD::ADFun<ADBase> ad_func(xp, y);

    // Optimize the operation sequence
    ad_func.optimize();

    // generates source code
    // TODO support sparse Jacobian/Hessian
    CppAD::cg::ModelCSourceGen<Scalar> source_gen(ad_func, model_name);
    if (order_ >= DerivativeOrder::First) {
        source_gen.setCreateJacobian(true);
    }
    if (order_ >= DerivativeOrder::Second) {
        source_gen.setCreateHessian(true);
    }

    CppAD::cg::ModelLibraryCSourceGen<Scalar> lib_source_gen(source_gen);
    CppAD::cg::GccCompiler<Scalar> compiler;
    CppAD::cg::DynamicModelLibraryProcessor<Scalar> lib_processor(
        lib_source_gen, make_library_generic_path());

    compiler.setCompileLibFlags(compile_flags);
    compiler.addCompileLibFlag("-shared");
    compiler.addCompileLibFlag("-rdynamic");

    // Compile the library
    std::unique_ptr<CppAD::cg::DynamicLib<Scalar>> lib =
        lib_processor.createDynamicLibrary(compiler);
    if (verbose) {
        std::cout << "Compiled library for model " << model_name << " to "
                  << make_library_real_path() << std::endl;
    }
}

template <typename Scalar>
typename ADModel<Scalar>::ADVector ADModel<Scalar>::function(
    const ADVector& x) const {
    // Default function is a just a single 0.
    return ADVector::Zero(1);
}

template <typename Scalar>
typename ADModel<Scalar>::ADVector ADModel<Scalar>::function(
    const ADVector& x, const ADVector& p) const {
    // We only want user to have to override one of the "function" methods,
    // which means neither can be pure virtual. This is one downside: the
    // compiler will let you compile a model with a default function value.
    //
    // This overload will always be called by the other code in the class,
    // so either:
    // * this one is overridden, and this implementation does not matter
    // * the first is overridden, which is called by this
    return function(x);
}

template <typename Scalar>
typename ADModel<Scalar>::ADVector ADModel<Scalar>::parameters() const {
    // Default is to have no parameters (i.e., an empty vector).
    return ADVector(0);
}
