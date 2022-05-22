#pragma once

template <typename Scalar>
CompiledModel<Scalar> ADModel<Scalar>::compile(
    const std::string& model_name, const std::string& directory_path,
    DerivativeOrder order, bool verbose, bool save_sources,
    std::vector<std::string> compile_flags) const {
    ADVector x = input();
    ADVector p = parameters();
    ADVector xp(x.rows() + p.rows());
    xp << x, p;

    CppAD::Independent(xp);

    // Apply the model function to get output
    ADVector y = function(xp.head(x.rows()), xp.tail(p.rows()));

    // Record the relationship for AD
    // It is more efficient to do:
    //   ADFun f;
    //   f.Dependent(x, y);
    //   f.optimize();
    // rather than
    //   ADFun f(x, y);
    //   f.optimize();
    // see <https://coin-or.github.io/CppAD/doc/optimize.htm>
    CppAD::ADFun<ADScalarBase> ad_func;
    ad_func.Dependent(xp, y);

    // Optimize the operation sequence
    ad_func.optimize();

    // Generate source code
    // TODO support sparse Jacobian/Hessian
    CppAD::cg::ModelCSourceGen<Scalar> source_gen(ad_func, model_name);
    if (order >= DerivativeOrder::First) {
        source_gen.setCreateJacobian(true);
    }
    if (order >= DerivativeOrder::Second) {
        source_gen.setCreateHessian(true);
    }

    const std::string lib_generic_path =
        get_library_generic_path(model_name, directory_path);
    CppAD::cg::ModelLibraryCSourceGen<Scalar> lib_source_gen(source_gen);
    CppAD::cg::GccCompiler<Scalar> compiler;
    CppAD::cg::DynamicModelLibraryProcessor<Scalar> lib_processor(
        lib_source_gen, lib_generic_path);

    if (save_sources) {
        CppAD::cg::SaveFilesModelLibraryProcessor<Scalar> lib_source_saver(
            lib_source_gen);
        lib_source_saver.saveSources();
    }

    compiler.setCompileLibFlags(compile_flags);
    compiler.addCompileLibFlag("-shared");
    compiler.addCompileLibFlag("-rdynamic");

    // Compile the library
    std::unique_ptr<CppAD::cg::DynamicLib<Scalar>> lib =
        lib_processor.createDynamicLibrary(compiler);
    if (verbose) {
        std::cout << "Compiled library for model " << model_name << " to "
                  << get_library_real_path(model_name, directory_path)
                  << std::endl;
    }
    return CompiledModel<Scalar>(model_name, lib_generic_path);
}

template <typename Scalar>
typename ADModel<Scalar>::ADVector ADModel<Scalar>::function(
    const ADVector& x) const {
    // Default is a single 0.
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
    // Default is an empty parameter vector.
    return ADVector(0);
}
