#pragma once

template <typename Scalar, size_t InputDim, size_t OutputDim, size_t ParamDim>
CompiledModel<Scalar> ADModel<Scalar, InputDim, OutputDim, ParamDim>::compile(
    const std::string& model_name, const std::string& directory_path,
    DerivativeOrder order, bool verbose,
    std::vector<std::string> compile_flags) const {

    Eigen::Matrix<ADScalar, InputDim + ParamDim, 1> xp;
    ADInput x_test = input();
    ADParameters p_test = parameters();
    xp << x_test, p_test;

    std::cout << "one" << std::endl;
    std::cout << x_test.rows() << std::endl;
    std::cout << p_test.rows() << std::endl;
    std::cout << xp.rows() << std::endl;

    ADVector xp_dynamic = Eigen::Map<ADVector>(xp.data(), xp.rows(), xp.cols());
    CppAD::Independent(xp_dynamic);

    std::cout << "two" << std::endl;

    // Apply the model function to get output
    ADInput x = xp.template head<InputDim>();
    ADParameters p = xp.template tail<ParamDim>();
    ADOutput y = function(x, p);

    std::cout << "three" << std::endl;

    // Record the relationship for AD
    CppAD::ADFun<ADScalarBase> ad_func(xp, y);

    std::cout << "four" << std::endl;

    // Optimize the operation sequence
    ad_func.optimize();

    // generates source code
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

template <typename Scalar, size_t InputDim, size_t OutputDim, size_t ParamDim>
typename ADModel<Scalar, InputDim, OutputDim, ParamDim>::ADOutput ADModel<
    Scalar, InputDim, OutputDim, ParamDim>::function(const ADInput& x) const {
    return ADOutput::Zero();
}

template <typename Scalar, size_t InputDim, size_t OutputDim, size_t ParamDim>
typename ADModel<Scalar, InputDim, OutputDim, ParamDim>::ADOutput
ADModel<Scalar, InputDim, OutputDim, ParamDim>::function(
    const ADInput& x, const ADParameters& p) const {
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

template <typename Scalar, size_t InputDim, size_t OutputDim, size_t ParamDim>
typename ADModel<Scalar, InputDim, OutputDim, ParamDim>::ADParameters
ADModel<Scalar, InputDim, OutputDim, ParamDim>::parameters() const {
    // TODO this is now problematic because it won't compile if parameters is
    // dynamic size---how does eigen handle this?
    // TODO we may just have to go fully fixed size... I guess this would be
    // fine; it makes things more concrete. The other option would be to always
    // just return some known size thing (like zero-length), but that violates
    // the idea of having fixed-size matrices
    return ADParameters::Ones();
}
