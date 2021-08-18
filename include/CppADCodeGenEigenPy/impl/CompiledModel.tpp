#pragma once

template <typename Scalar>
CompiledModel<Scalar>::CompiledModel(const std::string& model_name,
                                     const std::string& library_generic_path)
    : lib_(new CppAD::cg::LinuxDynamicLib<Scalar>(
          library_generic_path +
          CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION)) {
    // TODO we could do without the lib_ entirely, I think...
    // lib_.reset(new CppAD::cg::LinuxDynamicLib<Scalar>(
    //     library_generic_path +
    //     CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION));
    model_ = lib_->model(model_name);
    input_size_ = model_->Domain();
    output_size_ = model_->Range();
}

template <typename Scalar>
typename CompiledModel<Scalar>::Vector CompiledModel<Scalar>::evaluate(
    const Eigen::Ref<const Vector>& input) const {
    check_input_size(input.size());
    return model_->template ForwardZero<Vector>(input);
}

template <typename Scalar>
typename CompiledModel<Scalar>::Vector CompiledModel<Scalar>::evaluate(
    const Eigen::Ref<const Vector>& input,
    const Eigen::Ref<const Vector>& parameters) const {
    check_input_size_with_params(input.size(), parameters.size());

    Vector xp(input.size() + parameters.size());
    xp << input, parameters;
    return evaluate(xp);
}

template <typename Scalar>
typename CompiledModel<Scalar>::Matrix CompiledModel<Scalar>::jacobian(
    const Eigen::Ref<const Vector>& input) const {
    if (!model_->isJacobianAvailable()) {
        throw std::runtime_error(
            "Jacobian is not available: compiled model must be at least "
            "first-order.");
    }
    check_input_size(input.size());

    assert(input.rows() == input_size_);
    Vector J_vec = model_->template Jacobian<Vector>(input);
    Eigen::Map<Matrix> J(J_vec.data(), output_size_, input_size_);
    assert(J.allFinite());
    return J;
}

template <typename Scalar>
typename CompiledModel<Scalar>::Matrix CompiledModel<Scalar>::jacobian(
    const Eigen::Ref<const Vector>& input,
    const Eigen::Ref<const Vector>& parameters) const {
    check_input_size_with_params(input.size(), parameters.size());

    Vector xp(input.size() + parameters.size());
    xp << input, parameters;
    return jacobian(xp).leftCols(input.rows());
}

template <typename Scalar>
typename CompiledModel<Scalar>::Matrix CompiledModel<Scalar>::hessian(
    const Eigen::Ref<const Vector>& input, size_t output_dim) const {
    if (!model_->isHessianAvailable()) {
        throw std::runtime_error(
            "Hessian is not available: compiled model must be "
            "second-order.");
    }
    if (output_dim >= output_size_) {
        throw std::runtime_error("Specified output dimension for Hessian is " +
                                 std::to_string(output_dim) +
                                 ", but model has only " +
                                 std::to_string(output_size_) + " outputs.");
    }
    check_input_size(input.size());

    // Need to use the overload of the Hessian function that accepts a weight
    // vector w. Otherwise, CppADCodeGen creates a w vector but does not
    // initialize it to zero, which can cause errors.
    Vector w = Vector::Zero(output_size_);
    w(output_dim) = 1.0;
    Vector H_vec = model_->template Hessian<Vector>(input, w);
    Eigen::Map<Matrix> H(H_vec.data(), input_size_, input_size_);
    assert(H.allFinite());
    return H;
}

template <typename Scalar>
typename CompiledModel<Scalar>::Matrix CompiledModel<Scalar>::hessian(
    const Eigen::Ref<const Vector>& input,
    const Eigen::Ref<const Vector>& parameters, size_t output_dim) const {
    check_input_size_with_params(input.size(), parameters.size());

    Vector xp(input.size() + parameters.size());
    xp << input, parameters;
    return hessian(xp, output_dim).topLeftCorner(input.rows(), input.rows());
}

template <typename Scalar>
size_t CompiledModel<Scalar>::get_input_size() const {
    return input_size_;
}

template <typename Scalar>
size_t CompiledModel<Scalar>::get_output_size() const {
    return output_size_;
}

template <typename Scalar>
void CompiledModel<Scalar>::check_input_size(size_t size) const {
    if (size < input_size_) {
        throw std::runtime_error(
            "Model domain is " + std::to_string(input_size_) +
            ", but input is of size " + std::to_string(size) +
            ". Did you mean to pass parameters, too?");
    } else if (size > input_size_) {
        throw std::runtime_error(
            "Model domain is " + std::to_string(input_size_) +
            ", but input is of size " + std::to_string(size));
    }
}

template <typename Scalar>
void CompiledModel<Scalar>::check_input_size_with_params(
    size_t input_size, size_t param_size) const {
    size_t total_size = input_size + param_size;
    if (total_size > input_size_) {
        throw std::runtime_error(
            "Input size is " + std::to_string(input_size) +
            " and parameter size is " + std::to_string(param_size) +
            ". The total is " + std::to_string(total_size) +
            ", which larger than the model domain " +
            std::to_string(input_size_) +
            ". Maybe you meant not to pass parameters?");
    }
}
