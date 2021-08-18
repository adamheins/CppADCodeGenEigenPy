#pragma once

#include <cppad/cg.hpp>
#include <string>

std::string get_library_generic_path(const std::string& model_name,
                                     const std::string& directory_path) {
    return directory_path + "/lib" + model_name;
}

std::string get_library_real_path(const std::string& model_name,
                                  const std::string& directory_path) {
    std::string ext = CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION;
    return get_library_generic_path(model_name, directory_path) + ext;
}
