#pragma once

#include <cppad/cg.hpp>
#include <string>

std::string get_library_generic_path(const std::string& directory_name,
                                     const std::string& model_name) {
    return directory_name + "/lib" + model_name;
}

std::string get_library_real_path(const std::string& directory_name,
                                  const std::string& model_name) {
    std::string ext = CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION;
    return get_library_generic_path(directory_name, model_name) + ext;
}
