#include <pinocchio/parsers/urdf.hpp>

#include "inverse_dynamics_model.h"

namespace ad = CppADCodeGenEigenPy;

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: compile_model <directory> <urdf_path>"
                  << std::endl;
        return 1;
    }
    std::string output_dir_path = argv[1];
    std::string urdf_path = argv[2];

    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_path, model);
    InverseDynamicsModel<double>(model).compile(
        "InverseDynamicsModel", output_dir_path, ad::DerivativeOrder::First);
}
