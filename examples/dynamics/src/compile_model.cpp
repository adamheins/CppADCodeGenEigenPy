#include <CppADCodeGenEigenPy/ADModel.h>

#include "dynamics_model.h"
#include "rollout_model.h"

namespace ad = CppADCodeGenEigenPy;

using Scalar = double;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: compile_model <directory>" << std::endl;
        return 1;
    }
    std::string directory_path = argv[1];

    DynamicsModel<Scalar>().compile("DynamicsModel", directory_path,
                                    ad::DerivativeOrder::First);
    RolloutCostModel<Scalar>().compile("RolloutCostModel", directory_path,
                                       ad::DerivativeOrder::First);
}
