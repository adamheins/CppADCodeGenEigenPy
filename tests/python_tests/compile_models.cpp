#include <iostream>
#include <string>

#include "testing/models/BasicTestModel.h"

using namespace CppADCodeGenEigenPy;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Directory path is required." << std::endl;
        return 1;
    }
    std::string directory_path = argv[1];
    BasicModelTest::BasicTestModel<double>().compile(
        BasicModelTest::MODEL_NAME, directory_path, DerivativeOrder::Second,
        /* verbose = */ true);
}
