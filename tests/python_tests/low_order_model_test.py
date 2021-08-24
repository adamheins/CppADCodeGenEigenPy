import pytest
import numpy as np

from CppADCodeGenEigenPy import CompiledModel

MODEL_NAME = "LowOrderTestModel"
MODEL_LIB_NAME = "lib" + MODEL_NAME

NUM_INPUT = 3


@pytest.fixture
def model(pytestconfig):
    lib_path = str(
        pytestconfig.rootdir / pytestconfig.getoption("builddir") / MODEL_LIB_NAME
    )
    return CompiledModel(MODEL_NAME, lib_path)


def test_throws_on_jacobian_hessian(model):
    x = np.ones(NUM_INPUT)

    with pytest.raises(RuntimeError):
        model.jacobian(x)

    with pytest.raises(RuntimeError):
        model.hessian(x, 0)
