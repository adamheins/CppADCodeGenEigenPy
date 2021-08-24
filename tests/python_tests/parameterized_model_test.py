import pytest
import numpy as np

from CppADCodeGenEigenPy import CompiledModel

BUILD_DIR_NAME = "build"
MODEL_NAME = "ParameterizedTestModel"
MODEL_LIB_NAME = "lib" + MODEL_NAME

NUM_INPUT = 3
NUM_PARAM = NUM_INPUT
NUM_OUTPUT = 1


@pytest.fixture
def model(pytestconfig):
    lib_path = str(
        pytestconfig.rootdir / pytestconfig.getoption("builddir") / MODEL_LIB_NAME
    )
    return CompiledModel(MODEL_NAME, lib_path)


def test_model_evaluate(model):
    x = 2 * np.ones(NUM_INPUT)
    p = np.ones(NUM_INPUT)
    y_expected = 0.5 * sum(p * x * x)
    y_actual = model.evaluate(x, p)
    assert np.allclose(y_actual, y_expected)

    # incorrect input size should raise an error
    with pytest.raises(RuntimeError):
        model.evaluate(np.ones(NUM_INPUT + 1))

    # same thing with incorrectly-sized parameter vector
    with pytest.raises(RuntimeError):
        model.evaluate(x, np.ones(1))

    # input of size (NUM_INPUT + NUM_PARAM) is actually valid
    model.evaluate(np.ones(NUM_INPUT + NUM_PARAM))


def test_model_jacobian(model):
    x = 2 * np.ones(NUM_INPUT)
    p = np.ones(NUM_PARAM)
    P = np.diag(p)

    J_expected = x.dot(P)
    J_actual = model.jacobian(x, p)
    assert np.allclose(J_actual, J_expected)

    with pytest.raises(RuntimeError):
        model.jacobian(np.ones(NUM_INPUT + 1))

    with pytest.raises(RuntimeError):
        model.jacobian(x, np.ones(1))


def test_model_hessian(model):
    x = 2 * np.ones(NUM_INPUT)
    p = np.ones(NUM_PARAM)
    P = np.diag(p)

    H_expected = np.diag(P)
    H_actual = model.hessian(x, p, 0)

    assert np.allclose(H_actual, H_expected, atol=1e7)

    with pytest.raises(RuntimeError):
        model.hessian(np.ones(NUM_INPUT + 1), 0)

    with pytest.raises(RuntimeError):
        model.hessian(x, np.ones(1), 0)

    # invalid output dimension
    with pytest.raises(RuntimeError):
        model.hessian(x, 3)
