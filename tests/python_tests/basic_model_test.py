import pytest
import numpy as np

from CppADCodeGenEigenPy import CompiledModel

MODEL_NAME = "BasicTestModel"
MODEL_LIB = "libBasicTestModel"

NUM_INPUT = 3
NUM_OUTPUT = 3


@pytest.fixture
def model():
    # TODO want an error raised if lib not found
    return CompiledModel(MODEL_NAME, MODEL_LIB)


def test_model_dimensions(model):
    assert model.input_size == NUM_INPUT
    assert model.output_size == NUM_OUTPUT


def test_model_evaluate(model):
    x = np.ones(NUM_INPUT)
    y_expected = 2 * x
    y_actual = model.evaluate(x)
    assert np.allclose(y_actual, y_expected)

    # incorrect input size should raise an error
    with pytest.raises(RuntimeError):
        model.evaluate(np.ones(NUM_INPUT + 1))

    # same thing with erroneous parameter vector
    with pytest.raises(RuntimeError):
        model.evaluate(x, np.ones(1))

    # empty parameter is actually valid (functions are overloaded)
    model.evaluate(x, np.ones(0))


def test_model_jacobian(model):
    x = np.ones(NUM_INPUT)
    J_expected = 2 * np.eye(NUM_INPUT)
    J_actual = model.jacobian(x)
    assert np.allclose(J_actual, J_expected)

    with pytest.raises(RuntimeError):
        model.jacobian(np.ones(NUM_INPUT + 1))

    with pytest.raises(RuntimeError):
        model.jacobian(x, np.ones(1))


def test_model_hessian(model):
    x = np.ones(NUM_INPUT)

    H_expected = np.zeros((NUM_INPUT, NUM_INPUT))
    H0_actual = model.hessian(x, 0)
    H1_actual = model.hessian(x, 1)
    H2_actual = model.hessian(x, 2)

    assert np.allclose(H0_actual, H_expected, atol=1e7)
    assert np.allclose(H1_actual, H_expected, atol=1e7)
    assert np.allclose(H2_actual, H_expected, atol=1e7)

    with pytest.raises(RuntimeError):
        model.hessian(np.ones(NUM_INPUT + 1), 0)

    with pytest.raises(RuntimeError):
        model.hessian(x, np.ones(1), 0)

    # invalid output dimension
    with pytest.raises(RuntimeError):
        model.hessian(x, 3)
