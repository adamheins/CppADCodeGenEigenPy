import numpy as np
from CppADCodeGenEigenPy import CompiledModel

# note that the .so file extension not included on the second argument
model = CompiledModel("ExampleModel", "libExampleModel")

inputs = np.array([1, 2, 3])
params = np.ones(3)

# compute model output and first derivative
output = model.evaluate(inputs, params)
jacobian = model.jacobian(inputs, params)

print(f"output = {output}")
print(f"jacobian = {jacobian}")
