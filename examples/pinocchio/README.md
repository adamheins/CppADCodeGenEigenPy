# CppADCodeGenEigenPy Pinocchio Example

This is an example of using CppADCodeGenEigenPy with the
[Pinocchio](https://github.com/stack-of-tasks/pinocchio) kinematics and
dynamics library. The functionality is very similar to Pinocchio's built-in
[CppADCodeGen support](https://github.com/stack-of-tasks/pinocchio/tree/master/examples/codegen).

The [InverseDynamicsModel](include/inverse_dynamics_model.h) calls Pinocchio's
recursive Newton-Euler algorithm (RNEA) routine to compute the joint torques
required to achieve a desired joint acceleration given the joint position and
velocity. We compare the result against Pinocchio's built-in analytical
computation of the derivatives of RNEA to ensure correctness. Examples for UR5
and Kinova manipulators are included.

This particular example is not particularly useful in practice since Pinocchio
provides the derivatives of RNEA already. Of course, one can easily modify the
example to use arbitrary compositions of Pinocchio functions and other
operations as desired. Pinocchio does already have code generation support, but
this provides an alternative approach. Finally, this example may serve as a
blueprint for applying auto-differentiation to other libraries, as long as they
have implemented the scalar templating to support it.

## Usage

Compile the auto-differentiated models:
```
# this compiles the program to generate the model
make compiler

# actually compile the shared libraries, one for each robot
make model
```

Install required Python dependencies (note that Python 3 is expected):
```
pip install -r requirements.txt
```

Check that the auto-diff model matches the built-in Pinocchio model:
```
python scripts/test_inverse_dynamics_model.py
```
The robot being used can be changed by editing the `ROBOT_NAME` variable in the
script.
