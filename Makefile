CCPP=g++
PYTHON_DIR=python
MODULE_NAME=CppADCodeGenEigenPy
MODULE_PATH=$(PYTHON_DIR)/$(MODULE_NAME)$(shell python3.7-config --extension-suffix)
INCLUDE_DIRS=-I/usr/local/include/eigen3 -Iinclude -Ideps/pybind11/include $(shell python3.7-config --includes)

# TODO
.PHONY: bindings
bindings:
	@mkdir -p $(PYTHON_DIR)
	$(CCPP) -O3 -Wall -shared $(INCLUDE_DIRS) -std=c++11 -fPIC src/bindings.cpp -ldl -o $(MODULE_PATH)

.PHONY: clean
clean:
	@rm -rf $(PYTHON_DIR)
