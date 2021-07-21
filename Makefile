CCPP=g++
BIN_DIR=bin
LIB_DIR=lib

# TODO
bindings:
	$(CCPP) -O3 -Wall -shared -I/usr/local/include/eigen3 -Iinclude -Ideps/pybind11/include $(shell python3.7-config --includes) -std=c++11 -fPIC src/bindings.cpp -ldl -o $(LIB_DIR)/example$(shell python3.7-config --extension-suffix)

clean:
	@rm -rf $(BIN_DIR)

.PHONY: clean bindings model compiler evaluator
