CCPP=g++
BIN_DIR=bin

bin/evaluate_model: autogen/libmy_model.so
	@mkdir -p $(BIN_DIR)
	$(CCPP) -I/usr/local/include/eigen3 -Iinclude -std=c++11 src/evaluate.cpp -ldl -o $(BIN_DIR)/evaluate_model

model: $(BIN_DIR)/model_compiler
	./$(BIN_DIR)/model_compiler

compiler: src/compile.cpp
	@mkdir -p $(BIN_DIR)
	$(CCPP) -I/usr/local/include/eigen3 -Iinclude -std=c++11 src/compile.cpp -ldl -lboost_system -lboost_filesystem -o $(BIN_DIR)/model_compiler

clean:
	@rm -rf $(BIN_DIR)

bindings:
	$(CCPP) -O3 -Wall -shared -I/usr/local/include/eigen3 -Iinclude -Ideps/pybind11/include $(shell python3.7-config --includes) -std=c++11 -fPIC src/bindings.cpp -o lib/example$(shell python3.7-config --extension-suffix)


.PHONY: clean bindings model compiler
