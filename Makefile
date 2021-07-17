CCPP=g++

BIN_DIR=bin

bin/evaluate_model: autogen/libmy_model.so
	@mkdir -p $(BIN_DIR)
	$(CCPP) -I/usr/local/include/eigen3 -Iinclude -std=c++11 src/evaluate.cpp -ldl -o $(BIN_DIR)/evaluate_model

autogen/libmy_model.so: $(BIN_DIR)/model_compiler
	./$(BIN_DIR)/model_compiler

$(BIN_DIR)/model_compiler: src/compile.cpp
	@mkdir -p $(BIN_DIR)
	$(CCPP) -I/usr/local/include/eigen3 -Iinclude -std=c++11 src/compile.cpp -ldl -lboost_system -lboost_filesystem -o $(BIN_DIR)/model_compiler

clean:
	@rm -rf $(BIN_DIR)

.PHONY: clean
