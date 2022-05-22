CCPP=g++
SRC_DIR=src
BIN_DIR=bin
LIB_DIR=lib
COMPILER_BIN=$(BIN_DIR)/compile_model
COMPILER_SRC=$(SRC_DIR)/compile_model.cpp

INCLUDE_DIRS=-I/usr/local/include/eigen3 -Iinclude
CPP_FLAGS=-std=c++11

# make the compiler for the model
.PHONY: compiler
compiler:
	@mkdir -p $(BIN_DIR)
	$(CCPP) $(INCLUDE_DIRS) $(CPP_FLAGS) $(COMPILER_SRC) -ldl -o $(COMPILER_BIN)

# use the compiler to generate the model itself
.PHONY: model
model:
	@mkdir -p $(LIB_DIR)
	./$(COMPILER_BIN) $(LIB_DIR)

# clean up
.PHONY: clean
clean:
	@rm -rf $(BIN_DIR)
	@rm -rf $(LIB_DIR)
