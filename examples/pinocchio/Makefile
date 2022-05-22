CCPP=g++
SRC_DIR=src
BIN_DIR=bin
LIB_DIR=lib
URDF_DIR=urdf
COMPILER_BIN=$(BIN_DIR)/compile_model
COMPILER_SRC=$(SRC_DIR)/compile_model.cpp

UR_NAME=ur5
KINOVA_NAME=kinova
URDF_PATH=urdf/kinova.urdf

INCLUDE_DIRS=-I/usr/local/include/eigen3 -Iinclude
CPP_FLAGS=-std=c++11
PINOCCHIO_FLAGS=$(shell pkg-config --cflags --libs pinocchio)

# make the compiler for the model
.PHONY: compiler
compiler:
	@mkdir -p $(BIN_DIR)
	$(CCPP) $(INCLUDE_DIRS) $(CPP_FLAGS) $(COMPILER_SRC) -ldl -o $(COMPILER_BIN) $(PINOCCHIO_FLAGS)

# use the compiler to generate the robot inverse dynamics models
.PHONY: model
model:
	@mkdir -p $(LIB_DIR)/$(UR_NAME)
	@mkdir -p $(LIB_DIR)/$(KINOVA_NAME)
	./$(COMPILER_BIN) $(LIB_DIR)/$(UR_NAME) $(URDF_DIR)/$(UR_NAME).urdf
	./$(COMPILER_BIN) $(LIB_DIR)/$(KINOVA_NAME) $(URDF_DIR)/$(KINOVA_NAME).urdf

# clean up
.PHONY: clean
clean:
	@rm -rf $(BIN_DIR)
	@rm -rf $(LIB_DIR)
