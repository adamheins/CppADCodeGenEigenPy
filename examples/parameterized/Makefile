CCPP=g++
SRC_DIR=src
BIN_DIR=bin
LIB_DIR=lib
COMPILER_BIN=$(BIN_DIR)/compile_model
COMPILER_SRC=$(SRC_DIR)/compile_model.cpp
MODEL_NAME=MyParameterizedModel
MODEL_LIB=$(LIB_DIR)/lib$(MODEL_NAME).so
EVALUATOR_SRC=$(SRC_DIR)/evaluate_model.cpp
EVALUATOR_BIN=$(BIN_DIR)/evaluate_model

INCLUDE_DIRS=-I/usr/local/include/eigen3 -Iinclude

.PHONY: evaluate
evaluate: $(EVALUATOR_BIN)
	./$(EVALUATOR_BIN)

# evaluate the model
.PHONY: evaluator
evaluator: $(EVALUATOR_BIN)

$(EVALUATOR_BIN): $(MODEL_LIB) $(EVALUATOR_SRC)
	@mkdir -p $(BIN_DIR)
	$(CCPP) $(INCLUDE_DIRS) -std=c++11 $(EVALUATOR_SRC) -ldl -o $(EVALUATOR_BIN)

# compile the model
.PHONY: model
model: $(MODEL_LIB)

$(MODEL_LIB): $(COMPILER_BIN)
	./$(COMPILER_BIN)

# make the model compiler
.PHONY: compiler
compiler: $(COMPILER_BIN)

$(COMPILER_BIN): $(COMPILER_SRC)
	@mkdir -p $(BIN_DIR)
	$(CCPP) $(INCLUDE_DIRS) -std=c++11 $(COMPILER_SRC) -ldl -lboost_system -lboost_filesystem -o $(COMPILER_BIN)

.PHONY: clean
clean:
	@rm -rf $(BIN_DIR) $(LIB_DIR)
