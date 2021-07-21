CCPP=g++
SRC_DIR=src
BIN_DIR=bin
LIB_DIR=lib
INCLUDE_DIRS=-I/usr/local/include/eigen3 -I../include -Iinclude

# evaluate the model
evaluator: $(LIB_DIR)/libmy_model.so
	@mkdir -p $(BIN_DIR)
	$(CCPP) $(INCLUDE_DIRS) -std=c++11 $(SRC_DIR)/evaluate_model.cpp -ldl -o $(BIN_DIR)/evaluate_model

# compile the model
model: $(BIN_DIR)/compile_model
	./$(BIN_DIR)/compile_model

# make the model comiler
compiler:
	@mkdir -p $(BIN_DIR)
	$(CCPP) $(INCLUDE_DIRS) -std=c++11 $(SRC_DIR)/compile_model.cpp -ldl -lboost_system -lboost_filesystem -o $(BIN_DIR)/compile_model

clean:
	@rm -rf $(BIN_DIR) $(LIB_DIR)

.PHONY: clean model compiler evaluator
