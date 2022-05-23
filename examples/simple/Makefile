EIGEN_INCLUDE_DIRS=-I/usr/include/eigen3 -I/usr/local/include/eigen3

.PHONY: all
all:
	g++ -std=c++11 $(EIGEN_INCLUDE_DIRS) model.cpp -ldl -o make_model
	./make_model
