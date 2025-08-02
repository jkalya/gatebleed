CXX = g++
CXXFLAGS = -g -Wall -Iinclude -std=c++11 -march=native

TARGET = neural_net_in_cpp
SRCS = src/main.cpp src/NetworkModel.cpp src/MNISTDataLoader.cpp src/Tensor.cpp src/FullyConnected.cpp src/Sigmoid.cpp src/SoftmaxClassifier.cpp src/Dropout.cpp src/ReLU.cpp src/Conv2d.cpp src/MaxPool.cpp src/LinearLRScheduler.cpp
OBJS = $(patsubst src/%.cpp,build/%.o,$(SRCS))

all: build/$(TARGET)

build/$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

build/%.o: src/%.cpp
	mkdir -p build  # Create the 'build' directory at the top level
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf build

.PHONY: all clean

# sudo taskset -c 28 ./build/neural_net_in_cpp data
# sudo taskset -c 55 ./build/neural_net_in_cpp data