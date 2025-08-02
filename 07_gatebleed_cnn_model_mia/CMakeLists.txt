cmake_minimum_required(VERSION 3.5)
project(neural_net_in_cpp)

set(CMAKE_CXX_STANDARD 11)
#set (CMAKE_CXX_FLAGS "-pg") -- for profiling

# Add compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

add_executable(neural_net_in_cpp src/main.cpp src/NetworkModel.cpp src/MNISTDataLoader.cpp src/Tensor.cpp src/FullyConnected.cpp src/Sigmoid.cpp src/SoftmaxClassifier.cpp src/Dropout.cpp src/ReLU.cpp src/Conv2d.cpp src/MaxPool.cpp src/LinearLRScheduler.cpp)
