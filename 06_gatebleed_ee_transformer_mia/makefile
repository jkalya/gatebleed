CXX = g++
CXXFLAGS = -g -Wall -Iinclude -std=c++17 -march=native

TARGET = transformer_app
SRCS = src/main.cpp src/attention.cpp src/embedding.cpp src/feed_forward.cpp src/positional_encoding.cpp src/transformer.cpp src/utils.cpp src/config.cpp src/dataset.cpp src/layer_normalization.cpp src/simple_normalizer.cpp

SRCS_INFER = src/infer.cpp src/attention.cpp src/embedding.cpp src/feed_forward.cpp src/positional_encoding.cpp src/transformer.cpp src/utils.cpp src/config.cpp src/dataset.cpp src/layer_normalization.cpp src/simple_normalizer.cpp

SRCS_INFER_WITH_ENTROPY = src/infer_with_entropy.cpp src/attention.cpp src/embedding.cpp src/feed_forward.cpp src/positional_encoding.cpp src/transformer.cpp src/utils.cpp src/config.cpp src/dataset.cpp src/layer_normalization.cpp src/simple_normalizer.cpp

SRCS_FUZZ = src/fuzz.cpp src/attention.cpp src/embedding.cpp src/feed_forward.cpp src/positional_encoding.cpp src/transformer.cpp src/utils.cpp src/config.cpp src/dataset.cpp src/layer_normalization.cpp src/simple_normalizer.cpp

OBJS = $(patsubst src/%.cpp,build/%.o,$(SRCS))

all: build/$(TARGET) build/infer

build/$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

build/%.o: src/%.cpp
	mkdir -p build  # Create the 'build' directory at the top level
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf build
.PHONY: all clean

build/infer: build/infer.o
	$(CXX) $(CXXFLAGS) $(SRCS_INFER) -o build/infer
build/fuzz:
	$(CXX) $(CXXFLAGS) $(SRCS_FUZZ) -o build/fuzz
build/infer_with_entropy:
	$(CXX) $(CXXFLAGS) $(SRCS_INFER_WITH_ENTROPY) -o build/infer_with_entropy

# sudo taskset -c 28 ./transformer_app 
