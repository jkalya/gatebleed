#include "layer_normalization.h"
#include <fstream>
#include <iostream>
#include <numeric>
#include <cmath>
#include <vector>

LayerNormalization::LayerNormalization(size_t d_model, bool load_parameters_yes_no, int layer_index) {
    mean_cache.resize(d_model, 0.0f);
    stddev_cache.resize(d_model, 0.0f);

    gamma.resize(d_model, 1.0f); // Default scaling: 1.0
    beta.resize(d_model, 0.0f);  // Default shifting: 0.0f

    if (load_parameters_yes_no) {
        const std::string file_name = "normalize_weights_layer_" + std::to_string(layer_index) + ".bin";
        std::ifstream file(file_name, std::ios::binary);
        if (file.is_open()) {
            file.read(reinterpret_cast<char*>(gamma.data()), gamma.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(beta.data()), beta.size() * sizeof(float));
            file.close();
            std::cout << "LayerNormalization parameters for layer " << layer_index << " loaded from file.\n";
        } else {
            std::cerr << "Warning: Could not load normalization weights for layer " << layer_index << ". Using defaults.\n";
        }
    }
}

std::vector<std::vector<float>> LayerNormalization::forward(const std::vector<std::vector<float>>& input) {
    size_t rows = input.size();
    size_t cols = input[0].size();
    std::vector<std::vector<float>> output(rows, std::vector<float>(cols));

    for (size_t i = 0; i < rows; ++i) {
        float mean = std::accumulate(input[i].begin(), input[i].end(), 0.0f) / cols;
        float variance = 0.0f;

        for (float val : input[i]) {
            variance += (val - mean) * (val - mean);
        }
        variance /= cols;
        float stddev = std::sqrt(variance + 1e-6); // Add epsilon for numerical stability

        for (size_t j = 0; j < cols; ++j) {
            output[i][j] = gamma[j] * (input[i][j] - mean) / stddev + beta[j];
        }

        // Cache mean and stddev for backprop
        mean_cache[i] = mean;
        stddev_cache[i] = stddev;
    }

    return output;
}
std::vector<std::vector<float>> LayerNormalization::backward(
    const std::vector<std::vector<float>>& grad_output,
    const std::vector<std::vector<float>>& input,
    float epsilon
)
{    // Compute gradients as before
    size_t rows = input.size();
    size_t cols = input[0].size();
    std::vector<std::vector<float>> grad_input(rows, std::vector<float>(cols));

    for (size_t i = 0; i < rows; ++i) {
  //      float mean = mean_cache[i];
        float stddev = stddev_cache[i];

        for (size_t j = 0; j < cols; ++j) {
            grad_input[i][j] = gamma[j] * grad_output[i][j] / stddev;
        }
    }

    return grad_input;
}

void LayerNormalization::save_parameters(int layer_index) {
    const std::string file_name = "normalize_weights_layer_" + std::to_string(layer_index) + ".bin";
    std::ofstream file(file_name, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(gamma.data()), gamma.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(beta.data()), beta.size() * sizeof(float));
        file.close();
        std::cout << "LayerNormalization parameters for layer " << layer_index << " saved to file.\n";
    } else {
        std::cerr << "Error: Could not save LayerNormalization parameters to file.\n";
        exit(EXIT_FAILURE);
    }
}
