#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include "config.h"
#include <iostream>
#include <unordered_map>
#include <string>
#include <set>

namespace Utils {
    // Matrix multiplication
    std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b);
    
    // Transpose a matrix
    std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& matrix);
    
    // Apply softmax to a vector
    std::vector<float> softmax(const std::vector<float>& input);
    float simple_L2_norm(const std::vector<std::vector<float>>& matrix);


std::vector<std::vector<float>> softmax_backward(
    const std::vector<std::vector<float>>& grad_output,
    const std::vector<std::vector<float>>& softmax_out
);

    void scale_inplace(std::vector<std::vector<float>>& matrix, float scale_factor);
    // Print a matrix
    void print_matrix(const std::vector<std::vector<float>>& matrix);
    void print_matrix_shape(const std::vector<std::vector<float>>& matrix);
    
    // Check matrix dimensions
    void check_matrices(const std::vector<std::vector<float>>& A,
                        const std::vector<std::vector<float>>& B,
                        const std::vector<std::vector<float>>& C);

    // Leaky ReLU for a single float
    float leaky_relu(float x);
    
    float leaky_relu_derivative(float x);
    std::vector<std::vector<float>> leaky_relu_derivative(const std::vector<std::vector<float>>& input);

    // Leaky ReLU for a matrix
    std::vector<std::vector<float>> leaky_relu(const std::vector<std::vector<float>>& input);

    //mask padding
    std::vector<std::vector<float>> mask_padding(const std::vector<std::vector<float>> &matrix, const std::vector<int> &padding_mask);

    bool check_vocabs(const std::unordered_map<std::string, int>& vocab);
};

#endif // UTILS_H


