#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <vector>
#include <string>
#include "config.h"
#include <iostream>

class Embedding {
public:
    // Constructor with optional file loading
    Embedding(int vocab_size, int d_model, bool load_parameters_yes_no);

    // Forward pass
    std::vector<std::vector<float>> forward(const std::vector<int>& input);

    // Backward pass
    void backward(const std::vector<std::vector<float>>& grad_embedding);

    // Apply gradients to the embedding matrix
    void apply_gradients(const std::vector<int>& input, const std::vector<std::vector<float>>& grad_embedding, float learning_rate);
    
    // Method to save the embedding matrix
    void save_embedding_matrix();

private:
    std::vector<std::vector<float>> embedding_matrix; // The embedding lookup table
    std::string embed_matrix_file_name = "embedding_matrix.bin"; // File name for saving/loading
};

#endif // EMBEDDING_H
