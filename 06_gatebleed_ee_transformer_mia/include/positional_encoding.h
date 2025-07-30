#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include <vector>
#include "config.h"
class PositionalEncoding {
public:
    PositionalEncoding(int max_len, int d_model); // Constructor

    // Adds positional encoding to input embeddings
    std::vector<std::vector<float>> add_positional_encoding(const std::vector<std::vector<float>>& input);
    // Backpropagates gradients through positional encoding
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& grad_output);

private:
    std::vector<std::vector<float>> pos_encoding; // The positional encoding matrix
};

#endif // POSITIONAL_ENCODING_H

