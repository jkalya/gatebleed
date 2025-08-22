#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include "embedding.h"
#include "positional_encoding.h"
#include "attention.h"
#include "feed_forward.h"
#include "config.h"
#include "utils.h"
#include "layer_normalization.h"
#include "simple_normalizer.h"

class Transformer
{
public:
    Transformer(int vocab_size, int d_model, int num_heads, int max_len, int d_ff, int num_layers, bool load_parameters_yes_no)
        : embedding(vocab_size, d_model, load_parameters_yes_no),
          pos_encoding(max_len, d_model),
          num_heads(num_heads),
          max_len(max_len),
          d_model(d_model)          

    {
        num_layers_local = num_layers;
        for (int i = 0; i < num_layers; ++i)
        {
            attention_layers.emplace_back(d_model, num_heads, max_len, load_parameters_yes_no, i);
            feed_forward_layers.emplace_back(d_model, d_ff, load_parameters_yes_no, i);

            attention_L2_norm.emplace_back();// Instantiate layer normalization object
            feed_forward_L2_norm.emplace_back();// Instantiate layer normalization object
        }

        std::cout << "Transformer initialized with " << num_layers << " layers." << std::endl;
    }

    std::vector<std::vector<float>> forward(const std::vector<int>& input, const std::vector<int>& padding_mask);
	std::vector<std::vector<float>> forward_with_early_exit(const std::vector<int>& input, const std::vector<int>& padding_mask, bool earlyExit);

    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& grad_pooled);

    void save_embedding_matrix();
    void save_attention_weights();
    void save_feed_forward_weights();
    void save_LayerNormalization_weights();

    float read_attention_weight(
        int layer_index,
        const std::string& matrix_type,
        int row,
        int col
    ) const;

private:
    Embedding embedding;
    PositionalEncoding pos_encoding;
    std::vector<MultiHeadAttention> attention_layers;
    std::vector<FeedForward> feed_forward_layers;

    std::vector<SimpleNormalization> attention_L2_norm;
    std::vector<SimpleNormalization> feed_forward_L2_norm;


    // Intermediate values for backpropagation
    std::vector<int> input_tokens;
    std::vector<std::vector<std::vector<float>>> residual_connections;
    std::vector<std::vector<std::vector<float>>> attention_inputs;
    std::vector<std::vector<std::vector<float>>> feedforward_outputs;

    // Helper functions
    std::vector<std::vector<float>> add_matrices(const std::vector<std::vector<float>> &a, const std::vector<std::vector<float>> &b);
    int num_layers_local;
    int num_heads;
    int max_len;
    int d_model;    
};

#endif // TRANSFORMER_H
