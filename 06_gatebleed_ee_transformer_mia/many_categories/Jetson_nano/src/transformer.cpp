#include "transformer.h"
#include <stdexcept> // for std::out_of_range

using namespace std;
std::vector<std::vector<float>> Transformer::add_matrices(const std::vector<std::vector<float>> &a, const std::vector<std::vector<float>> &b) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
    std::vector<std::vector<float>> result = a;
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[0].size(); ++j) {
            result[i][j] += b[i][j];
        }
    }
    return result;
}
void Transformer::save_embedding_matrix()
{
    embedding.save_embedding_matrix();
}
void Transformer::save_attention_weights()
{
    for(int i=0;i<num_layers_local;i++)
    {
        attention_layers[i].save_weights(i);
    }
}
void Transformer::save_feed_forward_weights()
{
    for (int i = 0; i < num_layers_local; i++)
    {
        feed_forward_layers[i].save_weights(i);
    }
}

void Transformer::save_LayerNormalization_weights()
{
//not used
}

std::vector<std::vector<float>> Transformer::forward(const std::vector<int>& input, const std::vector<int>& padding_mask) {
    // Step 1: Embedding and positional encoding
    std::vector<std::vector<float>> transformer_matrix = embedding.forward(input);
    transformer_matrix = pos_encoding.add_positional_encoding(transformer_matrix);
    input_tokens = input;//Used for backprop
    // Step 2: Iterate through attention and feedforward layers
    attention_inputs.clear();
    for (size_t i = 0; i < attention_layers.size(); ++i) {
        // Save the input for residual connection
        residual_connections.clear();
        residual_connections.push_back(transformer_matrix);
        auto transformer_matrix_input = transformer_matrix;//Make a "input" Copy for attention heads loop where ouptut not will overwrite the input
        attention_inputs.push_back(transformer_matrix_input);//for backprop
        // Apply MultiHeadAttention with padding mask
        for(int j=0;j<num_heads;j++)
        {
            transformer_matrix = attention_layers[i].forward(transformer_matrix_input, transformer_matrix_input, transformer_matrix_input, padding_mask, j);
        }
        transformer_matrix = attention_L2_norm[i].forward(transformer_matrix);
        // Mask padding in attention output
        auto masked_attention_output = Utils::mask_padding(transformer_matrix, padding_mask);
        // Add residual connection and apply layer normalization
         auto norm_attention_output = add_matrices(residual_connections.back(), masked_attention_output);
        // Mask padding in normalization output
        transformer_matrix = Utils::mask_padding(norm_attention_output, padding_mask);
        // Save the input for residual connection before FeedForward
        residual_connections.clear();
        residual_connections.push_back(transformer_matrix);
        // Apply FeedForward
        auto feedforward_output = feed_forward_layers[i].forward(transformer_matrix);
        feedforward_output = feed_forward_L2_norm[i].forward(feedforward_output);
        // Mask padding in feedforward output
        auto masked_feedforward_output = Utils::mask_padding(feedforward_output, padding_mask);
        // Add residual connection and apply layer normalization
        auto norm_feedforward_output = add_matrices(residual_connections.back(), masked_feedforward_output);
        // Mask padding in final normalized output
        transformer_matrix = Utils::mask_padding(norm_feedforward_output, padding_mask);
        
    }

    return transformer_matrix;
}


float Transformer::read_attention_weight(
    int layer_index,
    const std::string& matrix_type,
    int row,
    int col
) const
{
    // Safety check
    if (layer_index < 0 || layer_index >= static_cast<int>(attention_layers.size())) {
        throw std::out_of_range("Invalid layer_index in read_attention_weight()");
    }

    // Forward the call to the appropriate MultiHeadAttention in our vector
    return attention_layers[layer_index].read_weight(matrix_type, row, col);
}



std::vector<std::vector<float>> Transformer::backward(const std::vector<std::vector<float>>& grad_pooled) {
    auto gradient = grad_pooled;
    for (int i = attention_layers.size() - 1; i >= 0; --i) {
        residual_connections.clear();
        residual_connections.push_back(grad_pooled);
        // Backprop feedforward
        gradient = feed_forward_L2_norm[i].backward(grad_pooled);
        gradient = feed_forward_layers[i].backward(gradient);
        gradient = add_matrices(gradient, residual_connections.back());
        // --- NEW: Update feed-forward weights after backward
        feed_forward_layers[i].update_weights();
        // Backprop attention
        residual_connections.clear();
        residual_connections.push_back(gradient);
        auto gradient_attention_input = gradient;//Make a "input" Copy for attention heads loop where ouptut not will overwrite the input
        for(int j=0;j<Transformer::num_heads;j++)
        {
            gradient = attention_layers[i].backward(gradient_attention_input, attention_inputs[i], j);
        }

        attention_layers[i].update_weights();
        gradient = add_matrices(gradient, residual_connections.back());
    }
    // Backprop positional encoding
    auto grad_pos = pos_encoding.backward(gradient);
    // Backprop embedding
    embedding.apply_gradients(input_tokens, grad_pos, GLOBAL_learning_rate);
    return grad_pos;
}
