#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <ctime>
#include "config.h"

class FeedForward {
public:
    FeedForward(int d_model, int d_ff, bool load_parameters_yes_no, int layer_index);
    // d_ff: Dimensionality of the hidden layer in the feed-forward network.
    //       Each feed-forward network in the transformer consists of two linear layers:
    //       - The first layer expands the input dimensionality (d_model) to a larger hidden size (d_ff).
    //       - The second layer projects the hidden layer back down to the original dimensionality (d_model).
    //       This expansion allows the model to learn richer, non-linear representations
    //       by operating in a higher-dimensional space during the intermediate steps.
    //
    //       Typical values for d_ff are 2-4 times larger than d_model.
    //       For example:
    //         d_model = 128, d_ff = 256 or d_ff = 512.
    //       This ratio balances the model's capacity with computational efficiency.

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input);

     // Backward pass to compute gradients
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& grad_output);

       // New method to update weights AFTER backward pass
    void update_weights();
    // Save weights to binary files
    void save_weights(int layer_index);

private:
    std::vector<std::vector<float>> velocity_weights1; // Momentum for weights1
    std::vector<std::vector<float>> velocity_weights2; // Momentum for weights2
    std::vector<std::vector<float>> input_activations; // Cache input for backward pass
    std::vector<std::vector<float>> hidden_activations; // Cache activations for backpropagation
 
    std::vector<std::vector<float>> weights1; // First linear layer weights
    std::vector<std::vector<float>> weights2; // Second linear layer weights
    // NEW: We store the gradients for weights1 and weights2 
    //      that we compute during backward().
    std::vector<std::vector<float>> grad_weights1;
    std::vector<std::vector<float>> grad_weights2;    
    static const std::string file_prefix_feed_forward_weights;

    // Add for bias
    std::vector<float> bias1; 
    std::vector<float> bias2;

    // Gradients for bias
    std::vector<float> grad_bias1;
    std::vector<float> grad_bias2;

    // Momentum for bias
    std::vector<float> velocity_bias1;
    std::vector<float> velocity_bias2;
   

};
#endif // FEED_FORWARD_H