#ifndef LAYER_NORMALIZATION_H
#define LAYER_NORMALIZATION_H

#include <vector>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>

class LayerNormalization {
public:
    LayerNormalization(size_t d_model, bool load_parameters_yes_no, int layer_index);

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& grad_output, const std::vector<std::vector<float>>& input, float epsilon = 1e-6);

    void save_parameters(int layer_index);

private:
    std::vector<float> gamma;       // Scaling parameters
    std::vector<float> beta;        // Shifting parameters
    std::vector<float> mean_cache;   // Cache mean for backprop
    std::vector<float> stddev_cache; // Cache stddev for backprop
};

#endif // LAYER_NORMALIZATION_H
