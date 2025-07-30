#ifndef SIMPLENORMALIZATION_H
#define SIMPLENORMALIZATION_H

#include <vector>
#include <iostream>

class SimpleNormalization {
public:
    SimpleNormalization(void);
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& grad_output);

private:
    float norm_gain_cache;   // Cache norm gain for backprop
    std::vector<std::vector<float>> input_cache; // Cache input for backward pass
};

#endif // SIMPLENORMALIZATION