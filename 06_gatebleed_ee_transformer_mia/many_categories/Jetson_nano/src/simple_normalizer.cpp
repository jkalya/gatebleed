
#include "utils.h"   // For Utils::simple_L2_norm (or you can inline the norm code)
#include <cmath>
#include <stdexcept>
#include "simple_normalizer.h"

SimpleNormalization::SimpleNormalization(void)
    : norm_gain_cache(0.0f)
{}

/**
 * @brief Forward pass: 
 *  1) Calculate the global L2 norm of the entire input matrix.
 *  2) Store it in norm_gain_cache.
 *  3) Return normalized matrix: out[i][j] = in[i][j] / norm_gain_cache.
 */
std::vector<std::vector<float>> SimpleNormalization::forward(const std::vector<std::vector<float>>& input)
{
    input_cache = input;//used for backward

    // 1) Compute global L2 norm
    float sum_squares = 0.0f;
    for (const auto& row : input) {
        for (auto val : row) {
            sum_squares += val * val;
        }
    }
    float epsilon = 1e-9f;  // to avoid division by zero
    norm_gain_cache = std::sqrt(sum_squares + epsilon);

    // 2) Create an output matrix of the same shape
    std::vector<std::vector<float>> output = input;

    // 3) Normalize each element by the single norm_gain_cache
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < input[i].size(); ++j) {
            output[i][j] /= norm_gain_cache;
        }
    }

    return output;
}

/**
 * @brief Backward pass:
 *  Given grad_output = dL/d(normalized_matrix), compute dL/d(input).
 *
 *  For each element x_ij:
 *    dL/dx_ij = (1 / norm_gain_cache) * grad_output_ij
 *               - [ (x dot grad_output) / (norm_gain_cache^3) ] * x_ij
 *
 *  where (x dot grad_output) = sum over all (i, j) of [ x_ij * grad_output_ij ].
 */
std::vector<std::vector<float>> SimpleNormalization::backward(
    const std::vector<std::vector<float>>& grad_output)
{

   // 1) Compute the dot product: dotVal = sum_{i,j} ( x_ij * grad_output_ij )

    float dotVal = 0.0f;
    for (size_t i = 0; i < input_cache.size(); ++i) {
        for (size_t j = 0; j < input_cache[i].size(); ++j) {
            dotVal += input_cache[i][j] * grad_output[i][j];
        }
    }

    // 2) Compute gradient wrt each element of input
    std::vector<std::vector<float>> grad_input(input_cache.size(),
        std::vector<float>(input_cache[0].size(), 0.0f));

    float denom = norm_gain_cache * norm_gain_cache * norm_gain_cache; 
    // norm_gain_cache^3
    for (size_t i = 0; i < input_cache.size(); ++i) {
        for (size_t j = 0; j < input_cache[i].size(); ++j) {
            float term1 = grad_output[i][j] / norm_gain_cache;
            float term2 = (dotVal / denom) * input_cache[i][j];
            grad_input[i][j] = term1 - term2;
        }
    }
    return grad_input;
}
