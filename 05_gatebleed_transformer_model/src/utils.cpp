#include "utils.h"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm> // Add this for std::max_element
#include <iomanip>   // For formatted output
#include <cstring>
#include <iostream>
#include <cstdint>

#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdbool.h>

bool Utils::check_vocabs(const std::unordered_map<std::string, int>& vocab) {
    std::set<int> indices; // To track used indices
    std::set<std::string> strings; // To track duplicate strings
    int max_index = vocab.size() - 1;

    for (const auto& [word, index] : vocab) {
        // Check for duplicate strings
        if (strings.find(word) != strings.end()) {
            std::cerr << "Error: Duplicate word found in vocab: " << word << "\n";
            return false;
        }
        strings.insert(word);

        // Check if indices are within valid range and unique
        if (index < 0 || index > max_index) {
            std::cerr << "Error: Index " << index << " is out of range for word: " << word << "\n";
            return false;
        }
        if (indices.find(index) != indices.end()) {
            std::cerr << "Error: Duplicate index found in vocab: " << index << " for word: " << word << "\n";
            return false;
        }
        indices.insert(index);
    }

    // Check if all indices are contiguous from 0 to max_index
    if (indices.size() != vocab.size()) {
        std::cerr << "Error: Missing indices. Vocab size is " << vocab.size()
                  << " but unique indices are " << indices.size() << "\n";
        return false;
    }

    for (int i = 0; i <= max_index; ++i) {
        if (indices.find(i) == indices.end()) {
            std::cerr << "Error: Missing index " << i << " in vocab.\n";
            return false;
        }
    }

    std::cout << "Vocab validation passed: No duplicates, and indices are contiguous.\n";
    return true;
}

float Utils::leaky_relu(float x) {
    return (x > 0) ? x : GLOBAL_LEAKY_SLOPE * x;
}

std::vector<std::vector<float>> Utils::leaky_relu(const std::vector<std::vector<float>>& input) {
    std::vector<std::vector<float>> output = input;
    for (auto& row : output) {
        for (auto& val : row) {
            val = leaky_relu(val);
        }
    }
    return output;
}
float Utils::leaky_relu_derivative(float x) {
    return (x > 0) ? 1.0f : GLOBAL_LEAKY_SLOPE;
}

std::vector<std::vector<float>> Utils::leaky_relu_derivative(const std::vector<std::vector<float>>& input) {
    std::vector<std::vector<float>> output = input;
    for (auto& row : output) {
        for (auto& val : row) {
            val = leaky_relu_derivative(val);
        }
    }
    return output;
}

// Function to get the current value from the RDTSC
static inline uint64_t rdtsc() {
    unsigned int lo, hi;
    asm volatile("mfence");
    __asm__ volatile ("rdtsc" : "=a" (lo), "=d" (hi));
    asm volatile("mfence");
    return ((uint64_t)hi << 32) | lo;
}


namespace Utils {

    // Function to check matrix dimensions
    void check_matrices(const std::vector<std::vector<float>>& A,
                        const std::vector<std::vector<float>>& B,
                        const std::vector<std::vector<float>>& C) {
        if (A.size() != B.size() || A.size() != C.size()) {
            std::cout << "Error: Number of rows in Q, K, and V must match. Program terminated.\n";
            exit(1);
        }
        for (size_t i = 0; i < A.size(); ++i) {
            if (A[i].size() != B[i].size() || A[i].size() != C[i].size()) {
                std::cout << "Error: Row " << i << " of Q, K, and V must have the same number of columns. Program terminated.\n";
                exit(1);
            }
        }
        std::cout << "Matrix sizes are valid.\n";
    }

    // Existing functions (matmul, transpose, softmax, etc.) remain here
}


std::vector<std::vector<float>> Utils::matmul_old(const std::vector<std::vector<float>> &a, const std::vector<std::vector<float>> &b)
{
    if (a[0].size() != b.size()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    size_t rows = a.size();
    size_t cols = b[0].size();
    size_t inner_dim = b.size();
    // std::cout << "rows: " << rows << " - cols: "  << cols << " - inner_dim: " << inner_dim << ".\n";
    
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols, 0.0f));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            for (size_t k = 0; k < inner_dim; ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return result;
}

std::vector<std::vector<float>> Utils::matmul(const std::vector<std::vector<float>> &a, const std::vector<std::vector<float>> &b)
{
    if (a[0].size() != b.size()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    size_t rows = a.size();
    size_t cols = b[0].size();
    size_t inner_dim = b.size();
    //std::cout << "rows: " << rows << " - cols: "  << cols << " - inner_dim: " << inner_dim << ".\n";
    size_t rows_split = (rows + 15) / 16;
    size_t cols_split = (cols + 15) / 16;
    size_t inner_split = (inner_dim + 31) / 32;
    // std::cout << "rows_split: " << rows_split << " - cols_split: "  << cols_split << " - inner_split: " << inner_split << ".\n";
    
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols, 0.0f));

    float value;
    uint32_t bits;
    //uint16_t upperBits;
    uint16_t Tile_A_Values[16*32];
    uint16_t Tile_B_Values[16*32];
    float Tile_Res_Values[16*16];
    //int a_zero_count = 0, b_zero_count = 0;

    //uint64_t start_cycles, end_cycles, elapsed_cycles;

    for (size_t r = 0; r < rows_split; ++r) {
        for (size_t c = 0; c < cols_split; ++c) {
            //std::fill(std::begin(Tile_Res_Values), std::end(Tile_Res_Values), (float)0);
            //_tile_loadd (0, Tile_Res_Values, 64);   // Load zeros into tile 0
            _tile_zero(0);
            for (size_t in = 0; in < inner_split; ++in) {
                //a_zero_count = 0; b_zero_count = 0;
                // preparing Tile1 data from Matrix a
                std::fill(std::begin(Tile_A_Values), std::end(Tile_A_Values), (uint16_t)0);
                for (size_t i = 0; i < std::min((size_t)16, rows-r*16); ++i) {
                    for (size_t j = 0; j < std::min((size_t)32, inner_dim-in*32); ++j) {
                        value = a[r*16+i][in*32+j];
                        //a_zero_count = (a[r*16+i][in*32+j] == 0) ? a_zero_count + 1 : a_zero_count;
                        //a_zero_count = a_zero_count + 1;
                        std::memcpy(&bits, &value, sizeof(bits));
                        Tile_A_Values[i*32+j] = static_cast<uint16_t>(bits >> 16); // upperbits: bfloat16
                    }
                }
                //_mm_mfence();
                //uint64_t start_cycles = rdtsc();
                _tile_loadd (1, Tile_A_Values, 64);   // Load data from Matrix a into tile 1
                //_mm_mfence();
                //uint64_t end_cycles = rdtsc();
                //uint64_t elapsed_cycles =  (end_cycles - start_cycles);
                //std::cout << "matmul-load1: " << elapsed_cycles << "\n";
                // preparing Tile2 data from Matrix b
                std::fill(std::begin(Tile_B_Values), std::end(Tile_B_Values), 0);
                for (size_t i = 0; i < std::min((size_t)32, inner_dim-in*32); ++i) {
                    for (size_t j = 0; j < std::min((size_t)16, cols-c*16); ++j) {
                        value = b[in*32+i][c*16+j];
                        //b_zero_count = (b[in*32+i][c*16+j] == 0) ? b_zero_count + 1 : b_zero_count;
                        //b_zero_count = b_zero_count + 1;
                        std::memcpy(&bits, &value, sizeof(bits));
                        Tile_B_Values[((size_t)(i/2))*32 + i%2 + 2*j] = static_cast<uint16_t>(bits >> 16); // upperbits: bfloat16
                    }
                }
                //std::cout << a_zero_count << "\t" << b_zero_count << "\n";
                //_mm_mfence();
                //start_cycles = rdtsc();
                _tile_loadd (2, Tile_B_Values, 64);   // Load data from Matrix b into tile 2
                //_mm_mfence();
                //end_cycles = rdtsc();
                //elapsed_cycles =  (end_cycles - start_cycles);
                //std::cout << "matmul-load2: " << elapsed_cycles << "\n";

                //_mm_mfence();
                //start_cycles = rdtsc();
                //elapsed_cycles =  (start_cycles - end_cycles);
                //_tile_zero(3); // _tile_zero(3); // added to check zero dependency
                _tile_dpbf16ps (0, 1, 2);
                //_mm_mfence();
                //end_cycles = rdtsc();
                //std::cout << "matmul-interval: " << elapsed_cycles << "\n";
                //elapsed_cycles =  (end_cycles - start_cycles);
                //std::cout << elapsed_cycles << "\n";
            }
            //_mm_mfence();
            //uint64_t start_cycles = rdtsc();
            _tile_stored (0, Tile_Res_Values, 64);
            //_mm_mfence();
            //uint64_t end_cycles = rdtsc();
            //uint64_t elapsed_cycles =  (end_cycles - start_cycles);
            //std::cout << "matmul-store: " << elapsed_cycles << "\n";
            // Filling the result
            for (size_t i = 0; i < std::min((size_t)16, rows-r*16); ++i) {
                for (size_t j = 0; j < std::min((size_t)16, cols-c*16); ++j) {
                    result[r*16+i][c*16+j] = Tile_Res_Values[i*16+j];
                }
            }
        }
    }
    return result;
}

std::vector<std::vector<float>> Utils::mask_padding(
    const std::vector<std::vector<float>> &matrix,
    const std::vector<int> &padding_mask)
{
    std::vector<std::vector<float>> masked_matrix = matrix;
    for (size_t i = 0; i < masked_matrix.size(); ++i)
    {
        if (padding_mask[i] == 0)
        { // If it's a [PAD] token
            std::fill(masked_matrix[i].begin(), masked_matrix[i].end(), 0.0f);
        }
    }
    return masked_matrix;
}

std::vector<std::vector<float>> Utils::transpose(const std::vector<std::vector<float>> &matrix)
{
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    std::vector<std::vector<float>> result(cols, std::vector<float>(rows));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = matrix[i][j];
        }
    }

    return result;
}

std::vector<float> Utils::softmax(const std::vector<float> &input)
{
    float max_input = *std::max_element(input.begin(), input.end()); // For numerical stability
    float sum_exp = 0.0f;

    std::vector<float> result(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = std::exp(input[i] - max_input);
        sum_exp += result[i];
    }

    for (size_t i = 0; i < input.size(); ++i) {
        result[i] /= sum_exp;
    }

    return result;
}
/**
 * @brief Computes gradient w.r.t. the input of a softmax function
 *        (row-wise softmax) for a 2D matrix.
 *
 * @param grad_output   Gradient from the next layer, same shape as softmax_out.
 *                      For each row i, grad_output[i] is dL/d(softmax_out[i]).
 * @param softmax_out   The output of softmax(input) row-by-row.
 *                      For each row i, softmax_out[i] sums to 1.
 * @return              Gradient w.r.t. the original 2D input of softmax, same shape.
 */

namespace Utils {
std::vector<std::vector<float>> softmax_backward(
    const std::vector<std::vector<float>>& grad_output,
    const std::vector<std::vector<float>>& softmax_out
) {
    // Prepare dInput with the same shape as grad_output.
    std::vector<std::vector<float>> dInput(grad_output.size());
    for (size_t i = 0; i < grad_output.size(); ++i) {
        dInput[i].resize(grad_output[i].size(), 0.0f);
    }

    // Process each row independently
    for (size_t row = 0; row < grad_output.size(); ++row) {
        // 1) Compute dot = sum_j(grad_output[row][j] * softmax_out[row][j])
        float dot = 0.0f;
        for (size_t col = 0; col < grad_output[row].size(); ++col) {
            dot += grad_output[row][col] * softmax_out[row][col];
        }

        // 2) Compute dInput[row][col] = softmax_out[row][col] * (grad_output[row][col] - dot)
        for (size_t col = 0; col < grad_output[row].size(); ++col) {
            dInput[row][col] = softmax_out[row][col] * (grad_output[row][col] - dot);
        }
    }

    return dInput;
}
}
void Utils::scale_inplace(std::vector<std::vector<float>>& matrix, float scale_factor)
{
    for (auto& row : matrix) {
        for (auto& value : row) {
            value *= scale_factor;
        }
    }
}


void Utils::print_matrix(const std::vector<std::vector<float>>& matrix)
{
    std::cout << "[\n";
    for (const auto& row : matrix) {
        std::cout << "  [";
        for (size_t i = 0; i < row.size(); ++i) {
            std::cout << std::fixed << std::setprecision(8) << row[i];
            if (i < row.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }
    std::cout << "]\n";

}


    void Utils::print_matrix_shape(const std::vector<std::vector<float>>& matrix) {
        size_t rows = matrix.size();
        size_t cols = (!matrix.empty()) ? matrix[0].size() : 0;
        std::cout << "vector_shape_size[" << rows << "][" << cols << "]\n";
    }
float Utils::simple_L2_norm(const std::vector<std::vector<float>>& matrix)
{
    float sum_squares = 0.0f;
    for (const auto& row : matrix)
    {
        for (const auto& val : row)
        {
            sum_squares += val * val;
        }
    }
    return std::sqrt(sum_squares);
}
