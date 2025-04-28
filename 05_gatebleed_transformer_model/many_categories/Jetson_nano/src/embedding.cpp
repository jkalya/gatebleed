#include "embedding.h"
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
using namespace std;

Embedding::Embedding(int vocab_size, int d_model, bool load_parameters_yes_no) {
    embedding_matrix = std::vector<std::vector<float>>(vocab_size, std::vector<float>(d_model));
    const std::string embed_matrix_file_name = "embedding_matrix.bin";

    if (load_parameters_yes_no) {
        // Try to load the embedding matrix from the binary file
        std::ifstream file(embed_matrix_file_name, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open file " << embed_matrix_file_name
                      << ". Falling back to random initialization." << std::endl;
            load_parameters_yes_no = false; // Switch to random initialization
        } else {
            for (int i = 0; i < vocab_size; ++i) {
                file.read(reinterpret_cast<char*>(embedding_matrix[i].data()), d_model * sizeof(float));
                if (!file) {
                    std::cerr << "Error: Unexpected end of file in " << embed_matrix_file_name << "." << std::endl;
                    exit(EXIT_FAILURE);
                }

                //for (int j = 0; j < d_model; ++j) {
                  //  cout << "Read embedding_matrix[" << i << "][" << j << "]: " << embedding_matrix[i][j] << endl;
                //}
            }
            file.close();
            std::cout << "Embedding matrix loaded from file: " << embed_matrix_file_name << std::endl;
            return; // Early exit if successful
        }
    }

    // Fallback to random initialization
    std::srand(std::time(0));
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < d_model; ++j) {
            embedding_matrix[i][j] = static_cast<float>(std::rand()) / RAND_MAX;
          //  cout << "Init embedding_matrix[" << i << "][" << j << "]: " << embedding_matrix[i][j] << endl;
        }
    }
    std::cout << "Embedding matrix initialized with random values." << std::endl;
}

void Embedding::save_embedding_matrix() {
    std::ofstream save_file(embed_matrix_file_name, std::ios::binary);
    if (!save_file.is_open()) {
        std::cerr << "Error: Could not open file " << embed_matrix_file_name << " for saving embeddings." << std::endl;
        exit(EXIT_FAILURE);
    }
    for (const auto& row : embedding_matrix) {
        save_file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }
    save_file.close();
    std::cout << "Embedding matrix saved to file: " << embed_matrix_file_name << std::endl;
}

// Forward pass
std::vector<std::vector<float>> Embedding::forward(const std::vector<int>& input) {
    std::vector<std::vector<float>> result;

    // Fetch the embedding vector for each token ID
    for (int token_id : input) {
        result.push_back(embedding_matrix[token_id]);
    }

    return result;
}

// Update function for the embedding matrix
void Embedding::apply_gradients(const std::vector<int>& input, const std::vector<std::vector<float>>& grad_embedding, float learning_rate) {
    for (size_t idx = 0; idx < input.size(); ++idx) {
        int token_id = input[idx];
        for (size_t j = 0; j < embedding_matrix[token_id].size(); ++j) {
            embedding_matrix[token_id][j] -= learning_rate * grad_embedding[idx][j]; // Gradient descent step
        }
    }
}
/*
// Apply gradients
void Embedding::apply_gradients(const std::vector<int>& input, const std::vector<std::vector<float>>& grad_embedding, float learning_rate) {
    for (size_t i = 0; i < input.size(); ++i) {
        int idx = input[i];
        for (size_t j = 0; j < grad_embedding[i].size(); ++j) {
            embedding_matrix[idx][j] -= learning_rate * grad_embedding[i][j];
        }
    }
}
*/