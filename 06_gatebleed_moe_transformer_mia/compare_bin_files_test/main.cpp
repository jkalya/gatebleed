#include "transformer.h"
#include <iostream>
#include "dataset.h"
#include <vector>
#include "utils.h"
#include <unordered_map>
using namespace std;

#include "config.h"
#ifdef TEST_UTILS
#include "attention.h"
#endif

#ifdef TEST_FEEDFORWARD
#include "feed_forward.h"
#endif

#include <algorithm> // For Fisher-Yates shuffle
#include <random>    // For random number generation
#include <chrono>    // For seeding random number generator
#include <algorithm> // std::min



#ifdef TEST_FEEDFORWARD_TRAIN
#include "feed_forward.h"
#include <cmath>
#include <random> // std::mt19937, std::uniform_real_distribution
#include <ctime>  // std::time

// Simple MSE loss function for 2D outputs
float mse_loss(const std::vector<std::vector<float>>& predictions,
               const std::vector<std::vector<float>>& targets)
{
    float total = 0.0f;
    int count = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        for (size_t j = 0; j < predictions[i].size(); ++j) {
            float diff = predictions[i][j] - targets[i][j];
            total += diff * diff;
            ++count;
        }
    }
    return total / static_cast<float>(count);
}

// Gradient of MSE w.r.t. predictions
std::vector<std::vector<float>> mse_loss_grad(
    const std::vector<std::vector<float>>& predictions,
    const std::vector<std::vector<float>>& targets)
{
    std::vector<std::vector<float>> grad(predictions.size(),
                                         std::vector<float>(predictions[0].size(), 0.0f));
    for (size_t i = 0; i < predictions.size(); ++i) {
        for (size_t j = 0; j < predictions[i].size(); ++j) {
            // d/dy of 0.5*(y - t)^2 = (y - t). 
            // (You can omit the 0.5 if you want; it just scales the gradient.)
            grad[i][j] = (predictions[i][j] - targets[i][j]);
        }
    }
    return grad;
}


#endif // TEST_FEEDFORWARD_TRAIN

// Cross-entropy loss gradient
std::vector<float> cross_entropy_loss_gradient(const std::vector<float>& probabilities, int label) {
    std::vector<float> gradient(probabilities.size(), 0.0f);
    for (size_t i = 0; i < probabilities.size(); ++i) {
        gradient[i] = probabilities[i] - (i == static_cast<size_t>(label) ? 1.0f : 0.0f);
    }
    return gradient;
}
// Function to compute cross-entropy loss
float cross_entropy_loss(const std::vector<float>& probabilities, int label) {
    return -std::log(probabilities[label] + 1e-9); // Add small epsilon for numerical stability
}



// Function to shuffle dataset
void fisher_yates_shuffle(std::vector<std::vector<int>>& dataset, std::vector<int>& labels) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);

    for (size_t i = dataset.size() - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dist(0, i);
        size_t j = dist(rng);

        std::swap(dataset[i], dataset[j]);
        std::swap(labels[i], labels[j]);
    }
}

// Function to pad a sequence to `max_len`
std::vector<int> pad_sequence(const std::vector<int>& sequence, int max_len) {
    std::vector<int> padded_sequence = sequence;
    if (padded_sequence.size() < (size_t)max_len) {
        padded_sequence.resize(max_len, 0); // Pad with 0s (assumed [PAD] token)
    }
    return padded_sequence;
}

std::vector<int> truncate_tokens_max_len(const std::vector<int>& sequence, int max_len) 
{
    // 1) Truncate if necessary
    // Use std::min to avoid out-of-range if sequence is shorter
    std::vector<int> truncated(sequence.begin(), 
                               sequence.begin() + std::min<size_t>(sequence.size(), max_len));

    // 2) If truncated.size() < max_len, pad with zeros
    if (truncated.size() < static_cast<size_t>(max_len)) {
        truncated.resize(max_len, 0); // 0 = [PAD]
    }

    return truncated;
}
// Function to create padding mask
std::vector<int> create_padding_mask(const std::vector<int>& sequence, int max_len) {
    std::vector<int> mask(max_len, 0);
    for (size_t i = 0; i < sequence.size(); ++i) {
        if (sequence[i] != 0) { // Assume non-zero tokens are valid
            mask[i] = 1;
        }
    }
    return mask;
}

// Mean Pooling
std::vector<float> mean_pooling(const std::vector<std::vector<float>>& output) {
    std::vector<float> pooled(output[0].size(), 0.0f);
    for (const auto& row : output) {
        for (size_t i = 0; i < row.size(); ++i) {
            pooled[i] += row[i];
        }
    }
    for (float& val : pooled) {
        val /= output.size();
    }
    return pooled;
}
//output_trans, pooled_output_gradient
std::vector<std::vector<float>> mean_pooling_backward(
    const std::vector<std::vector<float>>& output_from_transformer, 
    const std::vector<float>& grad_pooled
) {
    size_t rows = output_from_transformer.size();
    size_t cols = output_from_transformer[0].size();
    std::vector<std::vector<float>> grad_output_to_transformer(rows, std::vector<float>(cols, 0.0f));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            grad_output_to_transformer[i][j] = grad_pooled[j] / static_cast<float>(rows);
        }
    }
    return grad_output_to_transformer;
}

//Final Classification Layer 
std::vector<float> linear_layer(const std::vector<float>& input, const std::vector<std::vector<float>>& weights, const std::vector<float>& bias) {
    std::vector<float> output(weights[0].size(), 0.0f);

    for (size_t i = 0; i < weights[0].size(); ++i) { // For each output category
        for (size_t j = 0; j < input.size(); ++j) {  // For each input dimension
            output[i] += input[j] * weights[j][i];
        }
        output[i] += bias[i]; // Add bias term
    }

    return output;
}
//Final Classification Softmax Layer
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probabilities(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end()); // For numerical stability
    float sum_exp = 0.0f;

    for (float logit : logits) {
        sum_exp += std::exp(logit - max_logit);
    }

    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit) / sum_exp;
    }

    return probabilities;
}
#include <fstream>

// Save function for the final layer
void save_final_layer_weights(const std::vector<std::vector<float>>& weights, const std::vector<float>& bias) {
    std::ofstream file("final_layer_weight.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file final_layer_weight.bin for saving." << std::endl;
        return;
    }

    // Save weights
    for (const auto& row : weights) {
        file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }

    // Save bias
    file.write(reinterpret_cast<const char*>(bias.data()), bias.size() * sizeof(float));
    file.close();
    std::cout << "Final layer weights saved to final_layer_weight.bin." << std::endl;
}

// Load function for the final layer
bool load_final_layer_weights(std::vector<std::vector<float>>& weights, std::vector<float>& bias) {
    std::ifstream file("final_layer_weight.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open file final_layer_weight.bin for loading. Falling back to random initialization." << std::endl;
        return false;
    }

    // Load weights
    for (auto& row : weights) {
        file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
    }

    // Load bias
    file.read(reinterpret_cast<char*>(bias.data()), bias.size() * sizeof(float));
    file.close();
    std::cout << "Final layer weights loaded from final_layer_weight.bin." << std::endl;
    return true;
}
void print_float_vector_1D(std::vector<float> float_vector_1D)
{
    for (float val : float_vector_1D)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
void print_float_vector_2D(std::vector<std::vector<float>> float_vector_2D)
{
    for (const auto &row : float_vector_2D)
    {
        for (float val : row)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

void print_out_probabilities(std::vector<float> probabilities, std::vector<int> padded_input)
{
    // Print probabilities for debugging
    std::cout << "Input: ";
    for (int token : padded_input)
    {
        std::cout << token << " ";
    }
    std::cout << "\nProbabilities: ";
    print_float_vector_1D(probabilities);
}


int main() {
    
    bool load_parameters_yes_no = false;

#ifdef TEST_FEEDFORWARD_TRAIN
    // -------------------------------------------------------------
    // 1) Setup: Create two feed-forward layers
    // -------------------------------------------------------------
    //bool load_parameters_yes_no = false; // For this test, random init is fine
    int layer_index_1 = 0; 
    int layer_index_2 = 1; 
    int layer_index_3 = 2;
    int layer_index_4 = 3; 
    // Typically defined in config.h:
    GLOBAL_learning_rate = 0.001f;
    GLOBAL_momentum      = 0.9f;

    FeedForward ff1(/*d_model=*/2, /*d_ff=*/50, load_parameters_yes_no, layer_index_1);
    FeedForward ff2(/*d_model=*/2, /*d_ff=*/50, load_parameters_yes_no, layer_index_2);
    FeedForward ff3(/*d_model=*/2, /*d_ff=*/50, load_parameters_yes_no, layer_index_3);
    FeedForward ff4(/*d_model=*/2, /*d_ff=*/50, load_parameters_yes_no, layer_index_4);

    // -------------------------------------------------------------
    // 2) More advanced training data
    //    We'll generate random (x1, x2) in some range, and define
    //    the target as [sin(x1 + x2), x1^2 - x2^2].
    // -------------------------------------------------------------
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;

    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int num_samples = 1000;
    for (int i = 0; i < num_samples; ++i) {
        float x1 = dist(rng);
        float x2 = dist(rng);

        float t1 = std::sin(x1 + x2);   // Example nonlinear function
        float t2 = (x1 * x1) - (x2 * x2);

        inputs.push_back({x1, x2});
        targets.push_back({t1, t2});
    }

    // -------------------------------------------------------------
    // 3) Training loop
    // -------------------------------------------------------------
    int epochs_test = 200;
    for (int epoch = 1; epoch <= epochs_test; ++epoch)
    {
        float total_loss = 0.0f;

        // In a real setting, you could shuffle the data each epoch
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            // Wrap the single example into shape [1 x 2]
            std::vector<std::vector<float>> x_input = { inputs[i] };
            std::vector<std::vector<float>> y_target = { targets[i] };

            // Forward pass
            auto out1 = ff1.forward(x_input); // [1 x 2]
            auto out2 = ff2.forward(out1);    // [1 x 2]
            auto out3 = ff3.forward(out2);    // [1 x 2]
            auto out4 = ff4.forward(out3);    // [1 x 2]

            // Compute MSE loss
            float loss = mse_loss(out4, y_target);
            total_loss += loss;

            // Compute gradient of MSE w.r.t. out2
            auto grad_out2 = mse_loss_grad(out4, y_target);
            auto grad_ff4 = ff4.backward(grad_out2); 
            auto grad_ff3 = ff3.backward(grad_ff4);           
            auto grad_ff2 = ff2.backward(grad_ff3);
            // Backprop through first feedforward
            auto grad_ff1 = ff1.backward(grad_ff2);

            // Update weights
            ff4.update_weights();
            ff3.update_weights();                        
            ff2.update_weights();
            ff1.update_weights();
        }

        float avg_loss = total_loss / inputs.size();
        if (epoch < 50) {
            std::cout << "Epoch " << epoch
                      << " - MSE Loss: " << avg_loss << std::endl;
        }        
        if (epoch % 50 == 0) {
            std::cout << "Epoch " << epoch
                      << " - MSE Loss: " << avg_loss << std::endl;
        }
    }

    // -------------------------------------------------------------
    // 4) Test final results
    // -------------------------------------------------------------
    std::cout << "\n=== After Training, check predictions ===\n";
    for (int i = 0; i < 5; ++i) // Print first 5 random examples
    {
        std::vector<std::vector<float>> x_input = { inputs[i] };
        auto out1 = ff1.forward(x_input); // [1 x 2]
        auto out2 = ff2.forward(out1);    // [1 x 2]
        auto out3 = ff3.forward(out2);    // [1 x 2]
        auto out4 = ff4.forward(out3);    // [1 x 2]

        std::cout << "x1=" << inputs[i][0] 
                  << ", x2=" << inputs[i][1]
                  << " -> Prediction: ["
                  << out4[0][0] << ", " << out4[0][1] << "]"
                  << " | Target: ["
                  << targets[i][0] << ", " << targets[i][1] << "]\n";
    }

#endif // TEST_FEEDFORWARD_TRAIN

    cout << "========================================================================================================" << endl;
    cout << "Transformer Test in Mini Format (C/C++) - No Use of ML Libraries" << endl;
    cout << "The goal is to build and understand the Transformer algorithm from scratch using pure C++." << endl;
    cout << "========================================================================================================" << endl;
    cout << endl;


    // ----------------------------------------------------------------
    // Step 1: Load vocabulary from file
    // ----------------------------------------------------------------
    std::unordered_map<std::string, int> vocab;
    std::string vocab_file = "vocab.txt";
    if (!load_vocab_from_file(vocab_file, vocab)) {
        std::cerr << "Failed to load vocab from: " << vocab_file << std::endl;
        return -1;
    }

    // Optional: Print out some vocab entries
    // for (auto &kv : vocab) {
    //     std::cout << kv.first << " -> " << kv.second << std::endl;
    // }

    // ----------------------------------------------------------------
    // Step 2: Prepare dataset from question.txt and answer.txt
    // ----------------------------------------------------------------
    std::vector<std::vector<int>> dataset_2D;
    std::vector<int> labels;
    std::string question_file = "question.txt";
    std::string answer_file   = "answer.txt";
    if (!prepare_dataset_from_files(question_file, answer_file, dataset_2D, labels, vocab)) {
        std::cerr << "Failed to load dataset from: " << question_file 
                  << " and " << answer_file << std::endl;
        return -1;
    }


    // ----------------------------------------------------------------
    // Then continue your existing logic...
    //   - create the Transformer
    //   - define final layer weights
    //   - run training loop, etc.
    // ----------------------------------------------------------------


    std::cout << "Do you want to load an existing model parameter with embedding matrix from a file? (Y/N, y/n): ";
    std::string choice;
    std::cin >> choice;

     if (choice == "/Y" || choice == "y")
    {
        load_parameters_yes_no = true; // Load from file
    }
    else
    {
        load_parameters_yes_no = false; // Random initialization
    }

#ifdef TEST_FEEDFORWARD
    cout << "==================== Test: FeedForward ====================\n";

    // Define dimensions for the test
    int d_model_ffd_test = 4;  // Input dimensionality
    int d_ff_ffd_test = 6;     // Hidden layer dimensionality
    int layer_index_ffd_test = 0;

    // Create a FeedForward object
    FeedForward feed_forward(d_model_ffd_test, d_ff_ffd_test, load_parameters_yes_no, layer_index_ffd_test);

    // Define a small input matrix (e.g., 2 tokens with d_model dimensions)
    std::vector<std::vector<float>> input_ffd_test = {
        {1.0, 2.0, 3.0, 4.0},
        {0.5, 0.6, 0.7, 0.8}
    };

    cout << "input_ffd_test matrix (shape: " << input_ffd_test.size() << " x " << input_ffd_test[0].size() << "):\n";
    Utils::print_matrix(input_ffd_test);

    // Forward pass through the FeedForward network
    auto output_ffd_test = feed_forward.forward(input_ffd_test);

    // Print the output_ffd_test
    cout << "Output matrix (shape: " << output_ffd_test.size() << " x " << output_ffd_test[0].size() << "):\n";
    Utils::print_matrix(output_ffd_test);

    cout << "==========================================================\n";
#endif

#ifdef PRINT_OUT_TEST_ATTENTION_FORWARD_OPERATION
    // Make some PRINT_OUT_TEST_ATTENTION_FORWARD_OPERATION with small input matrix tests 
    // to understand one single layer of attention head in operation.
    cout << "==================== Test: Single Attention Layer ====================\n";

    // Input matrices
    std::vector<std::vector<float>> Q = {
        {1.0, 0.5, 0.1, 0.01}, // Query vector for token 1
        {0.2, 1.3, 0.2, 0.02}, // Query vector for token 2
        {1.2, 2.3, 3.2, 4.11}  // Query vector for token 3
    };
    std::vector<std::vector<float>> K = {
        {0.8, 0.3, 0.3, 0.03}, // Key vector for token 1
        {0.1, 0.9, 0.4, 0.04}, // Key vector for token 2
        {0.2, 0.3, 3.0, 1.11}  // Key vector for token 3
    };
    std::vector<std::vector<float>> V = {
        {1.2, 0.7, 0.5, 0.05}, // Value vector for token 1
        {0.5, 0.4, 0.6, 0.06}, // Value vector for token 2
        {2.2, 1.3, 0.0, 3.11}  // Value vector for token 3
    };

    // Check matrix sizes using Utils
    Utils::check_matrices(Q, K, V);

    // Simplified test setup
    
    int d_model_test = Q[0].size();               // Total embedding dimension
    int num_heads_test = 1;                  // Number of attention heads
    int d_k = d_model_test / num_heads_test; // Dimensionality of Q and K
    int d_v = d_model_test / num_heads_test; // Dimensionality of V

    cout << "The resolution of the positional encoding and embedding space, d_model: " << d_model_test << endl;
    MultiHeadAttention attention_layer_printout(d_model_test, 1, false, 0); // d_model=4, num_heads=1, no load, layer_index=0

    cout << "\n=== Relationship Between d_model, num_heads, and Matrix Dimensions ===\n";
    cout << "d_model (total embedding dimension): " << d_model_test << "\n";
    cout << "num_heads (number of attention heads): " << num_heads_test << "\n";
    cout << "d_k (key/query dimension per head): " << d_k << "\n";
    cout << "d_v (value dimension per head): " << d_v << "\n";

    cout << "\nExplanation:\n";
    cout << "- The total embedding dimension (d_model) is divided among all attention heads.\n";
    cout << "- With num_heads = 1, each head gets the full d_model, so d_k = d_model / num_heads = " << d_k << ".\n";
    cout << "- Similarly, d_v = d_model / num_heads = " << d_v << ".\n";
    cout << "In this case, each token is represented with " << d_k << " dimensions in Q and K, and "
         << d_v << " dimensions in V.\n";

    cout << "\n=== Hard coded Test Input Matrices ===\n";

    // Print matrices
    cout << "\nInput Q (Query):\n";
    Utils::print_matrix(Q);
    cout << "Each row represents a token, and each column represents one of the " << d_k << " dimensions of the query vector.\n";

    cout << "\nInput K (Key):\n";
    Utils::print_matrix(K);
    cout << "Each row represents a token, and each column represents one of the " << d_k << " dimensions of the key vector.\n";

    cout << "\nInput V (Value):\n";
    Utils::print_matrix(V);
    cout << "Each row represents a token, and each column represents one of the " << d_v << " dimensions of the value vector.\n";

    cout << "\nSummary:\n";
    cout << "- Q and K have " << d_k << " columns because they encode positional and content-related similarities.\n";
    cout << "- V has " << d_v << " columns because it contains the actual token content to be weighted and combined.\n";
    cout << "=====================================================================\n";
    
    // Call scaled_dot_product_attention_with_printout for testing
    auto attention_output_printout = attention_layer_printout.scaled_dot_product_attention_with_printout(Q, K, V);

    cout << "=====================================================================\n";

#else
    int length = 0;
    // Display tokenized sentences and their labels
    std::cout << "Tokenized Dataset:\n";
    for (size_t i = 0; i < dataset_2D.size(); ++i) {
        std::cout << (labels[i] == 0 ? "Question: " : "Answer: ");
        int token_cnt = 0;
        for (int token : dataset_2D[i]) {
            std::cout << token << " ";
            token_cnt++;
            if(length < token_cnt)
            {
                length = token_cnt;
            }
        }
        std::cout << "\n";
    }
    cout << "token_cnt length: " << length << endl;
    // Define parameters
    int vocab_size = 5000;
    int d_model = 128; // The "resolution" of the positional encoding and embedding space. 
                    // Think of it like a meter stick with 128 evenly spaced lines: 
                    // this determines how finely the meaning of a token can be represented
                    // across multiple dimensions.
                    //
                    // Each token (word or sub-word) is not just an isolated entity but carries 
                    // a representation that heavily depends on its position and relationships 
                    // to other tokens in the context. For example, the word "bank" could 
                    // mean "riverbank" or "financial bank," and its meaning is influenced 
                    // by neighboring words.
                    //
                    // In this context, "d_model" defines the number of dimensions (features) 
                    // used to represent these relationships. Higher d_model provides a finer 
                    // "resolution," allowing the model to encode more complex interactions 
                    // and associations across the sequence. 
                    //
                    // Increasing d_model expands the range of nuances and relationships that 
                    // the model can capture, enabling it to differentiate subtle differences 
                    // in meaning based on positional and contextual variations in the input 
                    // sequence.
                    //
                    // However, higher d_model also increases computational complexity and 
                    // the risk of overfitting for small datasets, so a balance is needed.

    int num_heads = 1;// 8
    int d_ff = 256;   // d_ff: Dimensionality of the hidden layer in the feed-forward network.
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
    int num_layers = 6;
   // int max_len = length; //64  Maximum sequence length (number of tokens in a single input)
    int max_len = 25;
#ifdef TEST_UTILS

    cout << "Test utils functions here: " << endl;

    // test utils funcftions
    // Test 1: Matrix Multiplication
    vector<vector<float>> mat1 = {{10.0, 20.0, 30.0}, {40.0, 50.0, 60.0}};
    vector<vector<float>> mat2 = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    vector<vector<float>> matmul_result = Utils::matmul(mat1, mat2);

    cout << "\nMatrix Multiplication Result:" << endl;
    for (const auto &row : matmul_result)
    {
        for (float val : row)
        {
            cout << val << " ";
        }
        cout << endl;
    }

    // Test 2: Matrix Transpose
    vector<vector<float>> transpose_result = Utils::transpose(mat1);

    cout << "\nMatrix Transpose Result:" << endl;
    for (const auto &row : transpose_result)
    {
        for (float val : row)
        {
            cout << val << " ";
        }
        cout << endl;
    }

    // Test 3: Softmax
    vector<float> logits = {2.0, 1.0, 0.1};
    vector<float> softmax_result = Utils::softmax(logits);

    cout << "\nSoftmax Result:" << endl;
    for (float val : softmax_result)
    {
        cout << val << " ";
    }
    cout << endl;

//
#endif
#ifdef TEST_ATTENTION

   cout << "Testing MultiHeadAttention functions..." << endl;

    // Test Inputs
    vector<vector<float>> query = {{1.0, 0.0}, {0.0, 1.0}};
    vector<vector<float>> key = {{1.0, 2.0}, {0.0, 3.0}};
    vector<vector<float>> value = {{4.0, 5.0}, {6.0, 7.0}};
    // Padding mask
    std::vector<int> padding_mask_test = {1, 1, 1};

    // Initialize MultiHeadAttention with 2 dimensions and 1 head (simplest case)
    MultiHeadAttention attention(2, 1, load_parameters_yes_no, num_layers);

    // Manually set weights for testing (simplified identity weights)
    attention.weights_q = {{1.0, 0.0}, {0.0, 1.0}};
    attention.weights_k = {{1.0, 0.0}, {0.0, 1.0}};
    attention.weights_v = {{1.0, 0.0}, {0.0, 1.0}};

    // Test Scaled Dot-Product Attention
    cout << "\nTesting Scaled Dot-Product Attention:" << endl;
    auto attention_output = attention.scaled_dot_product_attention(query, key, value, padding_mask_test);

    cout << "Attention Output:" << endl;
    for (const auto& row : attention_output) {
        for (float val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    // Test Full Forward Pass
    cout << "\nTesting Full Forward Pass:" << endl;
    auto forward_output = attention.forward(query, key, value, padding_mask_test);

    cout << "Forward Output:" << endl;
    for (const auto& row : forward_output) {
        for (float val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
#endif
/*
    // Define a simple vocabulary
    std::unordered_map<std::string, int> vocab = {
        {"[PAD]", 0}, {"[UNK]", 1}, {"what", 2}, {"time", 3}, {"is", 4}, {"it", 5}, {"now", 6},
        {"how", 7}, {"are", 8}, {"you", 9}, {"doing", 10}, {"today", 11},
        {"can", 12}, {"help", 13}, {"me", 14}, {"with", 15}, {"this", 16},
        {"where", 17}, {"the", 18}, {"nearest", 19}, {"bus", 20}, {"stop", 21},
        {"why", 22}, {"sky", 23}, {"blue", 24}, {"who", 25}, {"wrote", 26},
        {"book", 27}, {"which", 28}, {"movie", 29}, {"do", 30}, {"recommend", 31},
        {"when", 32}, {"will", 33}, {"meeting", 34}, {"start", 35}, {"going", 36},
        {"to", 37}, {"rain", 38}, {"could", 39}, {"explain", 40}, {"that", 41},
        {"again", 42}, {"three", 43}, {"oclock", 44}, {"am", 45}, {"well", 46},
        {"thank", 47}, {"yes", 48}, {"i", 49}, {"light", 50}, {"scattering", 51},
        {"jane", 52}, {"austen", 53}, {"inception", 54}, {"ten", 55},
        {"minutes", 56}, {"sure", 57}, {"later", 58}
    };
    if (!Utils::check_vocabs(vocab)) {
        std::cerr << "Vocabulary validation failed.\n";
        return 1; // Exit with error
    }
    std::cout << "Vocabulary validation succeeded.\n";
    // Prepare the dataset
    std::vector<std::vector<int>> dataset_2D;
    std::vector<int> labels;
  //  prepare_dataset(dataset_2D, labels, vocab);
*/

    // ================== Set up the transformer ==================
    vocab_size = vocab.size(); // Dynamically set to the actual vocabulary size
    cout << "vocab_size = " << vocab_size << endl;

    d_model = 25;
    d_ff = 100;//24

    // Initialize final layer weights and bias
    int num_categories = 2; // Number of output categories (Question/Answer)

    std::vector<std::vector<float>> final_weights(d_model, std::vector<float>(num_categories, 0.0f));
    std::vector<float> final_bias(num_categories, 0.0f);

    if (load_parameters_yes_no) {
        if (!load_final_layer_weights(final_weights, final_bias)) {
            // Fall back to random initialization if loading fails
            std::srand(std::time(0));
            for (auto& row : final_weights) {
                for (auto& val : row) {
                    val = static_cast<float>(std::rand()) / RAND_MAX; // Random values between 0 and 1
                }
            }
            for (auto& val : final_bias) {
                val = static_cast<float>(std::rand()) / RAND_MAX;
            }
        }
    } else {
        // Random initialization
        std::srand(std::time(0));
        for (auto& row : final_weights) {
            for (auto& val : row) {
                val = static_cast<float>(std::rand()) / RAND_MAX;
            }
        }
        for (auto& val : final_bias) {
            val = static_cast<float>(std::rand()) / RAND_MAX;
        }
    }

    // Create transformer
    Transformer transformer(vocab_size, d_model, max_len, num_heads, d_ff, num_layers, load_parameters_yes_no);

    // ============== Training loop ===================
    int epochs = 200;
    // Initialize velocity for weights and bias
    std::vector<std::vector<float>> velocity_weights(final_weights.size(),
                                                     std::vector<float>(final_weights[0].size(), 0.0f));
    std::vector<float> velocity_bias(final_bias.size(), 0.0f);
    const float warm_up_factor = 1.0;
    const int warm_up_epc_cnt = 1;
    float GLOBAL_learning_rate = GLOBAL_CONST_learning_rate * warm_up_factor;
    float GLOBAL_momentum = GLOBAL_CONST_momentum * warm_up_factor;
    std::cout << "learning_rate: " << GLOBAL_learning_rate << std::endl;
    std::cout << "momentum: " << GLOBAL_momentum << std::endl;
    // Training loop with gradient computation
 //   const int save_after_epc = 10;
         int save_epc_cnt = 0;
    for (int epoch = 1; epoch <= epochs; ++epoch)
    {
        if(epoch == warm_up_epc_cnt)
        {
            GLOBAL_learning_rate = GLOBAL_CONST_learning_rate;
            GLOBAL_momentum = GLOBAL_CONST_momentum;
            std::cout << "Set normal learning rate after warm up " << std::endl;
            std::cout << "learning_rate: " << GLOBAL_learning_rate << std::endl;
            std::cout << "momentum: " << GLOBAL_momentum << std::endl;            
        }
        std::cout << "Epoch " << epoch << " / " << epochs << "\n";
        // Shuffle dataset
        fisher_yates_shuffle(dataset_2D, labels);
        float epoch_loss = 0.0f; // Accumulate loss for the epoch

        for (size_t i = 0; i < dataset_2D.size(); ++i)
        {

            // Prepare input and padding mask
           // auto padded_input = pad_sequence(dataset_2D[i], max_len);
            auto trunc_sequence = truncate_tokens_max_len(dataset_2D[i], max_len);
            auto padding_mask = create_padding_mask(trunc_sequence, max_len);
#ifdef DEBUG_PRINT_MAIN            
            std::cout << "Padding Input:" << std::endl;
            for (int input : padded_input) {
                std::cout << input << " ";
            }
            std::cout << std::endl;

            std::cout << "Padding Mask:" << std::endl;
            for (int mask : padding_mask) {
                std::cout << mask << " ";
            }
            std::cout << std::endl;
#endif
            // Forward pass through transformer
            auto output_trans = transformer.forward(trunc_sequence, padding_mask);
#ifdef DEBUG_PRINT_MAIN 
            std::cout << "Transformer Output Before Pooling:" << std::endl;
            print_float_vector_2D(output_trans);
#endif
            // Reduce transformer output (e.g., by mean pooling)
            std::vector<float> pooled_output = mean_pooling(output_trans);
#ifdef DEBUG_PRINT_MAIN             
            std::cout << "Pooled Output (Input to Classification Layer):" << std::endl;
            print_float_vector_1D(pooled_output);
#endif

            // Apply final classification layer
            std::vector<float> logits = linear_layer(pooled_output, final_weights, final_bias);
            std::vector<float> probabilities = softmax(logits);
            // Backpropagation starts here
            // Step 1: Compute gradient of loss with respect to logits
            std::vector<float> grad_logits = cross_entropy_loss_gradient(probabilities, labels[i]);
            // Step 2: Compute gradients for final weights and bias
            std::vector<std::vector<float>> grad_final_weights(final_weights.size(),
                                                               std::vector<float>(final_weights[0].size(), 0.0f));
            std::vector<float> grad_final_bias(final_bias.size(), 0.0f);

            for (size_t j = 0; j < pooled_output.size(); ++j)
            {
                for (size_t k = 0; k < grad_logits.size(); ++k)
                {
                    grad_final_weights[j][k] += pooled_output[j] * grad_logits[k];
                }
            }

            for (size_t k = 0; k < grad_logits.size(); ++k)
            {
                grad_final_bias[k] += grad_logits[k];
            }

            // Step 3: Update final weights and bias using SGD with momentum
            for (size_t j = 0; j < final_weights.size(); ++j)
            {
                for (size_t k = 0; k < final_weights[0].size(); ++k)
                {
                    velocity_weights[j][k] = GLOBAL_momentum * velocity_weights[j][k] - GLOBAL_learning_rate * grad_final_weights[j][k];
                    final_weights[j][k] += velocity_weights[j][k];
                }
            }

            for (size_t k = 0; k < final_bias.size(); ++k)
            {
                velocity_bias[k] = GLOBAL_momentum * velocity_bias[k] - GLOBAL_learning_rate * grad_final_bias[k];
                final_bias[k] += velocity_bias[k];
            }

            // Compute gradient of final layer with respect to mean pooling output
            std::vector<float> pooled_output_gradient(pooled_output.size(), 0.0f);
            for (size_t i = 0; i < final_weights.size(); ++i) {
                for (size_t j = 0; j < grad_logits.size(); ++j) {
                    pooled_output_gradient[i] += grad_logits[j] * final_weights[i][j];
                }
            }

            // Backpropagate through mean pooling
            std::vector<std::vector<float>> grad_pooled = mean_pooling_backward(output_trans, pooled_output_gradient);

#ifdef DEBUG_PRINT_MAIN    
            // Print gradient for debugging
            std::cout << "Gradient w.r.t Mean Pooling Input:" << std::endl;
            print_float_vector_2D(grad_pooled);
#endif


           // Pick a layer, matrix, row, col to observe
           int layer_idx = 0;
           std::string which_matrix = "Q"; // or "K"/"V"
           int row = 2;
           int col = 3;

           float weight_before = transformer.read_attention_weight(layer_idx, which_matrix, row, col);
            // Backpropagate gradient through the Transformer 
           transformer.backward(grad_pooled);

           float weight_after = transformer.read_attention_weight(layer_idx, which_matrix, row, col);

          // std::cout << "Weight before: " << weight_before
          //           << ", after: " << weight_after << std::endl;

           // print_out_probabilities(probabilities, padded_input);// Print probabilities for debugging
           //  Compute loss and accumulate
           float loss = cross_entropy_loss(probabilities, labels[i]);
           epoch_loss += loss;


        }
           if(save_epc_cnt < 2)
           {
                save_epc_cnt++;
           }
           else
           {

GLOBAL_ATTENTION_learning_rate = GLOBAL_learning_rate;
GLOBAL_ATTENTION_momentum = GLOBAL_momentum;
GLOBAL_learning_rate = 0.0;
GLOBAL_momentum = 0.0;
            std::cout << "Set normal learning rate after warm up " << std::endl;
            std::cout << "learning_rate: " << GLOBAL_learning_rate << std::endl;
            std::cout << "momentum: " << GLOBAL_momentum << std::endl;   
                save_epc_cnt = 0;
                save_final_layer_weights(final_weights, final_bias);
                transformer.save_embedding_matrix();
                transformer.save_attention_weights();
                transformer.save_feed_forward_weights();    
                transformer.save_LayerNormalization_weights();                
           }
           
        // Print average loss for the epoch
        std::cout << "Average Loss for Epoch " << epoch << ": " << (epoch_loss / dataset_2D.size()) << "\n";
    }
    //========================== End training loop ===================

    // Save final layer weights (optional)
    save_final_layer_weights(final_weights, final_bias);
    transformer.save_embedding_matrix();
    transformer.save_attention_weights();
    transformer.save_feed_forward_weights();    
    transformer.save_LayerNormalization_weights();

    cout << "debug 1" << endl;
#endif

    return 0;

}
