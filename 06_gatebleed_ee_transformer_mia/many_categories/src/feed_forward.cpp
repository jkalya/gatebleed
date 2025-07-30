#include "feed_forward.h"
#include "utils.h"
const std::string FeedForward::file_prefix_feed_forward_weights = "ffd_weight_layer_";
FeedForward::FeedForward(int d_model, int d_ff, bool load_parameters_yes_no, int layer_index)
    : velocity_weights1(d_model, std::vector<float>(d_ff, 0.0f)),
    velocity_weights2(d_ff, std::vector<float>(d_model, 0.0f)),
    input_activations(d_model, std::vector<float>(d_ff, 0.0f)),
    hidden_activations(d_ff, std::vector<float>(d_model, 0.0f)),
    weights1(d_model, std::vector<float>(d_ff, 0.0f)),
    weights2(d_ff, std::vector<float>(d_model, 0.0f)),
    grad_weights1(d_model, std::vector<float>(d_ff, 0.0f)),
    grad_weights2(d_ff, std::vector<float>(d_model, 0.0f)),
    // Now for bias
    bias1(d_ff, 0.0f),
    bias2(d_model, 0.0f),
    grad_bias1(d_ff, 0.0f),
    grad_bias2(d_model, 0.0f),
    velocity_bias1(d_ff, 0.0f),
    velocity_bias2(d_model, 0.0f)      

{
    const std::string weights_file = file_prefix_feed_forward_weights + std::to_string(layer_index) + ".bin";
    bool loaded = false;

    if (load_parameters_yes_no) {
        std::ifstream file(weights_file, std::ios::binary);
        if (file.is_open()) {
            // Load weights1
            for (auto& row : weights1) {
                file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
            }
            // Load weights2
            for (auto& row : weights2) {
                file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
            }
            // Similarly for loading
            for (float &b : bias1) {
                file.read(reinterpret_cast<char*>(&b), sizeof(float));
            }
            for (float &b : bias2) {
                file.read(reinterpret_cast<char*>(&b), sizeof(float));
            }


            file.close();
            std::cout << "FeedForward weights for layer " << layer_index << " loaded from file.\n";
            loaded = true;
        } else {
            std::cerr << "Warning: Could not open " << weights_file << ". Falling back to random initialization.\n";
        }
    }

    if (!loaded) {
        std::srand(std::time(0));
        float scale1 = std::sqrt(2.0f / d_model);
        float scale2 = std::sqrt(2.0f / d_ff);

        // Randomly initialize weights1
        for (auto& row : weights1) {
            for (auto& val : row) {
                val = scale1 * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f;
            }
        }

        // Randomly initialize weights2
        for (auto& row : weights2) {
            for (auto& val : row) {
                val = scale2 * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f;
            }
        }

         // Randomly initialize bias1
        for (auto& val : bias1) {
            val = scale1 * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f;
        }
         // Randomly initialize bias2
        for (auto& val : bias2) {
            val = scale2 * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f;
        }                   
    }
}

void FeedForward::save_weights(int layer_index) {
    const std::string weights_file = file_prefix_feed_forward_weights + std::to_string(layer_index) + ".bin";
    std::ofstream file(weights_file, std::ios::binary);
    if (file.is_open()) {
        for (const auto& row : weights1) {
            file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
        }
        for (const auto& row : weights2) {
            file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
        }
        // After writing weights1 and weights2
        for (float b : bias1) {
            file.write(reinterpret_cast<const char*>(&b), sizeof(float));
        }
        for (float b : bias2) {
            file.write(reinterpret_cast<const char*>(&b), sizeof(float));
        }



        file.close();
        std::cout << "FeedForward weights for layer " << layer_index << " initialized and saved to file.\n";
    } else {
        std::cerr << "Error: Could not save FeedForward weights to file.\n";
        exit(EXIT_FAILURE);
    }
}

std::vector<std::vector<float>> FeedForward::forward(const std::vector<std::vector<float>>& input) {
    input_activations = input; // Cache input for backpropagation

    // Step 1: Linear transformation with weights1
    hidden_activations = Utils::matmul(input, weights1);
    // Add bias1 along the columns
    // hidden_activations shape: [batch_size x d_ff]
    for (auto &row : hidden_activations)
    {
        for (int j = 0; j < (int)row.size(); j++)
        {
            row[j] += bias1[j];
        }
    }

    // Step 2: Apply ReLU activation
    for (auto &row : hidden_activations)
    {
        for (auto &val : row)
        {
            val = Utils::leaky_relu(val);
        }
    }

    // Step 3: Linear transformation with weights2 + bias2
    auto out = Utils::matmul(hidden_activations, weights2);
    for (auto &row : out)
    {
        for (int j = 0; j < (int)row.size(); j++)
        {
            row[j] += bias2[j];
        }
    }
    return out;
}

std::vector<std::vector<float>> FeedForward::backward(const std::vector<std::vector<float>>& grad_output) {
    // 1) grad_hidden = grad_output * W2^T
    auto weights2_transposed = Utils::transpose(weights2);
    auto grad_hidden = Utils::matmul(grad_output, weights2_transposed);

    // 2) Multiply by derivative of Leaky ReLU
    for (size_t i = 0; i < grad_hidden.size(); ++i) {
        for (size_t j = 0; j < grad_hidden[i].size(); ++j) {
            grad_hidden[i][j] *= Utils::leaky_relu_derivative(hidden_activations[i][j]);
        }
    }
    // gradient for bias2
    //   Each bias2[j] = sum of grad_output[:, j] across the batch dimension
    std::fill(grad_bias2.begin(), grad_bias2.end(), 0.0f);
    for (size_t i = 0; i < grad_output.size(); ++i)
    { // over batch
        for (size_t j = 0; j < grad_output[i].size(); ++j)
        { // over output dim
            grad_bias2[j] += grad_output[i][j];
        }
    }
    // gradient for bias1
    //   Each bias1[j] = sum of grad_hidden[:, j] across the batch dimension
    std::fill(grad_bias1.begin(), grad_bias1.end(), 0.0f);
    for (size_t i = 0; i < grad_hidden.size(); ++i) {
        for (size_t j = 0; j < grad_hidden[i].size(); ++j) {
            grad_bias1[j] += grad_hidden[i][j];
        }
    }

    // ------------------------------------------------
    // GRADIENT for weights2
    //   grad_w2 = hidden_activations^T * grad_output
    // ------------------------------------------------
    grad_weights2 = Utils::matmul(Utils::transpose(hidden_activations), grad_output);
 
    // ------------------------------------------------
    // GRADIENT for weights1
    //   grad_w1 = input_activations^T * grad_hidden
    // ------------------------------------------------
    grad_weights1 = Utils::matmul(Utils::transpose(input_activations), grad_hidden);
 
    // ------------------------------------------------
    // Return gradient for the previous layer
    //   grad_input = grad_hidden * W1^T
    // ------------------------------------------------
    auto weights1_transposed = Utils::transpose(weights1);
    auto grad_input = Utils::matmul(grad_hidden, weights1_transposed);



    return grad_input;
}




void FeedForward::update_weights() {
    // Update weights2
    for (size_t i = 0; i < weights2.size(); ++i) {
        for (size_t j = 0; j < weights2[i].size(); ++j) {
            velocity_weights2[i][j] = GLOBAL_momentum * velocity_weights2[i][j]
                + GLOBAL_learning_rate * grad_weights2[i][j];
            weights2[i][j] -= velocity_weights2[i][j];
        }
    }
    
    // Update weights1
    for (size_t i = 0; i < weights1.size(); ++i) {
        for (size_t j = 0; j < weights1[i].size(); ++j) {
            velocity_weights1[i][j] = GLOBAL_momentum * velocity_weights1[i][j]
                + GLOBAL_learning_rate * grad_weights1[i][j];
            weights1[i][j] -= velocity_weights1[i][j];
        }
    }
    
    // Update bias2
    for (size_t i = 0; i < bias2.size(); ++i) {
        velocity_bias2[i] = GLOBAL_momentum * velocity_bias2[i] 
            + GLOBAL_learning_rate * grad_bias2[i];
        bias2[i] -= velocity_bias2[i];
    }
    
    // Update bias1
    for (size_t i = 0; i < bias1.size(); ++i) {
        velocity_bias1[i] = GLOBAL_momentum * velocity_bias1[i] 
            + GLOBAL_learning_rate * grad_bias1[i];
        bias1[i] -= velocity_bias1[i];
    }
}



