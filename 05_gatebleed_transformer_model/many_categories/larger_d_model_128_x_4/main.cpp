#include "transformer.h"
#include <iostream>
#include "dataset.h"
#include <vector>
#include "utils.h"
#include <unordered_map>
#include <fstream>
#include <filesystem>
using namespace std;
#include "config.h"
#include <algorithm> // For Fisher-Yates shuffle
#include <random>    // For random number generation
#include <chrono>    // For seeding random number generator
#include <algorithm> // std::min

// 1) Add a helper function to copy *.bin files to ./best_model/
#include <filesystem> // For filesystem operations

void copy_best_model_files_to_directory(const std::string &target_dir)
{
    // Create the directory if it doesn't exist
    std::filesystem::create_directories(target_dir);

    // Iterate over all files in the current directory
    for (const auto &entry : std::filesystem::directory_iterator("."))
    {
        // We only want to copy *.bin files
        if (entry.is_regular_file() && entry.path().extension() == ".bin")
        {
            // Create destination path like: ./best_model/filename.bin
            auto dest = std::filesystem::path(target_dir) / entry.path().filename();
            // Copy with overwrite
            std::filesystem::copy_file(entry.path(), dest, std::filesystem::copy_options::overwrite_existing);
            // Optional: Print a small message
            std::cout << "Copied " << entry.path().filename()
                      << " to " << dest << std::endl;
        }
    }
}

// Cross-entropy loss gradient
std::vector<float> cross_entropy_loss_gradient(const std::vector<float> &probabilities, int label)
{
    std::vector<float> gradient(probabilities.size(), 0.0f);
    for (size_t i = 0; i < probabilities.size(); ++i)
    {
        gradient[i] = probabilities[i] - (i == static_cast<size_t>(label) ? 1.0f : 0.0f);
    }
    return gradient;
}

// Function to compute cross-entropy loss
float cross_entropy_loss(const std::vector<float> &probabilities, int label)
{
    return -std::log(probabilities[label] + 1e-9); // Add small epsilon for numerical stability
}

// Function to shuffle dataset
void fisher_yates_shuffle(std::vector<std::vector<int>> &dataset, std::vector<int> &labels)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);

    for (size_t i = dataset.size() - 1; i > 0; --i)
    {
        std::uniform_int_distribution<size_t> dist(0, i);
        size_t j = dist(rng);

        std::swap(dataset[i], dataset[j]);
        std::swap(labels[i], labels[j]);
    }
}

// Function to truncate tokens and pad to `max_len`
std::vector<int> truncate_tokens_max_len(const std::vector<int> &sequence, int max_len)
{
    // 1) Truncate if necessary
    std::vector<int> truncated(sequence.begin(),
                               sequence.begin() + std::min<size_t>(sequence.size(), max_len));

    // 2) If truncated.size() < max_len, pad with zeros
    if (truncated.size() < static_cast<size_t>(max_len))
    {
        truncated.resize(max_len, 0); // 0 = [PAD]
    }
    return truncated;
}

// Function to create padding mask
std::vector<int> create_padding_mask(const std::vector<int> &sequence, int max_len)
{
    std::vector<int> mask(max_len, 0);
    for (size_t i = 0; i < sequence.size(); ++i)
    {
        if (sequence[i] != 0)
        {
            mask[i] = 1;
        }
    }
    return mask;
}

// Mean Pooling
std::vector<float> mean_pooling(const std::vector<std::vector<float>> &output)
{
    std::vector<float> pooled(output[0].size(), 0.0f);
    for (const auto &row : output)
    {
        for (size_t i = 0; i < row.size(); ++i)
        {
            pooled[i] += row[i];
        }
    }
    for (float &val : pooled)
    {
        val /= output.size();
    }
    return pooled;
}

// Backward for Mean Pooling
std::vector<std::vector<float>> mean_pooling_backward(
    const std::vector<std::vector<float>> &output_from_transformer,
    const std::vector<float> &grad_pooled)
{
    size_t rows = output_from_transformer.size();
    size_t cols = output_from_transformer[0].size();
    std::vector<std::vector<float>> grad_output_to_transformer(rows, std::vector<float>(cols, 0.0f));

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            grad_output_to_transformer[i][j] = grad_pooled[j] / static_cast<float>(rows);
        }
    }
    return grad_output_to_transformer;
}

// Final Classification Layer
std::vector<float> linear_layer(const std::vector<float> &input,
                                const std::vector<std::vector<float>> &weights,
                                const std::vector<float> &bias)
{
    std::vector<float> output(weights[0].size(), 0.0f);
    for (size_t i = 0; i < weights[0].size(); ++i)
    { // For each output category
        for (size_t j = 0; j < input.size(); ++j)
        { // For each input dimension
            output[i] += input[j] * weights[j][i];
        }
        output[i] += bias[i]; // Add bias term
    }
    return output;
}

// Softmax
std::vector<float> softmax(const std::vector<float> &logits)
{
    std::vector<float> probabilities(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end()); // For numerical stability
    float sum_exp = 0.0f;

    for (float logit : logits)
    {
        sum_exp += std::exp(logit - max_logit);
    }
    for (size_t i = 0; i < logits.size(); ++i)
    {
        probabilities[i] = std::exp(logits[i] - max_logit) / sum_exp;
    }
    return probabilities;
}

// Save function for the final layer
void save_final_layer_weights(const std::vector<std::vector<float>> &weights,
                              const std::vector<float> &bias)
{
    std::ofstream file("final_layer_weight.bin", std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file final_layer_weight.bin for saving." << std::endl;
        return;
    }
    // Save weights
    for (const auto &row : weights)
    {
        file.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(float));
    }
    // Save bias
    file.write(reinterpret_cast<const char *>(bias.data()), bias.size() * sizeof(float));
    file.close();
    std::cout << "Final layer weights saved to final_layer_weight.bin." << std::endl;
}

// Load function for the final layer
bool load_final_layer_weights(std::vector<std::vector<float>> &weights,
                              std::vector<float> &bias)
{
    std::ifstream file("final_layer_weight.bin", std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Warning: Could not open file final_layer_weight.bin for loading. "
                     "Falling back to random initialization."
                  << std::endl;
        return false;
    }
    // Load weights
    for (auto &row : weights)
    {
        file.read(reinterpret_cast<char *>(row.data()), row.size() * sizeof(float));
    }
    // Load bias
    file.read(reinterpret_cast<char *>(bias.data()), bias.size() * sizeof(float));
    file.close();
    std::cout << "Final layer weights loaded from final_layer_weight.bin." << std::endl;
    return true;
}

// Debug printing
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

// Run interactive prompt mode
void run_prompt_mode(Transformer &transformer,
                     int max_len,
                     const std::unordered_map<std::string, int> &vocab,
                     const std::vector<std::vector<float>> &weights,
                     const std::vector<float> &bias,
                     const std::vector<std::string> &categories)
{
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // ensure clean input buffer
    std::string input;
    while (true)
    {
        std::cout << "\nEnter a string (or type 'exit' to quit): ";
        if (!std::getline(std::cin, input))
        {
            // Handle EOF or input error
            break;
        }
        if (input == "exit")
        {
            std::cout << "Exiting mini prompt mode.\n";
            break;
        }

        // 1) Tokenize input
        std::vector<int> tokens = tokenize(input, vocab);

        // Print the original token IDs
        std::cout << "Token IDs: ";
        for (int t : tokens)
        {
            std::cout << t << " ";
        }
        std::cout << "\n";

        // 2) Truncate/pad + create mask
        auto trunc_sequence = truncate_tokens_max_len(tokens, max_len);
        auto padding_mask = create_padding_mask(trunc_sequence, max_len);

        // Forward pass
        auto output = transformer.forward(trunc_sequence, padding_mask);
        std::vector<float> pooled_output = mean_pooling(output);
        std::vector<float> logits = linear_layer(pooled_output, weights, bias);
        std::vector<float> probabilities = softmax(logits);

        // 3) Print all categories in descending order of probability
        //    First, create a vector of (probability, index) pairs
        std::vector<std::pair<float, size_t>> cat_probs;
        cat_probs.reserve(probabilities.size());
        for (size_t i = 0; i < probabilities.size(); ++i)
        {
            cat_probs.emplace_back(probabilities[i], i);
        }

        // Sort by probability descending
        std::sort(cat_probs.begin(), cat_probs.end(),
                  [](const auto &a, const auto &b) {
                      return a.first > b.first; // highest first
                  });

        std::cout << "\nCategory probabilities (sorted highest to lowest):\n";
        for (auto &cp : cat_probs)
        {
            std::cout << categories[cp.second] << " : " << cp.first << "\n";
        }

        // 4) Print best (highest-prob) category explicitly
        auto max_iter = std::max_element(probabilities.begin(), probabilities.end());
        size_t max_index = std::distance(probabilities.begin(), max_iter);
        std::cout << "\nPredicted category: " << categories[max_index] << "\n";
    }
}

int main()
{
    bool load_parameters_yes_no = false;

    std::cout << "========================================================================================================\n";
    std::cout << "Transformer Test in Mini Format (C/C++) - No Use of ML Libraries\n";
    std::cout << "The goal is to build and understand the Transformer algorithm from scratch using pure C++.\n";
    std::cout << "========================================================================================================\n\n";

    // Step 1: Load vocabulary from ./vocab.txt
    std::unordered_map<std::string, int> vocab;
    std::string vocab_file = "vocab.txt";
    if (!load_vocab_from_file(vocab_file, vocab))
    {
        std::cerr << "Failed to load vocab from: " << vocab_file << std::endl;
        return -1;
    }

    // Step 2: Load categories from ./labels.txt
    std::vector<std::string> categories;
    std::string labels_file = "labels.txt";
    std::ifstream labels_stream(labels_file);
    if (!labels_stream.is_open())
    {
        std::cerr << "Error: Could not open labels file: " << labels_file << std::endl;
        return -1;
    }
    {
        std::string line;
        while (std::getline(labels_stream, line))
        {
            categories.push_back(line);
        }
    }
    labels_stream.close();

    if (categories.empty())
    {
        std::cerr << "Error: Labels file is empty.\n";
        return -1;
    }

    int num_categories = categories.size();
    std::cout << "Loaded " << num_categories << " categories from labels.txt.\n";

    // Build file list for training
    std::vector<std::string> train_input_files;
    // Build file list for verification
    std::vector<std::string> verify_input_files;

    // For each category, expect ./train/<category>.txt and ./verify/<category>.txt
    for (const auto &cat : categories)
    {
        std::string train_path = std::string("./train/") + cat + ".txt";
        if (!std::filesystem::exists(train_path))
        {
            std::cerr << "Error: Missing train file for category: " << train_path << std::endl;
            return -1;
        }
        train_input_files.push_back(train_path);

        std::string verify_path = std::string("./verify/") + cat + ".txt";
        if (!std::filesystem::exists(verify_path))
        {
            std::cerr << "Error: Missing verify file for category: " << verify_path << std::endl;
            return -1;
        }
        verify_input_files.push_back(verify_path);
    }

    // Prepare training dataset
    std::vector<std::vector<int>> train_dataset_2D;
    std::vector<int> train_labels;

    // Prepare verification dataset
    std::vector<std::vector<int>> verify_dataset_2D;
    std::vector<int> verify_labels;

    // Create label map from category string to index
    std::unordered_map<std::string, int> label_map;
    for (size_t i = 0; i < categories.size(); ++i)
    {
        label_map[categories[i]] = static_cast<int>(i);
    }

    // Fill training dataset
    if (!prepare_dataset_from_files(train_input_files, label_map, train_dataset_2D, train_labels, vocab))
    {
        std::cerr << "Failed to prepare training dataset.\n";
        return -1;
    }
    // Fill verification dataset
    if (!prepare_dataset_from_files(verify_input_files, label_map, verify_dataset_2D, verify_labels, vocab))
    {
        std::cerr << "Failed to prepare verification dataset.\n";
        return -1;
    }

    // Determine the max sequence length from entire training set, or set manually
    int length = 0;
    for (auto &seq : train_dataset_2D)
    {
        if (static_cast<int>(seq.size()) > length)
        {
            length = (int)seq.size();
        }
    }
    // Optionally check verify set as well, if you want the maximum to cover both sets
    for (auto &seq : verify_dataset_2D)
    {
        if (static_cast<int>(seq.size()) > length)
        {
            length = (int)seq.size();
        }
    }
    // For demonstration, we can pick a final max_len smaller or bigger.
    // Let's do a small cap for test:
    int max_len = 26; // or use: int max_len = length;
    std::cout << "Largest token sequence found in dataset(s): " << length << "\n";
    std::cout << "We will use max_len = " << max_len << "\n";

    // Ask user to load existing model or not
    std::cout << "Do you want to load an existing model parameter with embedding matrix? (Y/N): ";
    std::string choice;
    std::cin >> choice;
    if (choice == "Y" || choice == "y")
    {
        load_parameters_yes_no = true;
    }
    else
    {
        load_parameters_yes_no = false;
    }

    // Set your Transformer hyperparameters
    int d_model = 128;
    int num_heads = 4;
    int d_ff = 256;
    int num_layers = 6;

    // Create the Transformer
    int vocab_size = (int)vocab.size();
    std::cout << "vocab_size = " << vocab_size << "\n";
    
    Transformer transformer(vocab_size, d_model, num_heads, max_len, d_ff, num_layers, load_parameters_yes_no);
    std::cout << "vocab_size = " << vocab_size << "\n";
    std::cout << "d_model = " << d_model << "\n";
    std::cout << "max_len = " << max_len << "\n";
    std::cout << "d_ff = " << d_ff << "\n";
    std::cout << "num_heads = " << num_heads << "\n";
    std::cout << "num_layers = " << num_layers << "\n";
    // Initialize or load final layer
    std::vector<std::vector<float>> final_weights(d_model, std::vector<float>(num_categories, 0.0f));
    std::vector<float> final_bias(num_categories, 0.0f);

    if (load_parameters_yes_no)
    {
        if (!load_final_layer_weights(final_weights, final_bias))
        {
            // Fall back to random init
            std::srand((unsigned)std::time(nullptr));
            for (auto &row : final_weights)
            {
                for (auto &val : row)
                {
                    val = static_cast<float>(std::rand()) / RAND_MAX;
                }
            }
            for (auto &val : final_bias)
            {
                val = static_cast<float>(std::rand()) / RAND_MAX;
            }
        }
    }
    else
    {
        // Random initialization
        std::srand((unsigned)std::time(nullptr));
        for (auto &row : final_weights)
        {
            for (auto &val : row)
            {
                val = static_cast<float>(std::rand()) / RAND_MAX;
            }
        }
        for (auto &val : final_bias)
        {
            val = static_cast<float>(std::rand()) / RAND_MAX;
        }
    }

    // Prompt mode or train mode
    std::cout << "Do you want to start mini prompt mode? (Y/N): ";
    std::string response;
    std::cin >> response;
    if (response == "Y" || response == "y")
    {
        run_prompt_mode(transformer, max_len, vocab, final_weights, final_bias, categories);
    }
    else
    {
        // ================== Training loop ==================
        int epochs = 1000; // For illustration, set to 5. Adjust as needed
        std::vector<std::vector<float>> velocity_weights(d_model,
                                                         std::vector<float>(num_categories, 0.0f));
        std::vector<float> velocity_bias(num_categories, 0.0f);

        GLOBAL_learning_rate = 0.005f;
        GLOBAL_momentum = 0.92f;
        GLOBAL_ATTENTION_learning_rate = GLOBAL_learning_rate;
        GLOBAL_ATTENTION_momentum = GLOBAL_momentum;
        std::cout << "Training learning_rate: " << GLOBAL_learning_rate << "\n";
        std::cout << "Training momentum: " << GLOBAL_momentum << "\n";

        float best_avg_loss = 999999.0f;
        const int print_dot_interval = 10;
        int print_dot_cnt = 0;

        for (int epoch = 1; epoch <= epochs; ++epoch)
        {
            std::cout << "\n=== Epoch " << epoch << " / " << epochs << " ===\n";
            // Shuffle training data
            fisher_yates_shuffle(train_dataset_2D, train_labels);

            float epoch_loss = 0.0f;
            int correct_count_train = 0;

            // ------------------- TRAINING (forward + backward) -------------------
            // transformer.inference_mode = false;
            for (size_t i = 0; i < train_dataset_2D.size(); ++i)
            {
                // print dots to show progress
                if (print_dot_cnt < print_dot_interval)
                {
                    print_dot_cnt++;
                }
                else
                {
                    std::cout << "." << std::flush;
                    print_dot_cnt = 0;
                }

                // Truncate/pad + mask
                auto trunc_seq = truncate_tokens_max_len(train_dataset_2D[i], max_len);
                auto pad_mask = create_padding_mask(trunc_seq, max_len);

                // Forward
        
                auto output_trans = transformer.forward(trunc_seq, pad_mask);

                auto pooled_output = mean_pooling(output_trans);
                auto logits = linear_layer(pooled_output, final_weights, final_bias);
                auto probabilities = softmax(logits);

                // Check prediction
                int predicted_idx = (int)std::distance(
                    probabilities.begin(),
                    std::max_element(probabilities.begin(), probabilities.end()));
                if (predicted_idx == train_labels[i])
                {
                    correct_count_train++;
                }

                // Loss + Grad
                std::vector<float> grad_logits = cross_entropy_loss_gradient(probabilities, train_labels[i]);

                // Grad for final weights/bias
                std::vector<std::vector<float>> grad_final_weights(d_model,
                                                                   std::vector<float>(num_categories, 0.0f));
                std::vector<float> grad_final_bias(num_categories, 0.0f);

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

                // Update final weights (SGD + momentum)
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

                // Backprop into Transformer
                // 1) pooled_output gradient
                std::vector<float> pooled_output_gradient(pooled_output.size(), 0.0f);
                for (size_t dim = 0; dim < (size_t)d_model; ++dim)
                {
                    for (size_t cat_idx = 0; cat_idx < grad_logits.size(); ++cat_idx)
                    {
                        pooled_output_gradient[dim] += grad_logits[cat_idx] * final_weights[dim][cat_idx];
                    }
                }
                auto grad_pooled = mean_pooling_backward(output_trans, pooled_output_gradient);

                // 2) pass that gradient to the Transformer
                transformer.backward(grad_pooled);

                // Compute training loss
                float loss = cross_entropy_loss(probabilities, train_labels[i]);
                epoch_loss += loss;
            }

            float correct_ratio_train = (float)correct_count_train / (float)train_dataset_2D.size();
            float avg_loss_train = epoch_loss / (float)train_dataset_2D.size();
            std::cout << "\nTrain avg_loss: " << avg_loss_train
                      << " | correct_ratio_train: " << correct_ratio_train << "\n";

            // --------------- VERIFICATION (forward only, no backward) ---------------
            // transformer.inference_mode = true;
            float verify_loss = 0.0f;
            int correct_count_verify = 0;
            for (size_t i = 0; i < verify_dataset_2D.size(); ++i)
            {
                auto trunc_seq = truncate_tokens_max_len(verify_dataset_2D[i], max_len);
                auto pad_mask = create_padding_mask(trunc_seq, max_len);

                auto output_trans = transformer.forward(trunc_seq, pad_mask);
                auto pooled_output = mean_pooling(output_trans);
                auto logits = linear_layer(pooled_output, final_weights, final_bias);
                auto probabilities = softmax(logits);

                // Check prediction
                int predicted_idx = (int)std::distance(
                    probabilities.begin(),
                    std::max_element(probabilities.begin(), probabilities.end()));
                if (predicted_idx == verify_labels[i])
                {
                    correct_count_verify++;
                }
                // Accumulate verify loss
                float loss = cross_entropy_loss(probabilities, verify_labels[i]);
                verify_loss += loss;
            }
            float avg_loss_verify = verify_loss / (float)verify_dataset_2D.size();
            float correct_ratio_verify = (float)correct_count_verify / (float)verify_dataset_2D.size();

            std::cout << "Verify avg_loss: " << avg_loss_verify
                      << " | correct_ratio_verify: " << correct_ratio_verify << "\n";

            // ========== SAVE THE MODEL FILES AT THE END OF EVERY EPOCH ==========
            save_final_layer_weights(final_weights, final_bias);
            transformer.save_embedding_matrix();
            transformer.save_attention_weights();
            transformer.save_feed_forward_weights();
            transformer.save_LayerNormalization_weights();
            std::cout << "[Epoch " << epoch << "] Model files saved.\n";

            // ========== IF THIS IS THE BEST LOSS SO FAR, ALSO COPY TO ./best_model/ ==========
            if (avg_loss_train < best_avg_loss)
            {
                best_avg_loss = avg_loss_train;

                // Copy *.bin files into ./best_model/
                copy_best_model_files_to_directory("./best_model");
                std::cout << ">> New best model detected. Copied *.bin files to ./best_model/.\n";
            }
        }

        // After training, save final weights
        save_final_layer_weights(final_weights, final_bias);
        transformer.save_embedding_matrix();
        transformer.save_attention_weights();
        transformer.save_feed_forward_weights();
        transformer.save_LayerNormalization_weights();
        std::cout << "Training completed.\n";
    }

    return 0;
}
