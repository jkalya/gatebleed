#include "attention.h"
#include "utils.h"
#include <cmath>
#include <cstdlib>  // For std::rand
#include <ctime>    // For std::time
#include <iostream> // For std::cout and std::endl
#include <fstream>  // For file I/O
#include <stdexcept> // for std::out_of_range

// #define PRINT_OUT_TEST_SCALED_DOT_PRODUCT_ATTENTION
// #define USE_KEY_VAL_ADD_BACKPROP
const std::string MultiHeadAttention::file_prefix_attention_weights_q_layer_ = "attention_weights_q_layer_";
const std::string MultiHeadAttention::file_prefix_attention_weights_k_layer_ = "attention_weights_k_layer_";
const std::string MultiHeadAttention::file_prefix_attention_weights_v_layer_ = "attention_weights_v_layer_";

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads, int max_len, bool load_parameters_yes_no, int layer_index)
    : weights_q(d_model, std::vector<float>(d_model, 0.0f)),
      weights_k(d_model, std::vector<float>(d_model, 0.0f)),
      weights_v(d_model, std::vector<float>(d_model, 0.0f)),

      // Velocity (momentum) buffers
      velocity_q(d_model, std::vector<float>(d_model, 0.0f)),
      velocity_k(d_model, std::vector<float>(d_model, 0.0f)),
      velocity_v(d_model, std::vector<float>(d_model, 0.0f)),

      // Gradients total
      grad_weights_q(d_model, std::vector<float>(d_model, 0.0f)),
      grad_weights_k(d_model, std::vector<float>(d_model, 0.0f)),
      grad_weights_v(d_model, std::vector<float>(d_model, 0.0f)),

      // Gradients local per head
      grad_weights_q_local(num_heads, std::vector<std::vector<float>>(d_model, std::vector<float>(d_model / num_heads, 0.0f))),
      grad_weights_k_local(num_heads, std::vector<std::vector<float>>(d_model, std::vector<float>(d_model / num_heads, 0.0f))),
      grad_weights_v_local(num_heads, std::vector<std::vector<float>>(d_model, std::vector<float>(d_model / num_heads, 0.0f))),

      W_q_local_cache(num_heads, std::vector<std::vector<float>>(d_model, std::vector<float>(d_model / num_heads, 0.0f))),
      W_k_local_cache(num_heads, std::vector<std::vector<float>>(d_model, std::vector<float>(d_model / num_heads, 0.0f))),
      W_v_local_cache(num_heads, std::vector<std::vector<float>>(d_model, std::vector<float>(d_model / num_heads, 0.0f))),

      grad_total_all_heads(max_len, std::vector<float>(d_model, 0.0f)),
      merged_attention_output(max_len, std::vector<float>(d_model, 0.0f)),
      attention_score_cache_local(num_heads, std::vector<std::vector<float>>(max_len, std::vector<float>(max_len, 0.0f))),

      // Caches
      query_cache_local(num_heads, std::vector<std::vector<float>>(max_len, std::vector<float>(d_model / num_heads, 0.0f))),
      key_cache_local(num_heads, std::vector<std::vector<float>>(max_len, std::vector<float>(d_model / num_heads, 0.0f))),
      value_cache_local(num_heads, std::vector<std::vector<float>>(max_len, std::vector<float>(d_model / num_heads, 0.0f))),
      num_heads(num_heads),
      max_len(max_len),
      d_model(d_model)


{
    const std::string weights_q_file = file_prefix_attention_weights_q_layer_ + std::to_string(layer_index) + ".bin";
    const std::string weights_k_file = file_prefix_attention_weights_k_layer_ + std::to_string(layer_index) + ".bin";
    const std::string weights_v_file = file_prefix_attention_weights_v_layer_ + std::to_string(layer_index) + ".bin";

    bool loaded = false;

    if (load_parameters_yes_no)
    {
        std::ifstream file_q(weights_q_file, std::ios::binary);
        std::ifstream file_k(weights_k_file, std::ios::binary);
        std::ifstream file_v(weights_v_file, std::ios::binary);

        if (file_q.is_open() && file_k.is_open() && file_v.is_open())
        {
            for (auto &row : weights_q)
            {
                file_q.read(reinterpret_cast<char *>(row.data()), row.size() * sizeof(float));
            }
            for (auto &row : weights_k)
            {
                file_k.read(reinterpret_cast<char *>(row.data()), row.size() * sizeof(float));
            }
            for (auto &row : weights_v)
            {
                file_v.read(reinterpret_cast<char *>(row.data()), row.size() * sizeof(float));
            }
            file_q.close();
            file_k.close();
            file_v.close();

            std::cout << "Attention weights for layer " << layer_index << " loaded from files." << std::endl;
            loaded = true; // Mark as successfully loaded
        }
        else
        {
            std::cerr << "Warning: Could not open weight files for layer " << layer_index << ". Falling back to random initialization." << std::endl;
        }
    }

    if (!loaded)
    {
        std::srand(std::time(0));
        float scale = std::sqrt(2.0f / d_model);

        for (auto &row : weights_q)
        {
            for (auto &val : row)
            {
                val = scale * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f; // Uniform [-scale, scale]
            }
        }

        for (auto &row : weights_k)
        {
            for (auto &val : row)
            {
                val = scale * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f; // Uniform [-scale, scale]
            }
        }

        for (auto &row : weights_v)
        {
            for (auto &val : row)
            {
                val = scale * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f; // Uniform [-scale, scale]
            }
        }

        std::cout << "Attention weights for layer " << layer_index << " initialized with random values." << std::endl;
    }
#ifdef PRINT_OUT_INIT_VECTORS
    // Print a few rows of weights_q, weights_k, and weights_v
    std::cout << "\nSample rows of weights_q for layer " << layer_index << ":" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(3, weights_q.size()); ++i)
    {
        for (float val : weights_q[i])
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nSample rows of weights_k for layer " << layer_index << ":" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(3, weights_k.size()); ++i)
    {
        for (float val : weights_k[i])
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nSample rows of weights_v for layer " << layer_index << ":" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(3, weights_v.size()); ++i)
    {
        for (float val : weights_v[i])
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
#endif
}

std::vector<std::vector<float>> MultiHeadAttention::backward(
    const std::vector<std::vector<float>> &grad_total,
    const std::vector<std::vector<float>>& qkv_input,
    int head_number)
{
    size_t grad_rows = grad_total.size();
    size_t grad_cols = grad_total[0].size() / num_heads;

    std::vector<std::vector<float>> grad_output(grad_rows, std::vector<float>(grad_cols, 0.0));
    for(size_t row_cnt=0; row_cnt < grad_rows; row_cnt++)
    {
        for(size_t col_cnt=0; col_cnt < grad_cols; col_cnt++)
        {
            grad_output[row_cnt][col_cnt] = grad_total[row_cnt][col_cnt + head_number * d_model / num_heads];//Copy over this heads gradient only
        }
    }

    //=============================================================
    // 1) Backprop through final matmul: output = A * V
    //=============================================================
    auto A = attention_score_cache_local[head_number];
    auto dV = Utils::matmul(Utils::transpose(A), grad_output);
    auto dA = Utils::matmul(grad_output, Utils::transpose(value_cache_local[head_number]));


    //=============================================================
    // 2) Backprop through softmax
    //=============================================================
    std::vector<std::vector<float>> dScores = Utils::softmax_backward(dA, A);

    //=============================================================
    // 3) Backprop through scaled dot-product attention
    //=============================================================
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(key_cache_local[0][0].size()));

    // Notice in forwar K is transposed before do matmul in forward like this matmul(Q,transpose(K)) 
    auto dQ = Utils::matmul(dScores, key_cache_local[head_number]);//dQ = dScore x (key^T)^T
    auto dK = Utils::matmul(Utils::transpose(dScores), query_cache_local[head_number]);//dK = dScore^T x Q
    Utils::scale_inplace(dQ, scale_factor);
    Utils::scale_inplace(dK, scale_factor);


    //=============================================================
    // 4) Backprop through the linear transformations
    //=============================================================
    grad_weights_q_local[head_number] = Utils::matmul(Utils::transpose(qkv_input), dQ);
    grad_weights_k_local[head_number] = Utils::matmul(Utils::transpose(qkv_input), dK);
    grad_weights_v_local[head_number] = Utils::matmul(Utils::transpose(qkv_input), dV);

    auto grad_query_local_single_head = Utils::matmul(dQ, Utils::transpose(W_q_local_cache[head_number]));
    auto grad_key_local_single_head = Utils::matmul(dK, Utils::transpose(W_k_local_cache[head_number]));
    auto grad_value_local_single_head = Utils::matmul(dV, Utils::transpose(W_v_local_cache[head_number]));

    //=============================================================
    // 5) Combine or return whichever gradient is relevant
    //=============================================================

    size_t rows = grad_total_all_heads.size();
    size_t cols = grad_total_all_heads[0].size();
    for(size_t row_cnt=0;row_cnt < rows;row_cnt++)
    {
        for(size_t col_cnt=0;col_cnt < cols;col_cnt++)
        {
            if(head_number == 0)
            {
                grad_total_all_heads[row_cnt][col_cnt] = grad_query_local_single_head[row_cnt][col_cnt];//Introduce with head 0 gradient to thie total gradient
            }
            else
            {
                grad_total_all_heads[row_cnt][col_cnt] += grad_query_local_single_head[row_cnt][col_cnt];//Add up the rest of the heads gradients
            }
#ifdef USE_KEY_VAL_ADD_BACKPROP
            std::cout << "USE_KEY_VAL_ADD_BACKPROP" << std::endl; 
            grad_total_all_heads[row_cnt][col_cnt] += grad_key_local_single_head[row_cnt][col_cnt];//Add up the key gradient
            grad_total_all_heads[row_cnt][col_cnt] += grad_value_local_single_head[row_cnt][col_cnt];//Add up the value gradient
#endif
        }
    }
    return grad_total_all_heads;
    
}

void MultiHeadAttention::save_weights(int layer_index)
{
    const std::string weights_q_file = file_prefix_attention_weights_q_layer_ + std::to_string(layer_index) + ".bin";
    const std::string weights_k_file = file_prefix_attention_weights_k_layer_ + std::to_string(layer_index) + ".bin";
    const std::string weights_v_file = file_prefix_attention_weights_v_layer_ + std::to_string(layer_index) + ".bin";

    std::ofstream save_file_q(weights_q_file, std::ios::binary);
    std::ofstream save_file_k(weights_k_file, std::ios::binary);
    std::ofstream save_file_v(weights_v_file, std::ios::binary);

    if (save_file_q.is_open() && save_file_k.is_open() && save_file_v.is_open())
    {
        for (const auto &row : weights_q)
        {
            save_file_q.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(float));
        }
        for (const auto &row : weights_k)
        {
            save_file_k.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(float));
        }
        for (const auto &row : weights_v)
        {
            save_file_v.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(float));
        }
        save_file_q.close();
        save_file_k.close();
        save_file_v.close();

        std::cout << "Attention weights for layer " << layer_index << " saved to files." << std::endl;
    }
    else
    {
        std::cerr << "Error: Could not save attention weights for layer " << layer_index << " to files." << std::endl;
        exit(EXIT_FAILURE);
    }
}

std::vector<std::vector<float>> MultiHeadAttention::forward(
    const std::vector<std::vector<float>> &query,
    const std::vector<std::vector<float>> &key,
    const std::vector<std::vector<float>> &value,
    const std::vector<int> &padding_mask,
    int head_number)

{

    // Copy over parts from total weights to local weights
    size_t d_segment_size = weights_q[0].size() / num_heads;
   
    for (size_t i = 0; i < weights_q.size(); ++i)
    {
        for (size_t j = 0; j < d_segment_size; j++)
        {
            W_q_local_cache[head_number][i][j] = weights_q[i][j + d_segment_size * head_number];
            W_k_local_cache[head_number][i][j] = weights_k[i][j + d_segment_size * head_number];
            W_v_local_cache[head_number][i][j] = weights_v[i][j + d_segment_size * head_number];
        }
    }

    // 1. Linear transformations for Q, K, V for this head
    query_cache_local[head_number] = Utils::matmul(query, W_q_local_cache[head_number]);
    key_cache_local[head_number] = Utils::matmul(key, W_k_local_cache[head_number]);
    value_cache_local[head_number] = Utils::matmul(value, W_v_local_cache[head_number]);

    // 2. Scaled dot-product attention
    //std::vector<std::vector<float>> attention_output;
    auto attention_output = scaled_dot_product_attention(query_cache_local[head_number], key_cache_local[head_number], value_cache_local[head_number], padding_mask, head_number);
    // Merge attention output back into original format
    for (size_t i = 0; i < attention_output.size(); ++i)
    {
        for (size_t j = 0; j < d_segment_size; ++j)
        {
            merged_attention_output[i][j + head_number * d_segment_size] = attention_output[i][j];
        }
    }

    return merged_attention_output;
    
    //return query;
}



float MultiHeadAttention::read_weight(const std::string &matrix_type, int row, int col) const
{
    // Decide which matrix to read from
    const std::vector<std::vector<float>> *target_matrix = nullptr;

    if (matrix_type == "Q")
    {
        target_matrix = &weights_q;
    }
    else if (matrix_type == "K")
    {
        target_matrix = &weights_k;
    }
    else if (matrix_type == "V")
    {
        target_matrix = &weights_v;
    }
    else
    {
        throw std::invalid_argument("Invalid matrix_type. Must be one of {\"Q\", \"K\", \"V\"}.");
    }

    // Safety check for out-of-range
    if (row < 0 || row >= static_cast<int>(target_matrix->size()))
    {
        throw std::out_of_range("Row index out of range in read_weight()");
    }
    if (col < 0 || col >= static_cast<int>((*target_matrix)[row].size()))
    {
        throw std::out_of_range("Column index out of range in read_weight()");
    }

    return (*target_matrix)[row][col];
}

void MultiHeadAttention::update_weights()
{
    // Example: read from some config or define here
    float learning_rate = GLOBAL_ATTENTION_learning_rate;
    float momentum = GLOBAL_ATTENTION_momentum;
    

    size_t d_segment_size = grad_weights_k_local[0][0].size();
    for (int h_cnt = 0; h_cnt < num_heads;h_cnt++)
    {
        for(size_t i = 0; i < grad_weights_k.size(); i++)
        {
            for(size_t j = 0; j < d_segment_size; j++ )
            {
                grad_weights_q[i][j + h_cnt * d_segment_size] = grad_weights_q_local[h_cnt][i][j];
                grad_weights_k[i][j + h_cnt * d_segment_size] = grad_weights_k_local[h_cnt][i][j];
                grad_weights_v[i][j + h_cnt * d_segment_size] = grad_weights_v_local[h_cnt][i][j];
            }
        }
    }
    // Update weights_q
    for (size_t i = 0; i < weights_q.size(); i++)
    {
        for (size_t j = 0; j < weights_q[0].size(); j++)
        {
            // velocity_q = momentum * velocity_q + grad
            velocity_q[i][j] = momentum * velocity_q[i][j] + learning_rate * grad_weights_q[i][j];
            // w_q -= velocity_q
            weights_q[i][j] -= velocity_q[i][j];
            // Optionally reset the grad to zero
            grad_weights_q[i][j] = 0.0f;
        }
    }

    // Update weights_k
    for (size_t i = 0; i < weights_k.size(); i++)
    {
        for (size_t j = 0; j < weights_k[0].size(); j++)
        {
            velocity_k[i][j] = momentum * velocity_k[i][j] + learning_rate * grad_weights_k[i][j];
            weights_k[i][j] -= velocity_k[i][j];
            grad_weights_k[i][j] = 0.0f;
        }
    }

    // Update weights_v
    for (size_t i = 0; i < weights_v.size(); i++)
    {
        for (size_t j = 0; j < weights_v[0].size(); j++)
        {
            velocity_v[i][j] = momentum * velocity_v[i][j] + learning_rate * grad_weights_v[i][j];
            weights_v[i][j] -= velocity_v[i][j];
            grad_weights_v[i][j] = 0.0f;
        }
    }
}
std::vector<std::vector<float>> MultiHeadAttention::scaled_dot_product_attention(
    const std::vector<std::vector<float>> &query,
    const std::vector<std::vector<float>> &key,
    const std::vector<std::vector<float>> &value,
    const std::vector<int> &padding_mask,
    int head_number)
{
    // 1. Compute QK^T
 
    auto scores = Utils::matmul(query, Utils::transpose(key));

    // 2. Scale scores by sqrt(d_k)
    float scale_factor = std::sqrt(static_cast<float>(key[0].size()));
    for (size_t i = 0; i < scores.size(); ++i)
    {
        for (size_t j = 0; j < scores[0].size(); ++j)
        {
            scores[i][j] /= scale_factor;
        }
    }
    // 3. Apply masking
    for (size_t i = 0; i < scores.size(); ++i)
    {
        for (size_t j = 0; j < scores[i].size(); ++j)
        {
            if (padding_mask[j] == 0)
            {
                scores[i][j] = -std::numeric_limits<float>::infinity();
            }
        }
    }
    // 4. Apply softmax to scores
    // Softmax
    for (size_t i = 0; i < scores.size(); ++i)
    {
        scores[i] = Utils::softmax(scores[i]);
    }
    // 5. Multiply scores with V
    // Store final attention distribution for backprop
    attention_score_cache_local[head_number] = scores;

    auto output = Utils::matmul(scores, value);
    return output;
}


