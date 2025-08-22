#ifndef ATTENTION_H
#define ATTENTION_H

#include <vector>
#include "config.h"
#include <iostream> // For std::cout and std::endl
#include <fstream>  // For file I/O

class MultiHeadAttention {
public:
    MultiHeadAttention(int d_model, int num_heads, int max_len, bool load_parameters_yes_no, int layer_index);

    std::vector<std::vector<float>> forward(
        const std::vector<std::vector<float>>& query,
        const std::vector<std::vector<float>>& key,
        const std::vector<std::vector<float>>& value,
        const std::vector<int>& padding_mask,
        int head_number
    );

    // Backward pass to compute gradients
    std::vector<std::vector<float>> backward(
        const std::vector<std::vector<float>>& grad_output, // Gradient from the next layer
        const std::vector<std::vector<float>>& qkv_input,      
        int head_number
    );

    // New method to update weights AFTER backward pass
    void update_weights();

    // Save weights to binary files
    void save_weights(int layer_index);
    float read_weight(const std::string& matrix_type, int row, int col) const;


#ifdef TEST_ATTENTION
public:
#else
    #ifdef PRINT_OUT_TEST_ATTENTION_FORWARD_OPERATION
    public:
    #else
    private:
    #endif
#endif

    // (Optionally public for testing or debugging)
    std::vector<std::vector<float>> scaled_dot_product_attention(
        const std::vector<std::vector<float>>& query,
        const std::vector<std::vector<float>>& key,
        const std::vector<std::vector<float>>& value,
        const std::vector<int>& padding_mask,
        int head_number
    );

    // ——————————————
    // Learned Weights
    // ——————————————
    std::vector<std::vector<float>> weights_q; // Query weights
    std::vector<std::vector<float>> weights_k; // Key weights
    std::vector<std::vector<float>> weights_v; // Value weights

//private:
public:
    // Static constants for storing/loading
    static const std::string file_prefix_attention_weights_q_layer_;
    static const std::string file_prefix_attention_weights_k_layer_;
    static const std::string file_prefix_attention_weights_v_layer_;

    // ——————————————
    // Momentum buffers
    // ——————————————
    std::vector<std::vector<float>> velocity_q; 
    std::vector<std::vector<float>> velocity_k;
    std::vector<std::vector<float>> velocity_v;

    // ——————————————
    // Gradients
    // ——————————————
    std::vector<std::vector<float>> grad_weights_q;
    std::vector<std::vector<float>> grad_weights_k;
    std::vector<std::vector<float>> grad_weights_v;

    std::vector<std::vector<std::vector<float>>> grad_weights_q_local;
    std::vector<std::vector<std::vector<float>>> grad_weights_k_local;
    std::vector<std::vector<std::vector<float>>> grad_weights_v_local;

    // local weight slices: segment_size x segment_size
    std::vector<std::vector<std::vector<float>>> W_q_local_cache;
    std::vector<std::vector<std::vector<float>>> W_k_local_cache;
    std::vector<std::vector<std::vector<float>>> W_v_local_cache;

    //This 3 could be summed up togheter it is the gradeient with respect to forward input side
    std::vector<std::vector<float>> grad_total_all_heads;//Could taken from Q all head summed up OR of all three matrix Q,K and V summed up
    std::vector<std::vector<float>> merged_attention_output;
     
    // ——————————————————
    // Caches for backprop
    // ——————————————————
    std::vector<std::vector<std::vector<float>>> attention_score_cache_local;
    // outputs of the linear transforms (Q_local, K_local, V_local)
    std::vector<std::vector<std::vector<float>>> query_cache_local; 
    std::vector<std::vector<std::vector<float>>> key_cache_local;   
    std::vector<std::vector<std::vector<float>>> value_cache_local; 
    int num_heads;
    int max_len;
    int d_model;
 
};

#endif // ATTENTION_H
