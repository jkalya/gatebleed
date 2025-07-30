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

#include <algorithm> // For Fisher-Yates shuffle & std::min
#include <random>    // For random number generation
#include <chrono>    // For seeding random number generator

#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdbool.h>

#include <cmath>
#include <stdexcept>
#include <iomanip>   // For formatted output
#include <cstring>
#include <cstdint>

#include <fcntl.h>
#include <sys/types.h>


#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18
#define MAX_ROWS 16
#define MAX_COLS 64

#define MSR_RAPL_POWER_UNIT 0x606
#define MSR_PKG_ENERGY_STATUS 0x611
#define MSR_DRAM_ENERGY_STATUS 0x619

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

// Function to get the current value from the RDTSC
static inline uint64_t rdtsc() {
    unsigned int lo, hi;
    asm volatile("mfence");
    __asm__ volatile ("rdtsc" : "=a" (lo), "=d" (hi));
    asm volatile("mfence");
    return ((uint64_t)hi << 32) | lo;
}

//Final Classification Layer 
std::vector<float> linear_layer(const std::vector<float>& input, const std::vector<std::vector<float>>& weights, const std::vector<float>& bias) {
    
    ///--- new
    //size_t rows = 1;
    size_t cols = weights[0].size();
    size_t inner_dim = weights.size();
    //std::cout << "rows: " << 1 << " - cols: "  << cols << " - inner_dim: " << inner_dim << ".\n";
    //size_t rows_split = 1;
    size_t cols_split = (cols + 15) / 16;
    size_t inner_split = (inner_dim + 31) / 32;
    // std::cout << "rows_split: " << rows_split << " - cols_split: "  << cols_split << " - inner_split: " << inner_split << ".\n";
    
    std::vector<float> output(weights[0].size(), 0.0f);

    float value;
    uint32_t bits;
    //uint16_t upperBits;
    uint16_t Tile_A_Values[16*32];
    uint16_t Tile_B_Values[16*32];
    float Tile_Res_Values[16*16];
    for (size_t c = 0; c < cols_split; ++c) {
        //std::fill(std::begin(Tile_Res_Values), std::end(Tile_Res_Values), (float)0);
        //_tile_loadd (0, Tile_Res_Values, 64);   // Load zeros into tile 0
        _tile_zero(0);
        for (size_t in = 0; in < inner_split; ++in) {
            // preparing Tile1 data from Matrix a
            std::fill(std::begin(Tile_A_Values), std::end(Tile_A_Values), (uint16_t)0);
            for (size_t j = 0; j < std::min((size_t)32, inner_dim-in*32); ++j) {
                value = input[in*32+j];
                std::memcpy(&bits, &value, sizeof(bits));
                Tile_A_Values[j] = static_cast<uint16_t>(bits >> 16); // upperbits: bfloat16
            }
            _tile_loadd (1, Tile_A_Values, 64);   // Load data from Matrix a into tile 1
            // preparing Tile2 data from Matrix b
            std::fill(std::begin(Tile_B_Values), std::end(Tile_B_Values), 0);
            for (size_t i = 0; i < std::min((size_t)32, inner_dim-in*32); ++i) {
                for (size_t j = 0; j < std::min((size_t)16, cols-c*16); ++j) {
                    value = weights[in*32+i][c*16+j];
                    std::memcpy(&bits, &value, sizeof(bits));
                    Tile_B_Values[((size_t)(i/2))*32 + i%2 + 2*j] = static_cast<uint16_t>(bits >> 16); // upperbits: bfloat16
                }
            }
            _tile_loadd (2, Tile_B_Values, 64);   // Load data from Matrix b into tile 2

            //_mm_mfence();
            //uint64_t start_cycles = rdtsc();
            _tile_dpbf16ps (0, 1, 2);
            //_mm_mfence();
            //uint64_t end_cycles = rdtsc();
            //uint64_t elapsed_cycles =  (end_cycles - start_cycles);
            //std::cout << "linear: " << elapsed_cycles << "\n";
        }
        _tile_stored (0, Tile_Res_Values, 64);
        // Filling the result + add bias term
        for (size_t j = 0; j < std::min((size_t)16, cols-c*16); ++j) {
            output[c*16+j] = Tile_Res_Values[j] + bias[c*16+j];
        }
    }
    return output;

//    //--- old
//    std::vector<float> output(weights[0].size(), 0.0f);
//    for (size_t i = 0; i < weights[0].size(); ++i) { // For each output category
//        for (size_t j = 0; j < input.size(); ++j) {  // For each input dimension
//            output[i] += input[j] * weights[j][i];
//        }
//        output[i] += bias[i]; // Add bias term
//    }
//
//    return output;
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

// Function to read an MSR register using the file interface
uint64_t read_msr(int fd, uint32_t reg) {
    uint64_t value = 0;
    // pread(fd, &value, sizeof(value), reg);
    return value;
}

void run_prompt_mode(Transformer& transformer, int max_len, const unordered_map<string, int>& vocab, const std::vector<std::vector<float>>& weights, const std::vector<float>& bias, bool iscold) {
    string input;
    std::string file   = "sample_1_prompt.txt"; // "answer_1_prompt.txt"
    size_t Iter_Num = 1;
    size_t HIST_ITERATIONS = 5000; // 50001

    uint64_t timestamps[HIST_ITERATIONS];
    double PKG_energy_readings[HIST_ITERATIONS];
    double DRAM_energy_readings[HIST_ITERATIONS];

    int fd;
    uint64_t msr_value;

    uint64_t Start_time, End_time, PKG_Energy, DRAM_Energy;
    double Start_PKG_Energy_Joul, End_PKG_Energy_Joul, Start_DRAM_Energy_Joul, End_DRAM_Energy_Joul;

    int junk1 = 0 , junk2 = 0, Delay; // Delay = 20000000 vs 0; // Cold vs Warm
    if (iscold)
    {
        Delay = 20000000;
    }
    else
    {
        Delay = 0;
    }
    unsigned long long start_cycles, end_cycles, elapsed_cycles;
    // Open MSR device file
    fd = open("/dev/cpu/28/msr", O_RDONLY);
    //if (fd < 0) {
	//printf("Failed to open /dev/cpu/28/msr\n");
	//return;
	//}
    // Read the MSR_RAPL_POWER_UNIT register once to get units
    msr_value = read_msr(fd, MSR_RAPL_POWER_UNIT);
    // Extract energy status unit bits (bits 12:8)
    int energy_units = (msr_value >> 8) & 0x1F;
    double energy_multiplier = 1.0 / (1 << energy_units);

    for (size_t k = 0; k < HIST_ITERATIONS; ++k) {
        //cout << "Loop Started\n";
        _mm_mfence();
        Start_time = rdtsc();

        msr_value = read_msr(fd, MSR_PKG_ENERGY_STATUS);
        PKG_Energy = msr_value & 0xFFFFFFFF;
        Start_PKG_Energy_Joul = PKG_Energy * energy_multiplier;

        msr_value = read_msr(fd, MSR_DRAM_ENERGY_STATUS);
        DRAM_Energy = msr_value & 0xFFFFFFFF;
        Start_DRAM_Energy_Joul = DRAM_Energy * energy_multiplier;
        // cout << "Start Time (rdtsc): " << Start_time << " - Start_PKG_Energy (joul): " << Start_PKG_Energy_Joul << " - Start_DRAM_Energy (joul): " << Start_DRAM_Energy_Joul << "\n";
        for (size_t i = 0; i < Iter_Num; ++i) {
            std::ifstream ifs(file);
            while (std::getline(ifs, input)) {
                if (input.empty()) continue;  // skip empty lines

                // Tokenize input
                vector<int> tokens = tokenize(input, vocab);

                // Truncate and create padding mask
                auto trunc_sequence = truncate_tokens_max_len(tokens, max_len);
                auto padding_mask = create_padding_mask(trunc_sequence, max_len);

                //cout << "Truncated tokens (max length " << max_len << "): ";
                //for (const auto& token : trunc_sequence) {
                //    cout << token << " ";
                //}
                //cout << "\n";

                // Run forward pass through transformer
                vector<vector<float>> output = transformer.forward(trunc_sequence, padding_mask);
                // Reduce transformer output (e.g., by mean pooling)
                std::vector<float> pooled_output = mean_pooling(output);
                // Apply final classification layer
                vector<float> logits = linear_layer(pooled_output, weights, bias);
                vector<float> probabilities = softmax(logits);

                // Print probabilities
                //cout << "Category probabilities:\n";

                //cout << "Question: " << probabilities[0] << "\n";
                //cout << "Answer: " << probabilities[1] << "\n";

                // Predict category
                string prediction = (probabilities[0] > probabilities[1]) ? "Question" : "Answer";
                //cout << "Predicted category: " << prediction << "\n";
            }
            ifs.close();
        }
        _mm_mfence();
        End_time = rdtsc();

        msr_value = read_msr(fd, MSR_PKG_ENERGY_STATUS);
        PKG_Energy = msr_value & 0xFFFFFFFF;
        End_PKG_Energy_Joul = PKG_Energy * energy_multiplier;

        msr_value = read_msr(fd, MSR_DRAM_ENERGY_STATUS);
        DRAM_Energy = msr_value & 0xFFFFFFFF;
        End_DRAM_Energy_Joul = DRAM_Energy * energy_multiplier;
        //cout << "End Time (rdtsc): " << End_time << " - End_PKG_Energy (joul): " << End_PKG_Energy_Joul << " - End_DRAM_Energy (joul): " << End_DRAM_Energy_Joul << "\n";
        timestamps[k] = End_time - Start_time;
        PKG_energy_readings[k] = End_PKG_Energy_Joul - Start_PKG_Energy_Joul;
        DRAM_energy_readings[k] = End_DRAM_Energy_Joul - Start_DRAM_Energy_Joul;

        // Prompt Interval
        _mm_mfence();
		junk1 = 0; junk2 = 0;
		start_cycles = rdtsc();
		for (int p = 0; p < Delay; p++) {
		    junk2 = junk1 + p;
		}
		_mm_mfence();
		end_cycles = rdtsc();
		elapsed_cycles =  (end_cycles - start_cycles);
        //cout << "Intrupt between prompts: " << elapsed_cycles << "\n";
    }
    // Close the MSR file
    close(fd);
    // Open output file for writing
    FILE *output_file = fopen("readings.txt", "w");
    if (output_file == NULL) {
        printf("Failed to open output file for writing\n");
        return;
    }
    // Write stored readings to the file
    for (size_t i = 0; i < HIST_ITERATIONS; i++) {
        fprintf(output_file, "%lu\t%f\t%f\n", timestamps[i], PKG_energy_readings[i], DRAM_energy_readings[i]);
    }

    // Close the output file
    fclose(output_file);
}

void run_prompt_mode_old(Transformer& transformer, int max_len, const unordered_map<string, int>& vocab, const std::vector<std::vector<float>>& weights, const std::vector<float>& bias) {
    string input;
    while (true) {
        cout << "\nEnter a string (or type 'exit' to quit): ";
        getline(cin, input);

        if (input == "exit") {
            cout << "Exiting mini prompt mode.\n";
            break;
        }

        // Tokenize input
        vector<int> tokens = tokenize(input, vocab);

        // Truncate and create padding mask
        auto trunc_sequence = truncate_tokens_max_len(tokens, max_len);
        auto padding_mask = create_padding_mask(trunc_sequence, max_len);

        cout << "Truncated tokens (max length " << max_len << "): ";
        for (const auto& token : trunc_sequence) {
            cout << token << " ";
        }
        cout << "\n";

        // Run forward pass through transformer
        vector<vector<float>> output = transformer.forward(trunc_sequence, padding_mask);
        // Reduce transformer output (e.g., by mean pooling)
        std::vector<float> pooled_output = mean_pooling(output);
        // Apply final classification layer
        vector<float> logits = linear_layer(pooled_output, weights, bias);
        vector<float> probabilities = softmax(logits);

        // Print probabilities
        cout << "Category probabilities:\n";
        
        cout << "Question: " << probabilities[0] << "\n";
        cout << "Answer: " << probabilities[1] << "\n";

        // Predict category
        string prediction = (probabilities[0] > probabilities[1]) ? "Question" : "Answer";
        cout << "Predicted category: " << prediction << "\n";
    }
}

//Define tile config data structure 
typedef struct __tile_config
{
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[16]; 
  uint8_t rows[16]; 
} __tilecfg;

/* Initialize tile config */
static void init_tile_config (__tilecfg *tileinfo)
{
  int i;
  tileinfo->palette_id = 1;
  tileinfo->start_row = 0;

  for (i = 0; i < 4; ++i)
  {
    tileinfo->colsb[i] = MAX_COLS;
    tileinfo->rows[i] =  MAX_ROWS;
  }

  _tile_loadconfig (tileinfo); // LDTILECFG  :  Load tile configuration.
}

/* Set_tiledata_use() - Invoke syscall to set ARCH_SET_STATE_USE */
static bool set_tiledata_use()
{
   if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) 
   {
      printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
      return false;
   }
   else
   {
      printf("\n TILE DATA USE SET - OK \n\n");
      return true;
   }

   return true;
}

bool isCold(const std::string& temperature) {
    if (temperature == "cold") {
        return true;
    } else {
        return false;
    }
}

int main() {

    std::string input;
    std::cout << "Enter 'cold' or 'warm': ";
    std::cin >> input;

    bool Iscold = isCold(input);
    
    bool load_parameters_yes_no = false;

    __tilecfg tile_data = {0};
    // Request permission to linux kernel to run AMX
    if (!set_tiledata_use())
        exit(-1);

    // Load tile configuration 
    init_tile_config (&tile_data);

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


    // Input matrices
    std::vector<std::vector<float>> A = {
        {1.0, 2.0, 3.0, 4.0}, 
        {5.0, 6.0, 7.0, 8.0}, 
        {9.0, 10.0, 11.0, 12.0}  
    };
    std::vector<std::vector<float>> B = {
        {1.0, 2.0}, 
        {3.0, 4.0}, 
        {5.0, 6.0}, 
        {7.0, 8.0}
    };


    auto Y = Utils::matmul(A, B);
    std::cout << "A: " << endl;
    Utils::print_matrix(A);
    std::cout << "B: " << endl;
    Utils::print_matrix(B);
    std::cout << "Y = A × B: " << endl;
    Utils::print_matrix(Y);

//If Y=A×B,then:
//dA=dY×BT,dB=AT×dY.

    auto dY = Y;
    auto dB = Utils::matmul(Utils::transpose(A), dY);
    auto dA = Utils::matmul(dY, Utils::transpose(B));

    std::cout << "If Y = A × B,then:  dA=dY×BT,dB=AT×dY. " << endl;
    std::cout << "dA " << endl;
    Utils::print_matrix(dA);
    std::cout << "dB " << endl;
    Utils::print_matrix(dB);



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

    // old
    //std::cout << "Do you want to load an existing model parameter with embedding matrix from a file? (Y/N, y/n): ";
    //std::string choice;
    //std::cin >> choice;

    //if (choice == "/Y" || choice == "y")
    //{
    //    load_parameters_yes_no = true; // Load from file
    //}
    //else
    //{
    //    load_parameters_yes_no = false; // Random initialization
    //}
    // new
    load_parameters_yes_no = true; // Load from file

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
    std::vector<int> padding_mask = {
        {1, 1, 1, 1}
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
    auto attention_output_printout = attention_layer_printout.scaled_dot_product_attention(Q, K, V, padding_mask);

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
    int d_model = 64; // The "resolution" of the positional encoding and embedding space. 
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

    int num_heads = 4;// Number of attention heads. The attention class split the Q,K and V vector bus into smaller attention vectors 
                      // and then the splitted Q_split,K_split and V_split vectors combined togheter again before enter the global Q,K and V vector bus feed forward
                      // so if num_heads = 4 and d_model = 64 each attention have only d_model/num_heads = 64/4 = 16 loacal dimentsion to calculate on
    int d_ff = 96;   // d_ff: Dimensionality of the hidden layer in the feed-forward network.
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
    int max_len = 12;

    std::cerr << "d_model: " << d_model << std::endl;
    std::cerr << "num_heads: " << num_heads << std::endl;
    int check_num_head_settings = d_model % num_heads;
    if(check_num_head_settings != 0)
    {
        std::cerr << "Failed check_num_head_settings != 0: " << check_num_head_settings << std::endl;
        return -1;
    }
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

    // ================== Set up the transformer ==================
    vocab_size = vocab.size(); // Dynamically set to the actual vocabulary size
    cout << "vocab_size = " << vocab_size << endl;



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
    Transformer transformer(vocab_size, d_model, num_heads, max_len, d_ff, num_layers, load_parameters_yes_no);
    std::cout << " ** vocab_size: " << vocab_size << std::endl;   
    std::cout << " ** d_model: " << d_model << std::endl;
    std::cout << " ** num_heads: " << num_heads << std::endl;
    std::cout << " ** max_len: " << max_len << std::endl;   
    std::cout << " ** num_layers: " << num_layers << std::endl;  

    //cout << "Do you want to start mini prompt mode? (Y/N): ";
    //string response;
    //cin >> response;
    //cin.ignore(); // Ignore trailing newline character from cin

    // if (response == "Y" || response == "y" || response == "Yes" || response == "yes" || response == "YES") {
	if (1) {
        run_prompt_mode(transformer, max_len, vocab, final_weights, final_bias, Iscold);
    } else {
        cout << "Continuing with training loop...\n";
        
    // ============== Training loop ===================
    int epochs = 200;
    // Initialize velocity for weights and bias
    std::vector<std::vector<float>> velocity_weights(final_weights.size(),
                                                     std::vector<float>(final_weights[0].size(), 0.0f));
    std::vector<float> velocity_bias(final_bias.size(), 0.0f);

    GLOBAL_learning_rate = 0.0002;
    GLOBAL_momentum = 0.92;
    GLOBAL_ATTENTION_learning_rate = GLOBAL_learning_rate;//0.1
    GLOBAL_ATTENTION_momentum = GLOBAL_momentum; //0.5  
    std::cout << "learning_rate: " << GLOBAL_learning_rate << std::endl;
    std::cout << "momentum: " << GLOBAL_momentum << std::endl;
    // Training loop with gradient computation
    float best_avg_loss = 10000.0;
    const int print_dot_interval = 10;
    int print_dot_cnt = 0;
    for (int epoch = 1; epoch <= epochs; ++epoch)
    {


        std::cout << "Epoch " << epoch << " / " << epochs << "\n";
        // Shuffle dataset
        fisher_yates_shuffle(dataset_2D, labels);
        float epoch_loss = 0.0f; // Accumulate loss for the epoch
        int correct_prob_cnt = 0;
        int data_set_cnt = 0;
        for (size_t i = 0; i < dataset_2D.size(); ++i)
        {
            if(print_dot_cnt < print_dot_interval)
            {
                print_dot_cnt++;
            }
            else
            {
                cout << "." << flush; 
                print_dot_cnt = 0;
            }
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

        //    std::cout << "Halt program debug here" << std::endl;
        //    while(1)
        //    {}
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
         //   cout << "Size of probabilities: " << probabilities.size() << endl;
            int idx=0;
            int predicted_idx = 0;
            float predict_max = 0.0;
            for(auto val : probabilities)
            {
                if(predict_max < val)
                {
                    predict_max = val;
                    predicted_idx = idx; 
                }
           //     cout << "probabilities[ " << idx << "] : "<< val << endl;
                idx++;
            }
           // cout << " labels[" << i << "] : " << labels[i] << " predicted_idx : " << predicted_idx << endl;
            if(predicted_idx == labels[i])
            {
                correct_prob_cnt++;
            }
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

            // Backpropagate gradient through the Transformer 
           transformer.backward(grad_pooled);

           // print_out_probabilities(probabilities, padded_input);// Print probabilities for debugging
           //  Compute loss and accumulate
           float loss = cross_entropy_loss(probabilities, labels[i]);
           epoch_loss += loss;

          data_set_cnt++;
        }
        float correct_prob = (float)correct_prob_cnt/(float)data_set_cnt;
        cout << "** correct_prob : " << correct_prob << endl;
        float avg_loss_this_epoch = epoch_loss / dataset_2D.size();
        if(best_avg_loss > avg_loss_this_epoch)
        {
            best_avg_loss = avg_loss_this_epoch;
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

    }

    // Save final layer weights (optional)
    save_final_layer_weights(final_weights, final_bias);
    transformer.save_embedding_matrix();
    transformer.save_attention_weights();
    transformer.save_feed_forward_weights();    
    transformer.save_LayerNormalization_weights();

    
#endif

    _tile_release (); // TILERELEASE  :  Release tile.
    return 0;

}
