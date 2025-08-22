#include "transformer.h"
#include <iostream>
#include "dataset.h"
#include <vector>
#include "utils.h"
#include <unordered_map>
#include "timer.h"
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
#endif


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

int main(int argc, char* argv[]) {
	if (argc != 3) {
		std::cerr << "Syntax: ./infer EARLY_EXIT WARMUP_AMX" << std::endl;
		exit(-1);
	}

	bool early_exit = (bool)atoi(argv[1]);
	bool warmup_amx = (bool)atoi(argv[2]);
	
    __tilecfg tile_data = {0};
    // Request permission to linux kernel to run AMX
    if (!set_tiledata_use())
        exit(-1);

    // Load tile configuration 
    init_tile_config (&tile_data);
	
    // ----------------------------------------------------------------
    // Step 1: Load vocabulary from file
    // ----------------------------------------------------------------
    std::unordered_map<std::string, int> vocab;
    std::string vocab_file = "vocab.txt";
    if (!load_vocab_from_file(vocab_file, vocab)) {
        std::cerr << "Failed to load vocab from: " << vocab_file << std::endl;
        return -1;
    }
	
	// ** vocab_size: 2748
	// ** d_model: 64
	// ** num_heads: 4
	// ** max_len: 12
	// ** num_layers: 6
	// d_ff = 96
	int vocab_size = 2748;
	int max_len = 12;
	int d_model = 64;
	int num_categories = 2;

	// Init transformer
	Transformer transformer(2748, 64, 4, max_len, 96, 6, true);

	std::vector<std::vector<float>> final_weights(d_model, std::vector<float>(num_categories, 0.0f));
    std::vector<float> final_bias(num_categories, 0.0f);
	load_final_layer_weights(final_weights, final_bias);

	// for (int i = 0; i < x; i++) {
		
	// }

	// Grab prompt from file
	//std::string file = "sample_1_prompt.txt";
	// std::ifstream ifs(file);
	// std::string input;
	// std::getline(ifs, input);
	// //if (input.empty()) {
	// 	std::cerr << "Could not get prompt!" << std::endl;
	// 	return 1;
	// 	//}
	// ifs.close();

	//std::vector<int> tokens = tokenize(input, vocab);

	std::vector<int> tokens = {1,1,1,1,1,1,1,1,1,1,0};
	
	// Truncate and create padding mask
	// for (int a = 0; a < vocab_size; a++) {
	// 	for (int b = 0; b < vocab_size; b++) {
	// 		for (int c = 0; c < vocab_size; c++) {
	// 			for (int d = 0; d < vocab_size; d++) {
	// 				for (int e = 0; e < vocab_size; e++) {
	// 					for (int f = 0; f < vocab_size; f++) {
	// 						for (int g = 0; g < vocab_size; g++) {
	// 							for (int h = 0; h < vocab_size; h++) {
	// 								for (int i = 0; i < vocab_size; i++) {
	// 									for (int j = 0; j < vocab_size; j++) {
	// 										for (int k = 0; k < vocab_size; k++) {
	// 											auto trunc_sequence = truncate_tokens_max_len(tokens, max_len);
	// 											auto padding_mask = create_padding_mask(trunc_sequence, max_len);
	// 											vector<vector<float>> output = transformer.forward_with_early_exit(trunc_sequence, padding_mask, early_exit);
												
	// 											tokens[0] = (++tokens[0]) % vocab_size;
												
	// 										}
	// 										tokens[1] = (++tokens[1]) % vocab_size;
	// 									}
	// 									tokens[2] = (++tokens[2]) % vocab_size;
	// 								}
	// 								tokens[3] = (++tokens[3]) % vocab_size;
	// 							}
	// 							tokens[4] = (++tokens[4]) % vocab_size;
	// 						}
	// 						tokens[5] = (++tokens[5]) % vocab_size;
	// 					}
	// 					tokens[6] = (++tokens[6]) % vocab_size;
	// 				}
	// 				tokens[7] = (++tokens[7]) % vocab_size;
	// 			}
	// 			tokens[8] = (++tokens[8]) % vocab_size;
	// 		}
	// 		tokens[9] = (++tokens[9]) % vocab_size;
	// 	}
	// 	tokens[10] = (++tokens[10]) % vocab_size;
	// }

	
	// RNG
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distrib(3, vocab_size-1);

	// Output file
    std::ofstream outFile("entropies.txt");
	if (!outFile) {
        std::cerr << "Error opening file for writing." << std::endl;
        return 1;
    }
	for (int i = 0; i < 1000000000; i++) { // Do for 1M random sequences
		if (i % 1000 == 0) {
			std::cerr << "HEARTBEAT! i=" << i << std::endl;
		}
		// Random sequence of tokens
		for (int j = 0; j < 11; j++) {
			tokens[j] = distrib(gen);
			//std::cerr << tokens[j] << ",";
		}
		//std::cerr << std::endl;

		auto trunc_sequence = truncate_tokens_max_len(tokens, max_len);
		auto padding_mask = create_padding_mask(trunc_sequence, max_len);

		// First, run with early exit and collect entropy
		vector<vector<float>> output_early = transformer.forward_with_early_exit(trunc_sequence, padding_mask, true);
		std::vector<float> pooled_output_early = mean_pooling(output_early);
		vector<float> logits_early = linear_layer(pooled_output_early, final_weights, final_bias);
		vector<float> probabilities_early = softmax(logits_early);
		double entropy_early = (probabilities_early[0]*log2(probabilities_early[0]) + probabilities_early[1]*log2(probabilities_early[1])) * -1.0;

		// Then, run full model and collect entropy
		vector<vector<float>> output_full = transformer.forward_with_early_exit(trunc_sequence, padding_mask, false);
		std::vector<float> pooled_output_full = mean_pooling(output_full);
		vector<float> logits_full = linear_layer(pooled_output_full, final_weights, final_bias);
		vector<float> probabilities_full = softmax(logits_full);
		double entropy_full = (probabilities_full[0]*log2(probabilities_full[0]) + probabilities_full[1]*log2(probabilities_full[1])) * -1.0;

		// Finally, log to file
		for (int t: trunc_sequence) {
			outFile << t << ",";
		}
		outFile << entropy_early << "," << entropy_full;
		outFile << std::endl;
		
	}
	
		
		
	// auto trunc_sequence = truncate_tokens_max_len(tokens, max_len);
	// auto padding_mask = create_padding_mask(trunc_sequence, max_len);

	// cout << "Truncated tokens (max length " << max_len << "): ";
	// for (const auto& token : trunc_sequence) {
	//    cout << token << " ";
	// }
	// cout << "\n";

	// if (warmup_amx) {
	// 	_tile_dpbssd(1, 2, 3);
	// 	_tile_dpbssd(1, 2, 3);
	// 	_tile_dpbssd(1, 2, 3);
	// 	_tile_dpbssd(1, 2, 3);
	// 	_tile_dpbssd(1, 2, 3);
	// }

	// TIMER_INIT();
	// TIMER_START();

	// vector<vector<float>> output = transformer.forward_with_early_exit(trunc_sequence, padding_mask, early_exit);
	// std::vector<float> pooled_output = mean_pooling(output);
	// vector<float> logits = linear_layer(pooled_output, final_weights, final_bias);
	// vector<float> probabilities = softmax(logits);

	// TIMER_END();

	// Print probabilities
	// cout << "Category probabilities:\n";

	//cout << "Question: " << probabilities[0] << "\n";
	//cout << "Answer: " << probabilities[1] << "\n";
	//string prediction = (probabilities[0] > probabilities[1]) ? "Question" : "Answer";
	//std::cout << prediction << std::endl;

	//std::cerr << TIMER_VALUE() << std::endl;

return 0;
	
}
