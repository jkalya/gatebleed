#include <iostream>
#include "../include/NetworkModel.h"
#include "../include/Module.h"
#include "../include/FullyConnected.h"
#include "../include/Sigmoid.h"
#include "../include/Dropout.h"
#include "../include/SoftmaxClassifier.h"
#include "../include/MNISTDataLoader.h"
#include "../include/ReLU.h"
#include "../include/Tensor.h"
#include "../include/Conv2d.h"
#include "../include/MaxPool.h"
#include "../include/LinearLRScheduler.h"

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

using namespace std;

/*
 * Train a neural network on the MNIST data set and evaluate its performance
 */

const int TILE_ROWS = 16;
const int TILE_COLS = 64;

alignas(64) int8_t matrix_A[TILE_ROWS][TILE_COLS];
alignas(64) int8_t matrix_B[TILE_COLS][TILE_ROWS];

// Function to get the current value from the RDTSC
static inline uint64_t rdtsc() {
    unsigned int lo, hi;
    asm volatile("mfence");
    __asm__ volatile ("rdtsc" : "=a" (lo), "=d" (hi));
    asm volatile("mfence");
    return ((uint64_t)hi << 32) | lo;
}

// Function to read an MSR register using the file interface
uint64_t read_msr(int fd, uint32_t reg) {
    uint64_t value;
    pread(fd, &value, sizeof(value), reg);
    return value;
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

  for (i = 0; i < 8; ++i)
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

void initialize_amx_matrices_and_config() {
    // Fill A with 1s
    memset(matrix_A, 1, sizeof(matrix_A));

    // Fill B with fixed random numbers
    std::mt19937 rng(42); // Fixed seed
    std::uniform_int_distribution<int8_t> dist(-10, 10);
    for (int i = 0; i < TILE_COLS; ++i) {
        for (int j = 0; j < TILE_ROWS; ++j) {
            matrix_B[i][j] = dist(rng);
        }
    }

    // // Set up tile configuration
    // __tile_config cfg = {};
    // cfg.palette_id = 1;
    // cfg.colsb[0] = TILE_COLS;  cfg.rows[0] = TILE_ROWS;  // A
    // cfg.colsb[1] = TILE_ROWS;  cfg.rows[1] = TILE_COLS;  // B
    // cfg.colsb[2] = TILE_ROWS;  cfg.rows[2] = TILE_ROWS;  // C

    // _tile_loadconfig(&cfg);
}

void run_amx_tile_load() {
    _tile_loadd(3, matrix_A, 64); // Tile 3: A (16x64)
    _tile_loadd(4, matrix_B, 64); // Tile 4: B (64x16)
    _tile_zero(5);                       // Tile 5: C accumulator (16x16)
}

void run_amx_dot_product() {
    _tile_dpbssd(5, 3, 4); // C = A · B using tiles 3, 4, and 5
}

int main(int argc, char **argv) {

    // AMX init
    __tilecfg tile_data = {0};
    // Request permission to linux kernel to run AMX
    if (!set_tiledata_use())
        exit(-1);

    // Load tile configuration 
    init_tile_config (&tile_data);

    // if (argc < 2) {
    //     throw runtime_error("Please provide the data directory path as an argument");
    // }

    if (argc < 7) {
        throw runtime_error("Usage: ./program <data_path> <loader_type> <membership_label> <threshold> <extra_ops> <output_filename>");
    }

    string data_path = argv[1];
    string loader_type = argv[2];
    int membership_label = atoi(argv[3]);
    double threshold = atof(argv[4]);
    int extra_ops = atoi(argv[5]);
    string output_filename = argv[6];

    // printf("Data directory: %s\n", argv[1]);
    // string data_path = argv[1];

    printf("Loading training set... ");
    fflush(stdout);
    MNISTDataLoader train_loader(data_path + "/train-class-images.idx3-ubyte", data_path + "/train-class-labels.idx1-ubyte", 1); // (32) loading images in each batch
    printf("Loaded.\n");


    int seed = 0;
    vector<Module *> modules = {new Conv2d(1, 8, 3, 1, 0, seed), new MaxPool(2, 2), new ReLU(), new FullyConnected(1352, 30, seed), new ReLU(),
                                new FullyConnected(30, 10, seed)};
    //Module* early_exit_fc = new FullyConnected(30, 10, seed);
    Module* early_exit_fc = new FullyConnected(1352, 10, seed);  // No need for Flatten()
    auto lr_sched = new LinearLRScheduler(0.2, -0.000005);
    NetworkModel model = NetworkModel(modules, new SoftmaxClassifier(), lr_sched);
    model.add_early_exit(2, early_exit_fc);
    //model.load("network_15_epoch.txt");
    model.load("network.txt");
    bool used_full_path = false;
    model.enableEarlyExit(true);  // Enable early exit
    // int epochs = 5;
    // printf("Training for %d epoch(s).\n", epochs);
    // // Train network
    // int num_train_batches = train_loader.getNumBatches();
    // for (int k = 0; k < epochs; ++k) {
    //    printf("Epoch %d\n", k + 1);
    //    for (int i = 0; i < num_train_batches; ++i) {
    //        pair<Tensor<double>, vector<int> > xy = train_loader.nextBatch_old();
    //        double loss = model.trainStep(xy.first, xy.second);
    //        if ((i + 1) % 10 == 0) {
    //            printf("\rIteration %d/%d - Batch Loss: %.4lf", i + 1, num_train_batches, loss);
    //            fflush(stdout);
    //        }
    //    }
    //    printf("\n");
    // }
    // // Save weights
    // model.save("network.txt");

    printf("Loading testing set... ");
    fflush(stdout);
    MNISTDataLoader test_loader(data_path + "/test-class-images.idx3-ubyte", data_path + "/test-class-labels.idx1-ubyte", 1); // (32) loading images in each batch
    printf("Loaded.\n");

    MNISTDataLoader* target_loader = nullptr;
    if (loader_type == "test") {
        target_loader = &test_loader;
    } else if (loader_type == "train"){
        target_loader = &train_loader;
    } else {
        throw runtime_error("Invalid loader type.");
    }

    // model.eval();

    size_t HIST_ITERATIONS = 200*10; // 50001

    uint64_t timestamps[HIST_ITERATIONS];
    double PKG_energy_readings[HIST_ITERATIONS];
    double DRAM_energy_readings[HIST_ITERATIONS];
    bool used_full_path_log[HIST_ITERATIONS];  // Add this

    int fd;
    uint64_t msr_value;

    uint64_t Start_time, End_time, PKG_Energy, DRAM_Energy;
    double Start_PKG_Energy_Joul, End_PKG_Energy_Joul, Start_DRAM_Energy_Joul, End_DRAM_Energy_Joul;

    int junk1 = 0 , junk2 = 0, Delay = 0; // Delay = 20000000 vs 0; // Cold vs Warm
    unsigned long long start_cycles, end_cycles, elapsed_cycles;
    // Open MSR device file
    fd = open("/dev/cpu/28/msr", O_RDONLY);
    if (fd < 0) {
        printf("Failed to open /dev/cpu/28/msr\n");
        //return 0;
    }
    // Read the MSR_RAPL_POWER_UNIT register once to get units
    msr_value = read_msr(fd, MSR_RAPL_POWER_UNIT);
    // Extract energy status unit bits (bits 12:8)
    int energy_units = (msr_value >> 8) & 0x1F;
    double energy_multiplier = 1.0 / (1 << energy_units);

    for (unsigned int i = 0; i < 100; ++i) {
       pair<Tensor<double>, vector<int>> xy_new_main = target_loader->nextBatch(i);
    //    vector<int> predictions_new = model.predict(xy_new_main.first);
        vector<int> predictions_new = model.predict(xy_new_main.first, used_full_path, threshold, extra_ops);
        //printf("\predictions: %d\n", predictions_new[0]);
    }

    _tile_release (); // TILERELEASE  :  Release tile.

    init_tile_config (&tile_data);

    // Test and measure accuracy
    int hits = 0;
    int total = 0;
    printf("Testing...\n");
    //int num_test_batches = target_loader->getNumBatches();
    for (int i = 0; i < 2000; ++i) {
       pair<Tensor<double>, vector<int> > xy = target_loader->nextBatch(i);
       _mm_mfence();
       vector<int> predictions = model.predict(xy.first, used_full_path, threshold, extra_ops);
    //    for (int j = 0; j < predictions.size(); ++j) {
    //        printf("\predictions %d %d\n", predictions[j], xy.second[j]);
    //        if (predictions[j] == xy.second[j]) {
    //            hits++;
    //        }
    //    }
    //    total += xy.second.size();
        run_amx_tile_load();
        _mm_mfence();
        Start_time = rdtsc();
        _mm_mfence();
        //printf("Running illegal\n");
        run_amx_dot_product();
        _mm_mfence();
        End_time = rdtsc();

        timestamps[i] = End_time - Start_time;
        used_full_path_log[i] = used_full_path;

        // Optional: Simulate noise via delay
        _mm_mfence();
        junk1 = 0; junk2 = 0;
        start_cycles = rdtsc();
        for (int p = 0; p < Delay; p++) {
            junk2 = junk1 + p;
        }
        _mm_mfence();
        end_cycles = rdtsc();
        elapsed_cycles =  (end_cycles - start_cycles);

    }

    //Output
    //FILE *output_file = fopen("train_3_member.txt", "w");
    FILE *output_file = fopen(output_filename.c_str(), "w");
    if (output_file == NULL) {
        printf("Failed to open output file for writing\n");
    }
    for (size_t i = 0; i < 2000; i++) {
        // fprintf(output_file, "%lu\t%d\t%.2f\n",
        //     timestamps[i],
        //     //used_full_path_log[i] ? 1 : 0,
        //     //(i % 2 == 0) ? 0.01 : 0.99
        //     //1,
        //     //0.99
        //     //0,
        //     used_full_path,
        //     0.01
        // );

        fprintf(output_file, "%lu\t%d\t%.2f\t%d\t%s\t\n",
            timestamps[i],               // AMX timing cycles
            used_full_path ? 1 : 0,      // Whether full path was used
            threshold,                   // Inference threshold
            membership_label,            // 0 = non-member, 1 = member
            loader_type.c_str()         // Loader type
            );          // Data path (optional but logged)

    }
    fclose(output_file);

    printf("\n");
    
    // printf("Accuracy: %.2f%% (%d/%d)\n", ((double) hits * 100) / total, hits, total);

    // if (total_samples > 0) {
    //     double percent_entropy_printed = (samples_with_entropy_printed * 100.0) / total_samples;
    //     printf("Percentage of samples where entropy was printed (no early exit): %.2f%%\n", percent_entropy_printed);
    // }

    // new (inference)

    _tile_release (); // TILERELEASE  :  Release tile.
    return 0;
}

