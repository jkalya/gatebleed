//
// Created by lucas on 12/04/19.
//

#include "../include/Tensor.h"
#include <cstring> // memset

#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdbool.h>

// Function to get the current value from the RDTSC
static inline uint64_t rdtsc() {
    unsigned int lo, hi;
    asm volatile("mfence");
    __asm__ volatile ("rdtsc" : "=a" (lo), "=d" (hi));
    asm volatile("mfence");
    return ((uint64_t)hi << 32) | lo;
}

template
class Tensor<int>;

template
class Tensor<float>;

template
class Tensor<double>;


template<typename T>
void Tensor<T>::zero() {
    memset(data_, 0, sizeof(T) * size_);
}

template<typename T>
T Tensor<T>::get(int i, int j) {
    assert(num_dims == 2);
    return data_[j + i * dims[1]];
}

template<typename T>
T Tensor<T>::get(int i) {
    assert(num_dims == 1);
    return data_[i];
}

template<typename T>
void Tensor<T>::set(int i, int j, T value) {
    assert(num_dims == 2);
    data_[j + i * dims[1]] = value;
}

template<typename T>
void Tensor<T>::set(int i, T value) {
    data_[i] = value;
}


template<typename T>
void Tensor<T>::add(int i, T value) {
    data_[i] += value;
}

template<typename T>
void Tensor<T>::view(int new_num_dims, int *new_dims) {
    assert(new_num_dims > 0 && new_num_dims <= 4);
    this->num_dims = new_num_dims;
    std::copy(new_dims, new_dims + 4, this->dims);
}

template<typename T>
Tensor<T>::Tensor(int num_dims, int const *dims) {
    assert(num_dims > 0 && num_dims <= 4);
    int size = 1;
    for (int i = 0; i < num_dims; ++i) {
        size *= dims[i];
        this->dims[i] = dims[i];
    }
    size_ = size;
//    std::shared_ptr<T[]> data_sp(new T[size_]);
    T *data_sp = new T[size_];
    data_ = data_sp;
    this->num_dims = num_dims;
}

template<typename T>
T Tensor<T>::get(int i, int j, int k) {
    assert(num_dims == 3);
    return data_[k + j * dims[2] + i * dims[1] * dims[2]];
}

template<typename T>
T Tensor<T>::get(int i, int j, int k, int l) {
    assert(num_dims == 4);
    return data_[l + k * dims[3] + j * dims[2] * dims[3] + i * dims[1] * dims[2] * dims[3]];
}

template<typename T>
void Tensor<T>::set(int i, int j, int k, T value) {
    assert(num_dims == 3);
    data_[k + j * dims[2] + i * dims[1] * dims[2]] = value;
}

template<typename T>
void Tensor<T>::set(int i, int j, int k, int l, T value) {
    assert(num_dims == 4);
    data_[l + k * dims[3] + j * dims[2] * dims[3] + i * dims[1] * dims[2] * dims[3]] = value;
}

template<typename T>
void Tensor<T>::add(int i, int j, int k, int l, T value) {
    assert(num_dims == 4);
    data_[l + k * dims[3] + j * dims[2] * dims[3] + i * dims[1] * dims[2] * dims[3]] += value;
}

template<typename T>
Tensor<T>::Tensor(const Tensor<T> &other) : size_(other.size_), num_dims(other.num_dims),
                                            data_(new T[other.size_]) {
    std::copy(other.data_, other.data_ + size_, data_);
    std::copy(other.dims, other.dims + 4, dims);
}

template<typename T>
Tensor<T>::~Tensor() {
    //delete[] data_;
}


template<typename T>
Tensor<T> Tensor<T>::matmul(Tensor<T> other) {
    //uint64_t Start_time, End_time;
    //_mm_mfence();
    //Start_time = rdtsc();
    //printf("hi mtmul %d %d %d\n",this->dims[0],other.dims[1],other.dims[0]);

    assert(num_dims == 2 && other.num_dims == 2);
    assert(dims[1] == other.dims[0]);

    int new_dims[] = {dims[0], other.dims[1]};
    Tensor<T> product(2, new_dims);

    size_t rows = dims[0];
    size_t cols = other.dims[1];
    size_t inner_dim = other.dims[0];
    //std::cout << "rows: " << rows << " - cols: "  << cols << " - inner_dim: " << inner_dim << ".\n";
    size_t rows_split = (rows + 15) / 16;
    size_t cols_split = (cols + 15) / 16;
    size_t inner_split = (inner_dim + 31) / 32;
    //std::cout << "rows_split: " << rows_split << " - cols_split: "  << cols_split << " - inner_split: " << inner_split << ".\n";

    float value;
    uint32_t bits;
    //uint16_t upperBits;
    uint16_t Tile_A_Values[16*32];
    uint16_t Tile_B_Values[16*32];
    float Tile_Res_Values[16*16];
    //int a_zero_count = 0, b_zero_count = 0, a_total_zero_count = 0, b_total_zero_count = 0;

    //uint64_t start_cycles, end_cycles, elapsed_cycles;
    
    for (size_t r = 0; r < rows_split; ++r) {
        for (size_t c = 0; c < cols_split; ++c) {
            //std::fill(std::begin(Tile_Res_Values), std::end(Tile_Res_Values), (float)0);
            //_tile_loadd (0, Tile_Res_Values, 64);   // Load zeros into tile 0
            _tile_zero(0);
            for (size_t in = 0; in < inner_split; ++in) {
                // a_zero_count = 0; b_zero_count = 0; a_total_zero_count = 0; b_total_zero_count = 0;
                // preparing Tile1 data from Matrix a
                std::fill(std::begin(Tile_A_Values), std::end(Tile_A_Values), (uint16_t)0);
                for (size_t i = 0; i < std::min((size_t)16, rows-r*16); ++i) {
                    for (size_t j = 0; j < std::min((size_t)32, inner_dim-in*32); ++j) {
                        value = static_cast<float>(this->get(r*16+i,in*32+j));
                        //value = static_cast<float>(0.0);
                        //a_zero_count = (this->get(r*16+i,in*32+j) == 0) ? a_zero_count + 1 : a_zero_count;
                        //a_total_zero_count = a_total_zero_count + 1;
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
                        value = static_cast<float>(other.get(in*32+i,c*16+j));
                        //value = static_cast<float>(0.0);
                        //b_zero_count = (other.get(in*32+i,c*16+j) == 0) ? b_zero_count + 1 : b_zero_count;
                        //b_total_zero_count = b_total_zero_count + 1;
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
                    product.set(r*16+i, c*16+j, static_cast<double>(Tile_Res_Values[i*16+j]));
                }
            }
        }
    }
    //std::cout << a_zero_count << "\t" << a_total_zero_count << "\t" << b_zero_count << "\t" << b_total_zero_count <<  "\n";
    //_mm_mfence();
    //End_time = rdtsc();
    //printf("hi matmul time: %u\n",End_time-Start_time);
    return product;
}

template<typename T>
Tensor<T> Tensor<T>::matmul_old(Tensor<T> other) {
    //printf("hi mtmul %d %d %d\n",this->dims[0],other.dims[1],other.dims[0]);
    //uint64_t Start_time, End_time;
    //_mm_mfence();
    //Start_time = rdtsc();
    assert(num_dims == 2 && other.num_dims == 2);
    assert(dims[1] == other.dims[0]);

    int new_dims[] = {dims[0], other.dims[1]};
    Tensor<T> product(2, new_dims);
    for (int i = 0; i < this->dims[0]; ++i) {
        for (int j = 0; j < other.dims[1]; ++j) {
            T value = 0;
            for (int k = 0; k < other.dims[0]; ++k) {
                value += this->get(i, k) * other.get(k, j);
            }
            product.set(i, j, value);
        }
    }
    //_mm_mfence();
    //End_time = rdtsc();
    //printf("hi matmul time: %u\n",End_time-Start_time);
    return product;
}

template<typename T>
Tensor<T> Tensor<T>::matrixTranspose() {
    assert(num_dims == 2);
    int new_dims[] = {dims[1], dims[0]};
    Tensor<T> transpose(num_dims, new_dims);
    for (int i = 0; i < dims[0]; ++i) {
        for (int j = 0; j < dims[1]; ++j) {
            transpose.set(j, i, get(i, j));
        }
    }

    return transpose;
}


template<typename T>
Tensor<T> Tensor<T>::relu() {
    Tensor<T> result(num_dims, dims);
    for (int i = 0; i < size_; ++i) {
        T x = data_[i];
        result.data_[i] = x > 0 ? x : 0;
    }

    return result;
}

template<typename T>
T sigmoid(T x) {
    return 1.0 / (1.0 + exp(-x));
}

template<typename T>
Tensor<T> Tensor<T>::sigmoid() {
    Tensor<T> result(num_dims, dims);
    for (int i = 0; i < size_; ++i) {
        T x = data_[i];
        result.data_[i] = ::sigmoid(x);
    }

    return result;
}

template<typename T>
T sigmoidPrime(T x) {
    return sigmoid(x) * (1.0 - sigmoid(x));
}

template<typename T>
Tensor<T> Tensor<T>::sigmoidPrime() {
    Tensor<T> result(num_dims, dims);
    for (int i = 0; i < size_; ++i) {
        T x = data_[i];
        result.data_[i] = ::sigmoidPrime(x);
    }

    return result;
}

template<typename T>
T Tensor<T>::sum() {
    T total = 0;
    for (int i = 0; i < size_; ++i) {
        total += data_[i];
    }
    return 0;
}

template<typename T>
Tensor<T> Tensor<T>::softmax() {
    assert(num_dims == 2);
    //Softmax with max trick to avoid overflows
    int rows = dims[0], cols = dims[1];
    Tensor<T> probabilities(2, dims);
    for (int i = 0; i < rows; ++i) {
        T row_max = -1; // useless value so my IDE stops screaming at me, will always be replaced
        for (int j = 0; j < cols; ++j) {
            if (j == 0 || get(i, j) > row_max) {
                row_max = get(i, j);
            }
        }

        T denominator = 0;
        for (int j = 0; j < cols; ++j) {
            T x = get(i, j);
            denominator += exp(get(i, j) - row_max);
        }


        for (int j = 0; j < cols; ++j) {
            probabilities.set(i, j, exp(get(i, j) - row_max) / denominator);
        }

    }
    return probabilities;
}

template<typename T>
Tensor<T> Tensor<T>::reluPrime() {
    Tensor<T> prime(num_dims, dims);
    for (int i = 0; i < size_; ++i) {
        prime.data_[i] = data_[i] > 0 ? 1 : 0;
    }
    return prime;
}

template<typename T>
Tensor<T> Tensor<T>::operator+(Tensor<T> &other) {
    if (other.num_dims == 1 && other.size_ == this->dims[1] && num_dims == 2) {
        // if other is a 1d tensor and this is a 2d tensor
        Tensor<T> sum(num_dims, dims);
        for (int k = 0; k < this->dims[0]; ++k) {
            for (int j = 0; j < this->dims[1]; ++j) {
                sum.set(k, j, get(k, j) + other.get(j));
            }
        }


        return sum;
    } else if (other.num_dims == num_dims && other.size_ == size_) {
        Tensor<T> sum(num_dims, dims);
        for (int i = 0; i < size_; ++i) {
            sum.data_[i] = data_[i] + other.data_[i];
        }
        return sum;
    }
    throw std::logic_error("Undefined sum");
}


template<typename T>
Tensor<T> Tensor<T>::operator*(Tensor<T> other) {
    assert(size_ == other.size_);
    Tensor<T> product(num_dims, dims);
    for (int i = 0; i < size_; ++i) {
        product.data_[i] = data_[i] * other.data_[i];
    }
    return product;
}

template<typename T>
Tensor<T> Tensor<T>::operator*(T multiplier) {
    Tensor<T> product(num_dims, dims);
    for (int i = 0; i < size_; ++i) {
        product.data_[i] = data_[i] * multiplier;
    }
    return product;
}

template<typename T>
Tensor<T> Tensor<T>::operator/(T divisor) {
    Tensor<T> quotient(num_dims, dims);
    for (int i = 0; i < size_; ++i) {
        quotient.data_[i] = data_[i] / divisor;
    }
    return quotient;
}

template<typename T>
Tensor<T> Tensor<T>::operator-=(Tensor<T> difference) {
    assert(size_ == difference.size_);
    for (int i = 0; i < size_; ++i) {
        data_[i] = data_[i] - difference.data_[i];
    }
    return *this;
}

template<typename T>
Tensor<T> Tensor<T>::columnWiseSum() {
    assert(num_dims == 2);
    int rows = dims[0], cols = dims[1];
    int sum_dims[] = {cols};
    Tensor<T> sum(1, sum_dims);
    for (int i = 0; i < cols; ++i) {
        T total = 0;
        for (int j = 0; j < rows; ++j) {
            total += get(j, i);
        }
        sum.set(i, total);
    }
    return sum;
}

template<>
void
Tensor<double>::randn(std::default_random_engine generator, std::normal_distribution<double> distribution,
                      double multiplier) {
    for (int i = 0; i < size_; ++i) {
        data_[i] = distribution(generator) * multiplier;
    }
}

template<>
void Tensor<double>::print() {
    if (num_dims == 2) {
        int rows = dims[0], cols = dims[1];
        std::cout << "Tensor2D (" << rows << ", " << cols << ")\n[";
        for (int i = 0; i < rows; ++i) {
            if (i != 0) std::cout << " ";
            std::cout << "[";
            for (int j = 0; j < cols; ++j) {
                if (j == (cols - 1)) {
                    printf("%.18lf", get(i, j));
                } else {
                    printf("%.18lf ", get(i, j));
                }

            }
            if (i == (rows - 1)) {
                std::cout << "]]\n";
            } else {
                std::cout << "]\n";
            }
        }
    } else {
        printf("Tensor%dd (", num_dims);
        for (int i = 0; i < num_dims; ++i) {
            printf("%d", dims[i]);
            if (i != (num_dims - 1)) {
                printf(",");
            }
        }
        printf(")\n[");
        for (int j = 0; j < size_; ++j) {
            printf("%lf ", data_[j]);
        }
        printf("]\n");
    }
}

template<typename T>
Tensor<T> &Tensor<T>::operator=(const Tensor<T> &other) {
    if (this != &other) {
        T *new_data = new T[other.size_];
        std::copy(other.data_, other.data_ + other.size_, new_data);
        if (size_ != -1) {
            delete[] data_;
        }
        size_ = other.size_;
        std::copy(other.dims, other.dims + 4, dims);
        num_dims = other.num_dims;
        data_ = new_data;
    }

    return *this;
}

template<typename T>
void Tensor<T>::dropout(std::default_random_engine generator, std::uniform_real_distribution<> distribution, double p) {
    for (int i = 0; i < size_; ++i) {
        data_[i] = (distribution(generator) < p) / p;
    }
}

template<typename T>
Tensor<T> Tensor<T>::convolve2d_old(Tensor<T> kernels, int stride, int padding, Tensor<T> bias) {
    //printf("hi convolve2d kernels %d %d %d %d\n",kernels.dims[0],kernels.dims[1],kernels.dims[2],kernels.dims[3]); // 8 1 3 3
    //printf("hi convolve2d dim %d %d %d %d\n",dims[0],dims[1],dims[2],dims[3]); // 1 1 28 28
    assert(kernels.dims[1] == dims[1]);
    int w = ((dims[3] + 2 * padding - (kernels.dims[3] - 1) - 1) / stride) + 1; // 26
    int h = ((dims[2] + 2 * padding - (kernels.dims[2] - 1) - 1) / stride) + 1; // 26
    int result_dims[] = {dims[0], kernels.dims[0], h, w};
    Tensor<T> output(4, result_dims);
    for (int i = 0; i < dims[0]; ++i) { // for each image in the batch (1)
        for (int j = 0; j < kernels.dims[0]; ++j) { // For each output volume (8:each kernel) 
            for (int k = 0; k < h; ++k) { // for each vertical k in the output volume (26)
                for (int l = 0; l < w; ++l) { // for each horizontal l in the output volume (26)
                    int im_si = stride * k - padding;
                    int im_sj = stride * l - padding;
                    T total = 0;
                    for (int m = 0; m < kernels.dims[1]; ++m) { // for each filter channel (1)
                        for (int n = 0; n < kernels.dims[2]; ++n) { // (3)
                            for (int o = 0; o < kernels.dims[3]; ++o) { // (3)
                                int x = im_si + n, y = im_sj + o;
                                if (x < 0 || x >= dims[2] || y < 0 || y >= dims[3])
                                    continue; // if it is a padding region, skip (add 0)
                                T a = get(i, m, x, y);
                                T b = kernels.get(j, m, n, o);
                                total += a * b;
                            }
                        }
                    }
                    output.set(i, j, k, l, total + bias.get(j));
                }
            }
        }
    }
    return output;
}

template<typename T>
Tensor<T> Tensor<T>::convolve2d(Tensor<T> kernels, int stride, int padding, Tensor<T> bias) {
    //std::cout << kernels.dims[1] << " " << dims[1] << std::endl;
    assert(kernels.dims[1] == dims[1]);
    this->keep_amx_unit_hot();
    int w = ((dims[3] + 2 * padding - (kernels.dims[3] - 1) - 1) / stride) + 1;
    this->keep_amx_unit_hot();
    int h = ((dims[2] + 2 * padding - (kernels.dims[2] - 1) - 1) / stride) + 1;
    this->keep_amx_unit_hot();
    int result_dims[] = {dims[0], kernels.dims[0], h, w};
    Tensor<T> output(4, result_dims);
    this->keep_amx_unit_hot();

    for (int i = 0; i < dims[0]; ++i) {  // for each image in the batch (1)
        int im2col_dims[] = {kernels.dims[1] * kernels.dims[2] * kernels.dims[3],h * w};
        Tensor<T> im2col_matrix(2, im2col_dims);
        this->keep_amx_unit_hot();
        for (int j = 0; j < h; ++j) { // for each vertical j in the output volume (26)
            this->keep_amx_unit_hot();
            for (int k = 0; k < w; ++k) {  // for each horizontal k in the output volume (26)
                this->keep_amx_unit_hot();
                int im_si = stride * j - padding;
                int im_sj = stride * k - padding;
                this->keep_amx_unit_hot();
                int col_index = 0;

                for (int m = 0; m < kernels.dims[1]; ++m) {  // For each channel in the kernel (1)
                    this->keep_amx_unit_hot();
                    for (int n = 0; n < kernels.dims[2]; ++n) { // (3)
                        this->keep_amx_unit_hot();
                        for (int o = 0; o < kernels.dims[3]; ++o) { // (3)
                            this->keep_amx_unit_hot();
                            int x = im_si + n, y = im_sj + o;
                            T value = 0;
                            if (x >= 0 && x < dims[2] && y >= 0 && y < dims[3]) {
                                this->keep_amx_unit_hot();
                                value = get(i, m, x, y);
                            }
                            this->keep_amx_unit_hot();
                            im2col_matrix.set(col_index++,j * w + k, value);
                        }
                    }
                }
            }
        }
        this->keep_amx_unit_hot();
        // Reshape kernels to 2D matrix
        int kernel_dim[] = {kernels.dims[0], kernels.dims[1] * kernels.dims[2] * kernels.dims[3]};
        Tensor<T> reshaped_kernels(2, kernel_dim);
        this->keep_amx_unit_hot();
        for (int j = 0; j < kernels.dims[0]; ++j) {  // For each output channel (8)
            this->keep_amx_unit_hot();
            int index = 0;
            for (int m = 0; m < kernels.dims[1]; ++m) {  // For each channel (1)
                this->keep_amx_unit_hot();
                for (int n = 0; n < kernels.dims[2]; ++n) { // 3
                    this->keep_amx_unit_hot();
                    for (int o = 0; o < kernels.dims[3]; ++o) { // 3
                        this->keep_amx_unit_hot();
                        reshaped_kernels.set(j, index++, kernels.get(j, m, n, o));
                    }
                }
            }
        }

        // Perform matrix multiplication
        this->keep_amx_unit_hot();
        Tensor<T> conv_result = reshaped_kernels.matmul(im2col_matrix);

        // Reshape conv_result back to output format
        for (int j = 0; j < kernels.dims[0]; ++j) {  // For each output channel (8)
            this->keep_amx_unit_hot();
            for (int k = 0; k < h; ++k) { // 26
                this->keep_amx_unit_hot();
                for (int l = 0; l < w; ++l) { // 26
                    this->keep_amx_unit_hot();
                    T result_value = conv_result.get(j, k * w + l) + bias.get(j);
                    this->keep_amx_unit_hot();
                    output.set(i, j, k, l, result_value);
                }
            }
        }
    }
    this->keep_amx_unit_hot();
    return output;
}

template<typename T>
void Tensor<T>::keep_amx_unit_hot() {
#ifdef __AMX_BF16__
    uint16_t Tile_A_Values[16 * 32] = {0};
    uint16_t Tile_B_Values[16 * 32] = {0};
    float Tile_Res_Values[16 * 16] = {0.0f};

    _tile_zero(0);                         // Zero accumulator tile
    _tile_loadd(1, Tile_A_Values, 64);           // Load all-zero tile A into tile1
    _tile_loadd(2, Tile_B_Values, 64);           // Load all-zero tile B into tile2
    _tile_dpbf16ps(0, 1, 2);   // Perform dummy tile multiplication
    _tile_stored(0, Tile_Res_Values, 64);        // Store result (not used)
#endif
}



//template<typename T>
//Tensor<T> Tensor<T>::convolve2d_old(Tensor<T> kernels, int stride, int padding, Tensor<T> bias) {
//    assert(kernels.dims[1] == dims[1]);
//
//    int w = ((dims[3] + 2 * padding - (kernels.dims[3] - 1) - 1) / stride) + 1;
//    int h = ((dims[2] + 2 * padding - (kernels.dims[2] - 1) - 1) / stride) + 1;
//    int result_dims[] = {dims[0], kernels.dims[0], h, w};
//    Tensor<T> output(4, result_dims);
//
//    for (int i = 0; i < dims[0]; ++i) {  // for each image in the batch (1)
//        int im2col_dims[] = {h * w, kernels.dims[1] * kernels.dims[2] * kernels.dims[3]};
//        Tensor<T> im2col_matrix(2, im2col_dims);
//
//        for (int j = 0; j < h; ++j) { // for each vertical j in the output volume (26)
//            for (int k = 0; k < w; ++k) {  // for each horizontal k in the output volume (26)
//                int im_si = stride * j - padding;
//                int im_sj = stride * k - padding;
//                int col_index = 0;
//
//                for (int m = 0; m < kernels.dims[1]; ++m) {  // For each channel in the kernel (1)
//                    for (int n = 0; n < kernels.dims[2]; ++n) { // (3)
//                        for (int o = 0; o < kernels.dims[3]; ++o) { // (3)
//                            int x = im_si + n, y = im_sj + o;
//                            T value = 0;
//                            if (x >= 0 && x < dims[2] && y >= 0 && y < dims[3]) {
//                                value = get(i, m, x, y);
//                            }
//                            im2col_matrix.set(j * w + k, col_index++, value);
//                        }
//                    }
//                }
//            }
//        }
//
//        // Reshape kernels to 2D matrix
//        int kernel_dim[] = {kernels.dims[0], kernels.dims[1] * kernels.dims[2] * kernels.dims[3]};
//        Tensor<T> reshaped_kernels(2, kernel_dim);
//        for (int j = 0; j < kernels.dims[0]; ++j) {  // For each output channel (8)
//            int index = 0;
//            for (int m = 0; m < kernels.dims[1]; ++m) {  // For each channel (1)
//                for (int n = 0; n < kernels.dims[2]; ++n) { // 3
//                    for (int o = 0; o < kernels.dims[3]; ++o) { // 3
//                        reshaped_kernels.set(j, index++, kernels.get(j, m, n, o));
//                    }
//                }
//            }
//        }
//
//        // Perform matrix multiplication
//        Tensor<T> conv_result = reshaped_kernels.matmul(im2col_matrix);
//
//        // Reshape conv_result back to output format
//        for (int j = 0; j < kernels.dims[0]; ++j) {  // For each output channel (8)
//            for (int k = 0; k < h; ++k) { // 26
//                for (int l = 0; l < w; ++l) { // 26
//                    T result_value = conv_result.get(j, k * w + l) + bias.get(j);
//                    output.set(i, j, k, l, result_value);
//                }
//            }
//        }
//    }
//
//    return output;
//}