#include "positional_encoding.h"
#include <cmath>
#include <iostream>
#include <fstream>
using namespace std;

// Constructor
PositionalEncoding::PositionalEncoding(int max_len, int d_model)
{
// "max_len" Maximum sequence length (number of tokens in a single input)
// "d_model" set the "resolution" of the positional encoding space. 
// Like a meter stick with 128 evenly spaced lines, 
// this determines how finely token positions are encoded.    

    pos_encoding = std::vector<std::vector<float>>(max_len, std::vector<float>(d_model, 0.0));
    ofstream sin_file("sin.dat");
    ofstream cos_file("cos.dat");
    cout << "PositionalEncoding Constructor" << endl;

    for (int pos = 0; pos < max_len; ++pos)
    {
        for (int i = 0; i < d_model; ++i)
        {
            if (i % 2 == 0)
            {
                pos_encoding[pos][i] = std::sin(pos / std::pow(10000.0, i / (float)d_model));
                //cout << "sin pos_encoding[" << pos << "][" << i << "]: " << pos_encoding[pos][i] << endl;
                sin_file << pos << " " << i << " " << pos_encoding[pos][i] << endl; // Write to sin.dat
            }
            else
            {
                pos_encoding[pos][i] = std::cos(pos / std::pow(10000.0, (i - 1) / (float)d_model));
                // cout << "cos pos_encoding[" << pos << "][" << i << "]: " << pos_encoding[pos][i] << endl;
                cos_file << pos << " " << i << " " << pos_encoding[pos][i] << endl; // Write to cos.dat
            }
        }
    }
    sin_file.close();
    cos_file.close();
}

// Add positional encoding to input embeddings
std::vector<std::vector<float>> PositionalEncoding::add_positional_encoding(const std::vector<std::vector<float>>& input) {
    std::vector<std::vector<float>> result = input;

    for (size_t pos = 0; pos < input.size(); ++pos) {
        for (size_t i = 0; i < input[0].size(); ++i) {
            result[pos][i] += pos_encoding[pos][i];
        }
    }

    return result;
}

// Backward function
std::vector<std::vector<float>> PositionalEncoding::backward(
    const std::vector<std::vector<float>>& grad_output
) {
    // Positional encoding is not trainable, so the gradient is directly passed through
    return grad_output;
}