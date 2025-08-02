//
// Created by lucas on 11/04/19.
//

#ifndef NEURAL_NET_IN_CPP_RELU_H
#define NEURAL_NET_IN_CPP_RELU_H


#include "Tensor.h"
#include "Module.h"

class ReLU : public Module{
private:
    Tensor<double> input_;
    Tensor<double> product_;
public:
    ReLU();

    Tensor<double> &forward(Tensor<double> &input) override;

    Tensor<double> backprop(Tensor<double> chainGradient, double learning_rate) override;

    void load(FILE *file_model) override;

    void save(FILE *file_model) override;
};


#endif //NEURAL_NET_IN_CPP_RELU_H
