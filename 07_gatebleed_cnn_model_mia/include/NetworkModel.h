//
// Created by lucas on 10/04/19.
//

#ifndef NEURAL_NET_IN_CPP_NETWORKMODEL_H
#define NEURAL_NET_IN_CPP_NETWORKMODEL_H

#include <vector>
#include "Tensor.h"
#include "Module.h"
#include "OutputLayer.h"
#include "../include/LRScheduler.h"

extern int total_samples;
extern int samples_with_entropy_printed;

/*
 * Train and test a neural network defined by Modules
 */
class NetworkModel {
private:
    std::vector<Module *> modules_;
    OutputLayer *output_layer_;
    LRScheduler* lr_scheduler_;
    bool use_early_exit = false;  // default is normal (no early exit)
    //bool used_full_path;
    std::vector<int> early_exit_points;
    std::vector<Module*> early_exit_heads;
    int iteration = 0;
public:

    NetworkModel(std::vector<Module *> &modules, OutputLayer *output_layer, LRScheduler* lr_scheduler);

    double trainStep(Tensor<double> &x, std::vector<int> &y);

    Tensor<double> forward(Tensor<double> &x);

    std::vector<int> predict_full(Tensor<double> &x);

    std::vector<int> predict_with_early_exit(Tensor<double> &x, double threshold, bool &used_full_path, int extra_ops);

    std::vector<int> predict(Tensor<double>& x, bool &used_full_path, double threshold, int extra_ops);

    void load(std::string path);

    void save(std::string path);

    void enableEarlyExit(bool enable);

    void add_early_exit(int layer_idx, Module* exit_head);

    virtual ~NetworkModel();

    void eval();
};


#endif //NEURAL_NET_IN_CPP_NETWORKMODEL_H
