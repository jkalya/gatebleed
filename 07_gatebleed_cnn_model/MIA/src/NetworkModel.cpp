//
// Created by lucas on 10/04/19.
//

#include "../include/NetworkModel.h"
#include "../include/LRScheduler.h"
#include "../include/Tensor.h"

using namespace std;
               
int total_samples = 0;
int samples_with_entropy_printed = 0;

NetworkModel::NetworkModel(std::vector<Module *> &modules, OutputLayer *output_layer, LRScheduler* lr_scheduler) {
    modules_ = modules;
    lr_scheduler_ = lr_scheduler;
    output_layer_ = output_layer;
}

double NetworkModel::trainStep(Tensor<double> &x, vector<int>& y) {
    // Forward
    Tensor<double> output = forward(x);

    //Backprop
    pair<double, Tensor<double>> loss_and_cost_gradient = output_layer_->backprop(y);
    Tensor<double> chain_gradient = loss_and_cost_gradient.second;
    for (int i = (int) modules_.size() - 1; i >= 0; --i) {
        chain_gradient = modules_[i]->backprop(chain_gradient, lr_scheduler_->learning_rate);
    }
    ++iteration;
    lr_scheduler_->onIterationEnd(iteration);
    // Return loss
    return loss_and_cost_gradient.first;
}

void NetworkModel::enableEarlyExit(bool enable) {
    use_early_exit = enable;
}

Tensor<double> NetworkModel::forward(Tensor<double> &x) {
    for (auto &module : modules_) {
        x = module->forward(x);
    }
    return output_layer_->predict(x);
}

std::vector<int> NetworkModel::predict_full(Tensor<double> &x) {
    Tensor<double> output = forward(x);
    std::vector<int> predictions;
    for (int i = 0; i < output.dims[0]; ++i) {
        int argmax = -1;
        double max = -1;
        for (int j = 0; j < output.dims[1]; ++j) {
            if (output.get(i, j) > max) {
                max = output.get(i, j);
                argmax = j;
            }
        }
        predictions.push_back(argmax);
    }

    return predictions;
}

// std::vector<int> NetworkModel::predict_with_early_exit(Tensor<double> &x, double threshold) {
//     Tensor<double> intermediate = x;

//     for (int i = 0; i < modules_.size(); ++i) {
//         intermediate = modules_[i]->forward(intermediate);

//         // If this layer has an early exit head
//         for (int e = 0; e < early_exit_points.size(); ++e) {
//             if (i == early_exit_points[e]) {
//                 Tensor<double> logits = early_exit_heads[e]->forward(intermediate);

//                 std::vector<int> predictions;
//                 for (int b = 0; b < logits.dims[0]; ++b) {
//                     double max_prob = -1.0;
//                     int argmax = -1;
//                     double sum = 0.0;

//                     for (int j = 0; j < logits.dims[1]; ++j)
//                         sum += std::exp(logits.get(b, j));

//                     for (int j = 0; j < logits.dims[1]; ++j) {
//                         double prob = std::exp(logits.get(b, j)) / sum;
//                         if (prob > max_prob) {
//                             max_prob = prob;
//                             argmax = j;
//                         }
//                     }

//                     if (max_prob >= threshold) {
//                         cout <<"Predict early exit early exit\n";
//                         predictions.push_back(argmax);
//                     } else {
//                         // Not confident enough, return empty to continue full forward
//                         cout<<"Predict full early exit\n";
//                         return predict_full(x);
//                     }
//                 }

//                 return predictions;
//             }
//         }
//     }

//     return predict_full(x);  // fallback if no early exit fired
// }

// Updated predict_with_early_exit to structured early exit control
std::vector<int> NetworkModel::predict_with_early_exit(Tensor<double> &x, double threshold, bool &used_full_path, int extra_ops) {
    Tensor<double> intermediate = x;
    used_full_path = true;

    for (int i = 0; i < modules_.size(); ++i) {
        intermediate = modules_[i]->forward(intermediate);

        for (int e = 0; e < early_exit_points.size(); ++e) {
            if (i == early_exit_points[e]) {
                Tensor<double> logits = early_exit_heads[e]->forward(intermediate);

                for (int b = 0; b < logits.dims[0]; ++b) {
                    total_samples++;
                    logits.keep_amx_unit_hot();
                    double max_prob = -1.0;
                    int argmax = -1;
                    double sum = 0.0;

                    for (int j = 0; j < logits.dims[1]; ++j) {
                        logits.keep_amx_unit_hot();
                        sum += std::exp(logits.get(b, j));
                    }

                    double entropy = 0.0;

                    for (int j = 0; j < logits.dims[1]; ++j) {
                        logits.keep_amx_unit_hot();
                        double prob = std::exp(logits.get(b, j)) / sum;
                        if (prob > 0.0) {
                            entropy -= prob * std::log(prob);
                        }
                        if (prob > max_prob) {
                            max_prob = prob;
                            argmax = j;
                        }
                    }

                    logits.keep_amx_unit_hot();
                    //std::cout << "Entropy of prediction early exit: " << entropy << "\n";

                    if (entropy < 2.15266) {
                        volatile double junk = 0.0;
                        for (int d = 0; d < extra_ops; ++d) {
                            junk += std::sqrt(static_cast<double>(d)) * std::log1p(static_cast<double>(d));
                        }
                        //std::cout << "Predict using early exit\n";

                        // Report percentage so far before return
                        // double printed_ratio = 100.0 * printed_entropy_samples / total_samples;
                        // std::cout << "Percentage of samples where entropy was printed: " 
                        //         << printed_ratio << "%\n";

                        used_full_path = false;
                        return {argmax};
                    } else {
                        samples_with_entropy_printed++;
                        //std::cout << "Entropy of prediction: " << entropy << "\n";
                    }
                }
            }
        }
    }

    // Compute final prediction without repeating the forward path
    Tensor<double> final_output = output_layer_->predict(intermediate);
    std::vector<int> predictions;
    for (int i = 0; i < final_output.dims[0]; ++i) {
        int argmax = -1;
        double max = -1;
        for (int j = 0; j < final_output.dims[1]; ++j) {
            if (final_output.get(i, j) > max) {
                max = final_output.get(i, j);
                argmax = j;
            }
        }
        predictions.push_back(argmax);
    }
    // Tensor<double> final_output = output_layer_->predict(intermediate);
    // std::vector<int> predictions;

    // for (int i = 0; i < final_output.dims[0]; ++i) {
    //     double sum = 0.0;
    //     for (int j = 0; j < final_output.dims[1]; ++j) {
    //         sum += std::exp(final_output.get(i, j));
    //     }

    //     double entropy = 0.0;
    //     int argmax = -1;
    //     double max = -1;

    //     for (int j = 0; j < final_output.dims[1]; ++j) {
    //         double prob = std::exp(final_output.get(i, j)) / sum;
    //         if (prob > 0.0) {
    //             entropy -= prob * std::log(prob);
    //         }
    //         if (final_output.get(i, j) > max) {
    //             max = final_output.get(i, j);
    //             argmax = j;
    //         }
    //     }

    //     std::cout << "Entropy of prediction (full path): " << entropy << "\n";
    //     predictions.push_back(argmax);
    // }
    return predictions;
}


std::vector<int> NetworkModel::predict(Tensor<double>& x, bool &used_full_path, double threshold, int extra_ops) {
    if (use_early_exit){
        return predict_with_early_exit(x, threshold, used_full_path, extra_ops);
    } else {
        return predict_full(x);
    }
}

void NetworkModel::load(std::string path) {
    FILE *model_file = fopen(path.c_str(), "r");
    if (!model_file) {
        throw std::runtime_error("Error reading model file.");
    }
    for (auto &module : modules_) {
        module->load(model_file);
    }
}

void NetworkModel::add_early_exit(int layer_idx, Module* exit_head) {
    early_exit_points.push_back(layer_idx);
    early_exit_heads.push_back(exit_head);
}

void NetworkModel::save(std::string path) {
    FILE *model_file = fopen(path.c_str(), "w");
    if (!model_file) {
        throw std::runtime_error("Error reading model file.");
    }
    for (auto &module : modules_) {
        module->save(model_file);
    }
}

NetworkModel::~NetworkModel() {
    for (auto &module : modules_) {
        delete module;
    }
    delete output_layer_;
    delete lr_scheduler_;
}

void NetworkModel::eval() {
    for (auto &module : modules_) {
        module->eval();
    }
}
