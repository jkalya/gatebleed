#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <iterator>

// Existing tokenize function from your code
std::vector<int> tokenize(const std::string &sentence, const std::unordered_map<std::string, int> &vocab);

// New function to load vocabulary from a file
bool load_vocab_from_file(const std::string &vocab_file, std::unordered_map<std::string, int> &vocab);

// New function to prepare dataset from question.txt and answer.txt
//bool prepare_dataset_from_files(const std::vector<std::string> &input_files,
//                                const std::unordered_map<std::string, int> &label_map,
//                                std::vector<std::vector<int>> &data,
//                                std::vector<int> &labels,
//                                const std::unordered_map<std::string, int> &vocab);

bool prepare_dataset_from_files(const std::string &question_file,
                                const std::string &answer_file,
                                std::vector<std::vector<int>> &data,
                                std::vector<int> &labels,
                                const std::unordered_map<std::string, int> &vocab);                              

#endif
