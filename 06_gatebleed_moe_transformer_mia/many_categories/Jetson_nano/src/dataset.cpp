#include "dataset.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_map>
#include <vector>
#include <set>
#include <cstdlib>  // For exit()
//#include <filesystem>

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

bool load_vocab_from_file(const std::string &vocab_file, std::unordered_map<std::string, int> &vocab)
{
    std::ifstream ifs(vocab_file);
    if (!ifs.is_open()) {
        std::cerr << "Error: Could not open vocab file: " << vocab_file << std::endl;
        return false;
    }

    // We'll store tokens in a vector first so we can detect duplicates, 
    // check for [PAD]/[UNK], then if needed re-save after cleaning.
    std::vector<std::string> tokens;

    std::string line;
    while (std::getline(ifs, line)) {
        // Remove trailing/leading whitespace
        // (though leading/trailing spaces are mostly handled by std::getline anyway)
        // We'll also remove *internal* spaces with std::remove_if, but typically
        // tokens are on separate lines, so there's no spaces to keep. 
        // If your tokens can have spaces, you'd need a different approach.
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        
        if (!line.empty()) {
            tokens.push_back(line);
        }
    }
    ifs.close();

    if (tokens.empty()) {
        std::cerr << "Warning: The vocabulary file is empty or not loaded properly.\n";
        return false;
    }

    // 1) Detect duplicates
    //    We'll insert tokens into a set to find duplicates.
    std::set<std::string> unique_tokens;
    bool duplicates_found = false;

    for (auto &tk : tokens) {
        if (!unique_tokens.insert(tk).second) {
            // .second == false if the insert didn't occur (meaning it's a duplicate)
            duplicates_found = true;
        }
    }

    // 2) If duplicates exist, remove them and rewrite file
    if (duplicates_found) {
        std::cerr << "\nWARNING: Duplicate tokens found in the vocab file." << std::endl;
        std::cerr << "Removing duplicates and rewriting the vocab file: " << vocab_file << std::endl;

        // Overwrite tokens with the unique set, preserving order in some way if you like
        // but typically a set is unordered (unless you use std::set which is sorted).
        // We can re-generate a vector from the set (which is alphabetically sorted if using std::set).
        // Alternatively, you might preserve original order by using an ordered approach for duplicates.
        
        // For simplicity, let's do alphabetical. If you want original order minus duplicates,
        // you’d need a different approach.
        tokens.clear();
        for (const auto &tk : unique_tokens) {
            tokens.push_back(tk);
        }

        // Rewrite the vocab file without duplicates
        std::ofstream ofs(vocab_file, std::ios::trunc);
        if (!ofs.is_open()) {
            std::cerr << "Error: Could not rewrite vocab file (open failed): " << vocab_file << std::endl;
            return false;
        }
        for (auto &tk : tokens) {
            ofs << tk << "\n";
        }
        ofs.close();

        std::cerr << "Cleaned vocab saved. Please restart the program to load the updated vocab.\n\n";
        std::exit(EXIT_FAILURE);  // Exit the program after rewriting
    }

    // 3) Ensure [PAD] and [UNK] exist in the vocabulary
    //    If they don’t, we’ll add them at the beginning.
    //    (If you prefer them at the top, we can reorder as well.)
    bool pad_found  = false;
    bool unk_found  = false;

    for (auto &tk : tokens) {
        if (tk == "[PAD]") pad_found = true;
        if (tk == "[UNK]") unk_found = true;
    }

    // If either is missing, insert them
    if (!pad_found || !unk_found) {
        std::cerr << "\n[PAD] or [UNK] is missing. We will add them.\n";
        // We might want [PAD] = 0, [UNK] = 1, so let's enforce that:
        
        // If the user wants them strictly at the top, let's:
        //  1) remove them if they exist anywhere else,
        //  2) then re-insert them at front in correct order.
        std::vector<std::string> new_tokens;
        // Remove any existing [PAD]/[UNK] (in case partial mismatch)
        for (auto &tk : tokens) {
            if (tk != "[PAD]" && tk != "[UNK]") {
                new_tokens.push_back(tk);
            }
        }
        // Now ensure they are at the front in the correct order
        new_tokens.insert(new_tokens.begin(), "[UNK]");
        new_tokens.insert(new_tokens.begin(), "[PAD]");

        // Overwrite tokens
        tokens = new_tokens;

        // Rewrite the file
        std::ofstream ofs(vocab_file, std::ios::trunc);
        if (!ofs.is_open()) {
            std::cerr << "Error: Could not rewrite vocab file to add [PAD]/[UNK]: " << vocab_file << std::endl;
            return false;
        }
        for (auto &tk : tokens) {
            ofs << tk << "\n";
        }
        ofs.close();

        std::cerr << "Updated vocab saved with [PAD], [UNK] at the top. Please restart the program.\n\n";
        std::exit(EXIT_FAILURE);  // Exit the program after rewriting
    }

    // 4) Now build the final in-memory vocab with each token’s index
    vocab.clear();
    for (int i = 0; i < (int)tokens.size(); ++i) {
        vocab[tokens[i]] = i;
    }

    std::cout << "Loaded vocabulary of size: " << vocab.size() << " from " << vocab_file << std::endl;
    return true;
}


// This function tokenizes a sentence using the vocabulary
std::vector<int> tokenize(const std::string &sentence, const std::unordered_map<std::string, int> &vocab) {
    std::istringstream iss(sentence);
    std::string word;
    std::vector<int> tokens;

    while (iss >> word) {
        // Convert word to lowercase for consistency
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);

        // Remove punctuation from the word
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());

        // If word not found, use [UNK] token. We assume [UNK] is ID=1 in the vocab.
        if (vocab.count(word)) {
            tokens.push_back(vocab.at(word));
        } else {
            tokens.push_back(vocab.at("[UNK]")); 
        }
    }

    return tokens;
}


bool load_labels_from_file(const std::string &label_file, std::unordered_map<std::string, int> &label_map) {
    std::ifstream ifs(label_file);
    if (!ifs.is_open()) {
        std::cerr << "Error: Could not open label file: " << label_file << std::endl;
        return false;
    }

    std::string line;
    int label_id = 0;
    while (std::getline(ifs, line)) {
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        if (!line.empty()) {
            label_map[line] = label_id++;
        }
    }

    ifs.close();
    if (label_map.size() != 10) {
        std::cerr << "Error: The label file must contain exactly 10 labels." << std::endl;
        return false;
    }

    return true;
}


// Updated function
bool prepare_dataset_from_files(const std::vector<std::string> &input_files,
                                const std::unordered_map<std::string, int> &label_map,
                                std::vector<std::vector<int>> &data,
                                std::vector<int> &labels,
                                const std::unordered_map<std::string, int> &vocab) {
    data.clear();
    labels.clear();

    for (const auto &file : input_files) {
        std::ifstream ifs(file);
        if (!ifs.is_open()) {
            std::cerr << "Error: Could not open file: " << file << std::endl;
            return false;
        }

        // Extract the category name from the file name (remove .txt extension)
        std::string category_name = fs::path(file).stem().string();
        auto label_it = label_map.find(category_name);
        if (label_it == label_map.end()) {
            std::cerr << "Error: No label mapping found for file: " << file << " (category: " << category_name << ")" << std::endl;
            return false;
        }

        int label = label_it->second;
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            std::vector<int> tokens = tokenize(line, vocab);
            data.push_back(tokens);
            labels.push_back(label);
        }

        ifs.close();
    }

    std::cout << "Loaded " << data.size() << " examples across " << input_files.size() << " categories." << std::endl;
    return true;
}
