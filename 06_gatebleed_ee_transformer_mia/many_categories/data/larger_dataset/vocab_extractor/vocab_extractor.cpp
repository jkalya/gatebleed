#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <set>
#include <random>
#include <cctype> // For tolower

using namespace std;

// Function to clean a word by removing only ? and . and converting to lowercase
string cleanWord(string word) {
    string cleanedWord = "";
    for (char c : word) {
        if (c != '?' && c != '.') {
            cleanedWord += tolower(c);
        }
    }
    return cleanedWord;
}

// Function to perform Fisher-Yates Shuffle
void fisherYatesShuffle(vector<string>& words) {
    random_device rd;
    mt19937 g(rd());

    for (size_t i = words.size() - 1; i > 0; --i) {
        uniform_int_distribution<size_t> dist(0, i);
        size_t j = dist(g);
        swap(words[i], words[j]);
    }
}

int main() {
    ifstream inputFile("input.txt");
    ofstream outputFile("vocab_extracted.txt");

    if (!inputFile.is_open()) {
        cerr << "Error opening input file." << endl;
        return 1;
    }

    if (!outputFile.is_open()) {
        cerr << "Error opening output file." << endl;
        return 1;
    }

    set<string> uniqueWords; // Use a set to store unique words

    // Write the special tokens at the beginning
    outputFile << "[PAD]" << endl;
    outputFile << "[UNK]" << endl;

    uniqueWords.insert("[PAD]");
    uniqueWords.insert("[UNK]");

    string line;
    while (getline(inputFile, line)) {
        stringstream ss(line);
        string word;
        while (ss >> word) {
            string cleanedWord = cleanWord(word);

            // Check if the cleaned word is one of the reserved words (case-insensitive)
            if (!cleanedWord.empty() && 
                cleanedWord != "[pad]" && cleanedWord != "[unk]" &&
                cleanedWord != "[PAD]" && cleanedWord != "[UNK]" &&
                uniqueWords.find(cleanedWord) == uniqueWords.end()) {

                uniqueWords.insert(cleanedWord);
                outputFile << cleanedWord << endl; // New line after each word
            }
        }
    }

    inputFile.close();
    outputFile.close();

    // Open the file we just created to read its content
    ifstream extractedFile("vocab_extracted.txt");
    ofstream randomizedFile("vocab_extracted_randomized.txt");

    if (!extractedFile.is_open()) {
        cerr << "Error opening vocab_extracted.txt." << endl;
        return 1;
    }

    if (!randomizedFile.is_open()) {
        cerr << "Error opening vocab_extracted_randomized.txt." << endl;
        return 1;
    }

    vector<string> words;
    string specialLine;

    // Read the special tokens first
    if (getline(extractedFile, specialLine)) words.push_back(specialLine);
    if (getline(extractedFile, specialLine)) words.push_back(specialLine);

    string word;
    while (getline(extractedFile, word)) {
        words.push_back(word);
    }

    // Separate the special tokens
    vector<string> specialTokens(words.begin(), words.begin() + 2);
    vector<string> normalWords(words.begin() + 2, words.end());

    // Shuffle the normal words
    fisherYatesShuffle(normalWords);

    // Write the special tokens back
    randomizedFile << specialTokens[0] << endl;
    randomizedFile << specialTokens[1] << endl;

    // Write the shuffled words to the new file
    for (const string& shuffledWord : normalWords) {
        randomizedFile << shuffledWord << endl;
    }

    extractedFile.close();
    randomizedFile.close();

    return 0;
}
