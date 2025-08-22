#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem> // Requires C++17
#include <iomanip>    // for std::setprecision if needed

namespace fs = std::filesystem;

// Compare two binary files interpreted as arrays of float.
// Returns the number of float positions where they differ.
std::size_t compareBinaryFloatFiles(const fs::path& file1, const fs::path& file2)
{
    // Open files in binary mode
    std::ifstream ifs1(file1, std::ios::binary);
    std::ifstream ifs2(file2, std::ios::binary);

    if (!ifs1.is_open() || !ifs2.is_open()) {
        std::cerr << "Error opening files: " << file1 << " or " << file2 << std::endl;
        return 0;
    }

    // Determine file sizes
    ifs1.seekg(0, std::ios::end);
    std::streampos size1 = ifs1.tellg();
    ifs1.seekg(0, std::ios::beg);

    ifs2.seekg(0, std::ios::end);
    std::streampos size2 = ifs2.tellg();
    ifs2.seekg(0, std::ios::beg);

    // If sizes differ, we can’t do a 1:1 float comparison across entire files
    // We'll compare up to the smaller size in floats. You can handle mismatch differently if needed.
    std::streampos minSize = (size1 < size2) ? size1 : size2;

    // Number of floats in the smaller file
    std::size_t floatCount = static_cast<std::size_t>(minSize / sizeof(float));

    std::size_t diffCount = 0;

    // Read and compare float by float
    for (std::size_t i = 0; i < floatCount; ++i) {
        float f1, f2;
        ifs1.read(reinterpret_cast<char*>(&f1), sizeof(float));
        ifs2.read(reinterpret_cast<char*>(&f2), sizeof(float));

        // Exact bitwise comparison:
        if (f1 != f2) {
            ++diffCount;
        }

        // If you want to compare within some tolerance (e.g., 1e-6), do:
        // if (std::fabs(f1 - f2) > 1e-6) {
        //     ++diffCount;
        // }
    }

    return diffCount;
}

int main(int argc, char* argv[])
{
    // Usage example: ./compare_bin T1 T2
    // If you want to hardcode the paths, just replace these strings:
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_T1> <path_to_T2>" << std::endl;
        return 1;
    }

    fs::path T1 = argv[1];
    fs::path T2 = argv[2];

    // Ensure T1 and T2 exist and are directories
    if (!fs::exists(T1) || !fs::is_directory(T1)) {
        std::cerr << "Error: " << T1 << " is not a valid directory." << std::endl;
        return 1;
    }
    if (!fs::exists(T2) || !fs::is_directory(T2)) {
        std::cerr << "Error: " << T2 << " is not a valid directory." << std::endl;
        return 1;
    }

    // Iterate over files in T1
    for (const auto& entry : fs::directory_iterator(T1)) {
        if (!entry.is_regular_file()) {
            continue; // skip sub-directories or symlinks
        }

        fs::path fileT1 = entry.path();
        // We only compare .bin files (you can omit this check if you want to compare all files)
        if (fileT1.extension() == ".bin") {
            // Construct the corresponding path in T2
            fs::path fileT2 = T2 / fileT1.filename();

            // Check if the same filename exists in T2
            if (fs::exists(fileT2) && fs::is_regular_file(fileT2)) {
                // Compare data
                std::size_t diffCount = compareBinaryFloatFiles(fileT1, fileT2);

                if (diffCount == 0) {
                    std::cout << "[SAME] " << fileT1.filename() << " matches (no float differences).\n";
                } else {
                    // The file contents differ
                    std::cout << "[DIFF] " << fileT1.filename() 
                              << " differs in " << diffCount << " float positions.\n";
                }
            } else {
                // If T2 doesn't have this file or it's not a regular file, skip
                std::cout << "[SKIP] " << fileT1.filename()
                          << " does not exist (or is not a regular file) in T2.\n";
            }
        }
    }

    return 0;
}
