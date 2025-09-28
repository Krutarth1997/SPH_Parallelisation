#include "updateTime.h"
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>

void updateTimeCSV(int nprocs, double elapsed_time) {
    std::string filename = "time.csv";
    std::vector<std::pair<int, double>> records;

    // Open the CSV file for reading
    std::ifstream infile(filename);
    std::string line;

    // Skip the header
    if (infile.good()) {
        std::getline(infile, line); // Skip the first line (header)
    }

    // Read the existing data and store it in a vector
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string nprocs_str, time_str;
        std::getline(ss, nprocs_str, ',');
        std::getline(ss, time_str);

        int file_nprocs = std::stoi(nprocs_str);
        double file_time = std::stod(time_str);
        records.push_back({file_nprocs, file_time});
    }
    infile.close();

    // Insert the new data in the correct position
    records.push_back({nprocs, elapsed_time});
    std::sort(records.begin(), records.end());

    // Open the CSV file for writing
    std::ofstream outfile(filename);

    // Write the header
    outfile << "nprocs,time\n";

    // Write the sorted data back to the file
    for (const auto& record : records) {
        outfile << record.first << "," << record.second << "\n";
    }

    outfile.close();
}

