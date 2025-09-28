#ifndef INITIALIZEFIBRE_H
#define INITIALIZEFIBRE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <algorithm>

std::pair<std::vector<float>, std::vector<float>> generateVerticalFibre(std::vector<float>& positions_fibre, std::vector<float>& velocities_fibre);

std::pair<std::vector<float>, std::vector<float>> generateHorizontalFibre(std::vector<float>& positions_fibre, std::vector<float>& velocities_fibre);

std::pair<std::vector<float>, std::vector<float>> generateFlatSurface(std::vector<float>& positions_fibre, std::vector<float>& velocities_fibre);

std::tuple<std::vector<float>, std::vector<float>, int> generateSTLFibre(std::vector<float>& positions_fibre, std::vector<float>& velocities_fibre, int n_part_fibre);

bool compareVector(const std::vector<float>& a, const std::vector<float>& b);

#endif 