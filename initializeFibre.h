#ifndef INITIALIZEFIBRE_H
#define INITIALIZEFIBRE_H

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> generateVerticalFibre(Eigen::MatrixXf& positions_fibre, Eigen::MatrixXf& velocities_fibre);

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> generateHorizontalFibre(Eigen::MatrixXf& positions_fibre, Eigen::MatrixXf& velocities_fibre);

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> generateFlatSurface(Eigen::MatrixXf& positions_fibre, Eigen::MatrixXf& velocities_fibre);

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> generateSTLFibre(Eigen::MatrixXf& positions_fibre, Eigen::MatrixXf& velocities_fibre, int* n_part_fibre);

bool compareEigenVector(const Eigen::Vector3f& a, const Eigen::Vector3f& b);

#endif 