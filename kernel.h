#ifndef KERNEL_H
#define KERNEL_H

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>

float CubicSplineKernel(float dist, float smooth);
float CubicSplineDifKernel(float dist, float smooth);
float CubicSplineDif2Kernel(float dist, float smooth);
float SpikyKernel(float dist, float smooth);
float SpikyDifKernel(float dist, float smooth);
float SpikyDif2Kernel(float dist, float smooth);
float Poly6Kernel(float dist, float smooth);
float Poly6DifKernel(float dist, float smooth);
float Poly6Dif2Kernel(float dist, float smooth);
float CohesionKernel(float dist, float smooth);
float AdhesionKernel(float dist, float smooth);

float DensityKernel(float dist, float smooth);
float DensityCorrectedKernel(float dist, float smooth);
Eigen::MatrixXf DensityKernelParticle(const std::vector<std::vector<int>>& neighbor_ids, Eigen::VectorXf& densities, const std::vector<std::vector<float>>& distances, float SMOOTHING_LENGTH, float PARTICLE_MASS, int start, int end);
Eigen::MatrixXf DensityCorrectedParticle(const std::vector<std::vector<int>>& neighbor_ids, Eigen::VectorXf& densities_corr, const std::vector<std::vector<float>>& distances, float SMOOTHING_LENGTH, float PARTICLE_MASS, Eigen::VectorXf densities, int start, int end);
Eigen::MatrixXf CheckBoundaryParticle(const std::vector<std::vector<int>>& neighbor_idsFluidFibre, Eigen::VectorXf& randpart, Eigen::VectorXf& densities, const std::vector<std::vector<float>>& distancesFluidFibre, float SMOOTHING_LENGTH_FIBRE, float PARTICLE_MASS_FIBRE, int start, int end);
Eigen::MatrixXf ColorFieldKernel(const std::vector<std::vector<int>>& neighbor_ids, Eigen::MatrixXf& gradColorField, const Eigen::MatrixXf& positions, const std::vector<std::vector<float>>& distances, float SMOOTHING_LENGTH, float PARTICLE_MASS, Eigen::VectorXf densities, int start, int end);
float pressureKernel(float dist, float smooth);
float viscosityKernel(float dist, float smooth);
float densityKernel(float dist, float smooth);

#endif 