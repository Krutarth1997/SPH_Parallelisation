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
Eigen::MatrixXf DensityKernelParticle(int n_particles, const std::vector<std::vector<int>>& neighbor_ids, const std::vector<std::vector<float>>& distances, float SMOOTHING_LENGTH, float PARTICLE_MASS);
Eigen::MatrixXf DensityCorrectedParticle(int n_particles, const std::vector<std::vector<int>>& neighbor_ids, const std::vector<std::vector<float>>& distances, float SMOOTHING_LENGTH, float PARTICLE_MASS, Eigen::VectorXf densities);
Eigen::MatrixXf CheckBoundaryParticle(int n_particles, const std::vector<std::vector<int>>& neighbor_idsFluidFibre, const std::vector<std::vector<float>>& distancesFluidFibre, float SMOOTHING_LENGTH_FIBRE, float PARTICLE_MASS_FIBRE);
Eigen::MatrixXf ColorFieldKernel(int n_particles, const std::vector<std::vector<int>>& neighbor_ids, const Eigen::MatrixXf& positions, const std::vector<std::vector<float>>& distances, float SMOOTHING_LENGTH, float PARTICLE_MASS, Eigen::VectorXf densities);
float pressureKernel(float dist, float smooth);
float viscosityKernel(float dist, float smooth);
float densityKernel(float dist, float smooth);

#endif 