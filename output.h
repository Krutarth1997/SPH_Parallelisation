#ifndef OUTPUT_H
#define OUTPUT_H

#include <Eigen/Dense>
#include <iostream>

void saveParticleAsVTK(const std::string& path, float timestep, int n_particles, const Eigen::MatrixXf& positions, const Eigen::MatrixXf& velocities,
const Eigen::VectorXf& pressure, const Eigen::VectorXf& density, const Eigen::VectorXf& density_corr, const Eigen::MatrixXf& gradColorfield,
const Eigen::MatrixXf& forces, const Eigen::VectorXf& randpart, const Eigen::VectorXi& wandpart, const Eigen::VectorXf& adhesion);

void saveFibreAsVTK(const std::string& path, float timestep, int n_particles, const Eigen::MatrixXf& positions);

void progressBar(float iter, float timestep);

#endif