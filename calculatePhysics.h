#ifndef CALCULATEPHYSICS_H
#define CALCULATEPHYSICS_H

#include <Eigen/Dense>
#include <vector>
#include <iostream>

void calculateDensities(std::vector<std::vector<int>>& int_neighbor, Eigen::VectorXf& int_densities, float int_smoothing_length, std::vector<std::vector<float>>& int_distances, float int_particle_mass, int start, int end);

Eigen::MatrixXf updateVelocities(Eigen::MatrixXf& int_velocities, Eigen::MatrixXf& int_forces, float int_timestep, int start, int end);

Eigen::MatrixXf updatePositions(Eigen::MatrixXf& int_positions, Eigen::MatrixXf& int_velocities, float int_timestep, int start, int end);

float calculateTimestepLength(Eigen::MatrixXf& int_velocities, Eigen::MatrixXf& int_forces, int int_particles, float int_smooth, float Co);

#endif
