#ifndef INITIALIZEPARTICLES_H
#define INITIALIZEPARTICLES_H

#include <Eigen/Dense>
#include <iostream>

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> generateBoxFlow(Eigen::MatrixXf& positions, Eigen::MatrixXf& velocities);

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> generateSphericalParticles(Eigen::MatrixXf& positions, Eigen::MatrixXf& velocities);

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> generateSemiSphericalParticles(Eigen::MatrixXf& positions, Eigen::MatrixXf& velocities);

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> generateVerticalFibreFluid(Eigen::MatrixXf& positions, Eigen::MatrixXf& velocities);

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> generateNewParticlesDroplet(Eigen::MatrixXf& positions, Eigen::MatrixXf& velocities, int n_new_particles);

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> generateWettedFibreHorizontal(Eigen::MatrixXf& positions, Eigen::MatrixXf& velocities);

#endif