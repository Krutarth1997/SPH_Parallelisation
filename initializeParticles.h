#ifndef INITIALIZEPARTICLES_H
#define INITIALIZEPARTICLES_H

#include <Eigen/Dense>
#include <iostream>

Eigen::MatrixXf generateBoxFlow(Eigen::MatrixXf& positions, Eigen::MatrixXf& velocities);

Eigen::MatrixXf generateSphericalParticles(Eigen::MatrixXf& positions, Eigen::MatrixXf& velocities);

Eigen::MatrixXf generateSemiSphericalParticles(Eigen::MatrixXf& positions, Eigen::MatrixXf& velocities);

Eigen::MatrixXf generateVerticalFibreFluid(Eigen::MatrixXf& positions, Eigen::MatrixXf& velocities);

Eigen::MatrixXf generateNewParticlesDroplet(Eigen::MatrixXf& positions, Eigen::MatrixXf& velocities, int n_new_particles);

Eigen::MatrixXf generateWettedFibreHorizontal(Eigen::MatrixXf& positions, Eigen::MatrixXf& velocities);

#endif