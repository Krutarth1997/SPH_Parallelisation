#ifndef INITIALIZEPARTICLES_H
#define INITIALIZEPARTICLES_H

#include <iostream>

std::pair<std::vector<float>, std::vector<float>> generateBoxFlow(std::vector<float>& positions, std::vector<float>& velocities);

std::pair<std::vector<float>, std::vector<float>> generateSphericalParticles(std::vector<float>& positions, std::vector<float>& velocities, int n_particles);

std::pair<std::vector<float>, std::vector<float>> generateSemiSphericalParticles(std::vector<float>& positions, std::vector<float>& velocities, int n_particles);

std::pair<std::vector<float>, std::vector<float>> generateVerticalFibreFluid(std::vector<float>& positions, std::vector<float>& velocities);

std::pair<std::vector<float>, std::vector<float>> generateNewParticlesDroplet(std::vector<float>& positions, std::vector<float>& velocities, int n_particles, int n_new_particles);

std::pair<std::vector<float>, std::vector<float>> generateWettedFibreHorizontal(std::vector<float>& positions, std::vector<float>& velocities, int n_particles);

#endif