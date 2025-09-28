#ifndef OUTPUT_H
#define OUTPUT_H

#include <iostream>

void saveParticleAsVTK(const std::string& path, float timestep, int n_particles, float* positions, float* velocities,
float* density, float* randpart, float* pressures, float* density_corr, float* gradColorField, float* forces, int* wandpart, float* adhesion);

void saveFibreAsVTK(const std::string& path, float timestep, int n_particles, float* positions);

void progressBar(float iter, float timestep);

#endif