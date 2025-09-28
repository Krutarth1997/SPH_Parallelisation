#ifndef CALCULATEPHYSICS_H
#define CALCULATEPHYSICS_H

#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

__global__ void calculateDensities(int* int_neighbors, float* int_densities, float int_smoothing_length,
                        float* int_distances, float int_particle_mass, int* d_n_particles, int* total_neighbors);

__global__ void updateVelocities(float* velocities, float* forces, float timestep, int n_particles);

__global__ void updatePositions(float* positions, float* velocities, float timestep, int n_particles);

#endif
