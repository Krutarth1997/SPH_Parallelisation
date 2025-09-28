#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include "calculatePhysics.h"
#include "kernel.h"

__global__ void calculateDensities(int* int_neighbors, float* int_densities, float int_smoothing_length,
                        float* int_distances, float int_particle_mass, int* d_n_particles, int* total_neighbors)
{
    int n_particles = *d_n_particles; // Dereference device pointer
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_particles) {

        float densities = 0.0f;

        for (int j = 0; j < total_neighbors[i]; ++j) {
            int neighbor_id = int_neighbors[i * n_particles + j];
            float dist = int_distances[i * n_particles + j];
            if (dist > 0.0f) {
            densities += densityKernel(dist, int_smoothing_length) * int_particle_mass;
            }
        }
        int_densities[i] = densities;
    }
}

__global__ void updateVelocities(float* velocities, float* forces, float timestep, int n_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        for (int j = 0; j < 3; ++j) {
            velocities[i * 3 + j] += timestep * forces[i * 3 + j];
        }
    }
}

__global__ void updatePositions(float* positions, float* velocities, float timestep, int n_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        for (int j = 0; j < 3; ++j) {
            positions[i * 3 + j] += timestep * velocities[i * 3 + j];
        }
    }
}

