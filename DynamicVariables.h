#ifndef DYNAMICVARIABLES_H
#define DYNAMICVARIABLES_H

#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

__global__ void ComputePressures(float* d_pressures, float* d_densities, float isotropic_exponent, float base_density, int n_particles);

__global__ void ForcesFluid( int* neighbor_ids, float* distances, float* positions, float* pressures, float* densities, float* velocities, float* forces, float* gradColorField, 
                                    float SMOOTHING_LENGTH, float PARTICLE_MASS, float viscosity_artificial, float DYNAMIC_VISCOSITY, float BASE_DENSITY, float artificial_surface_tension_cohesion,
                                    float artificial_surface_tension_curvature, float SURFACE_TENSION, float density_gas, float cw,
                                    float* CONSTANT_FORCE, float* velocity_gas, int n_particles, int* total_neighbors, int count);

__global__ void ForcesGeometry( int* neighbor_idsFluidFibre, float* distancesFluidFibre, float* positions, float* positionsFibre, 
                                        float* velocities, float* velocitiesFibre, float* densities, float* forces, float* adhesion, float SMOOTHING_LENGTH, 
                                        float SMOOTHING_LENGTH_FIBRE, float PARTICLE_MASS,float PARTICLE_MASS_FIBRE, float viscosity_artificial,
                                        float DYNAMIC_VISCOSITY_FIBRE, float BASE_DENSITY, float artificial_surface_tension_adhesion, float LJP_DISTANCE, 
                                        int LJP_P1, int LJP_P2, float LJP_COEF, int* total_neighbors, int n_particles, int* d_n_particlesFibre, int count);

__global__ void deleteunneccessaryparticles( float* velocities, float* forces, int n_particles);

#endif
