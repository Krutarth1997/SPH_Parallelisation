#ifndef KERNEL_H
#define KERNEL_H

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

__device__ double CubicSplineKernel(float dist, float smooth);
__device__ double CubicSplineDifKernel(float dist, float smooth);
__device__ double CubicSplineDif2Kernel(float dist, float smooth);
__device__ double SpikyKernel(float dist, float smooth);
__device__ double SpikyDifKernel(float dist, float smooth);
__device__ double SpikyDif2Kernel(float dist, float smooth);
__device__ double Poly6Kernel(float dist, float smooth);
__device__ double Poly6DifKernel(float dist, float smooth);
__device__ double Poly6Dif2Kernel(float dist, float smooth);
__device__ double CohesionKernel(float dist, float smooth);
__device__ double AdhesionKernel(float dist, float smooth);
__device__ double DensityKernel(float dist, float smooth);

__global__ void DensityKernelParticle(int* neighbor_ids, float* densities, float SMOOTHING_LENGTH, float* distances, float PARTICLE_MASS, int n_particles, int* total_neighbors);
__global__ void DensityCorrectedParticle(int* neighbor_ids, float* densities,float* densities_corr, float SMOOTHING_LENGTH, float* distances, float PARTICLE_MASS, int n_particles, int* total_neighbors);
__global__ void CheckBoundaryParticle(int* neighbor_idsFluidFibre,float* randpart, float* densities, float SMOOTHING_LENGTH_FIBRE, float* distancesFluidFibre, float PARTICLE_MASS_FIBRE, int n_particles,int* d_n_particles, int* total_neighbors);
__global__ void ColorFieldKernel(int* neighbor_ids, float* gradColorField, float* position, float* densities, float SMOOTHING_LENGTH, float* distances, float PARTICLE_MASS, int n_particles, int* total_neighbors);

__device__ double pressureKernel(float dist, float smooth);
__device__ double viscosityKernel(float dist, float smooth);
__device__ double densityKernel(float dist, float smooth);

#endif 