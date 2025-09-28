#ifndef MATRIXOPERATION_H
#define MATRIXOPERATION_H

#include <iostream>
#include <vector>

__global__ void calculateNeighborhoodKernelff(float* positions1, float* positions2, int* neighbor_ids, float* distances, int n_particles1, int n_particles2, float smoothing_length, int* total_neighbors);

__global__ void calculateNeighborhoodKernelfifi(float* positions1, float* positions2, int* neighbor_ids, float* distances, int* d_n_particles1, int* d_n_particles2, float smoothing_length, int* total_neighbors);

__global__ void calculateNeighborhoodKernelffi(float* positions1, float* positions2, int* neighbor_ids, float* distances, int n_particles1, int* d_n_particles2, float smoothing_length, int* total_neighbors);

__global__ void RemoveOwnElementKernel(int* neighbor_ids, float* distances, int n_particles, int* total_neighbor);

#endif 