#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include "matrixOperation.h"

// Function Overloading for fluid-fluid interaction
__global__ void calculateNeighborhoodKernelff(float* positions1, float* positions2, int* neighbor_ids, float* distances, int n_particles1, int n_particles2, float smoothing_length, int* total_neighbors) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_particles1) {
        int neighbors_count = 0;

        for (int j = 0; j < n_particles2; ++j) {

            float dx = positions1[i * 3] - positions2[j * 3];
            float dy = positions1[i * 3 + 1] - positions2[j * 3 + 1];
            float dz = positions1[i * 3 + 2] - positions2[j * 3 + 2];
            float dist = sqrtf(dx * dx + dy * dy + dz * dz);

            if ((dist < smoothing_length) && (dist > 0.0f)) {
                neighbor_ids[i * n_particles2 + neighbors_count] = j;
                distances[i * n_particles2 + neighbors_count] = dist;
                neighbors_count++;
            }
        }
        // Storing total neighbor count
        total_neighbors[i]= neighbors_count;
    }
}

// Function Overloading for fibre-fibre interaction
__global__ void calculateNeighborhoodKernelfifi(float* positions1, float* positions2, int* neighbor_ids, float* distances, int* d_n_particles1, int* d_n_particles2, float smoothing_length, int* total_neighbors) {
    int n_particles1 = *d_n_particles1; // Dereference device pointer
    int n_particles2 = *d_n_particles2; // Dereference device pointer

    //printf("n_particles1: %d, n_particles2: %d\n", n_particles1, n_particles2); //720 , 720

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_particles1) {
        int neighbors_count = 0;

        for (int j = 0; j < n_particles2; ++j) {

            float dx = positions1[i * 3] - positions2[j * 3];
            float dy = positions1[i * 3 + 1] - positions2[j * 3 + 1];
            float dz = positions1[i * 3 + 2] - positions2[j * 3 + 2];
            float dist = sqrtf(dx * dx + dy * dy + dz * dz);

            if ((dist < smoothing_length) && (dist > 0.0f)) {
                neighbor_ids[i * n_particles2 + neighbors_count] = j;
                distances[i * n_particles2 + neighbors_count] = dist;
                neighbors_count++;
            }
        }
        // Storing total neighbor count
        total_neighbors[i]= neighbors_count;
    }
}

// Function Overloading for fluid-fibre interaction
__global__ void calculateNeighborhoodKernelffi(float* positions1, float* positions2, int* neighbor_ids, float* distances, int n_particles1, int* d_n_particles2, float smoothing_length, int* total_neighbors) {
    int n_particles2 = *d_n_particles2;// Dereference device pointer
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_particles1) {
        int neighbors_count = 0;

        for (int j = 0; j < n_particles2; ++j) {

            float dx = positions1[i * 3] - positions2[j * 3];
            float dy = positions1[i * 3 + 1] - positions2[j * 3 + 1];
            float dz = positions1[i * 3 + 2] - positions2[j * 3 + 2];
            float dist = sqrtf(dx * dx + dy * dy + dz * dz);

            if ((dist < smoothing_length) && (dist > 0.0f)) {
                neighbor_ids[i * n_particles2 + neighbors_count] = j;
                distances[i * n_particles2 + neighbors_count] = dist;
                neighbors_count++;
            }
        }
        // Storing total neighbor count
        total_neighbors[i]= neighbors_count;
    }
}

// __global__ void calculateNeighborhoodKernel(
//     float* positions1, float* positions2, int* neighbor_ids, float* distances,
//     void* n_particles1, void* n_particles2, bool isPointer,
//     float smoothing_length, int* total_neighbors
// ) {
//     // Determine the number of particles
//     int n1 = isPointer ? *(int*)n_particles1 : *(int*)(&n_particles1);
//     int n2 = isPointer ? *(int*)n_particles2 : *(int*)(&n_particles2);

//     int i = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i < n1) {
//         int neighbors_count = 0;

//         for (int j = 0; j < n2; ++j) {
//             float dx = positions1[i * 3] - positions2[j * 3];
//             float dy = positions1[i * 3 + 1] - positions2[j * 3 + 1];
//             float dz = positions1[i * 3 + 2] - positions2[j * 3 + 2];
//             float dist = sqrtf(dx * dx + dy * dy + dz * dz);

//             if ((dist <= smoothing_length) && (dist > 0.0f)) {
//                 neighbor_ids[i * n2 + neighbors_count] = j;
//                 distances[i * n2 + neighbors_count] = dist;
//                 neighbors_count++;
//             }
//         }
//         total_neighbors[i] = neighbors_count;
//     }
// }

__global__ void RemoveOwnElementKernel(int* neighbor_ids, float* distances, int n_particles, int* total_neighbor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_particles) {
        for (int j = 0; j < total_neighbor[i]; ++j) {
            int index = i * n_particles + j;
            if (neighbor_ids[index] == i) {
                neighbor_ids[index] = -1;
                distances[index] = 0.0f;
            }
        }
    }
}