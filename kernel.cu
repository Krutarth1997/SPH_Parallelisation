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
#include <math_constants.h>
#include "kernel.h"

__device__ double CubicSplineKernel(float dist, float smooth)
{
    float norm = 8.0f / (3.14159265f * pow(smooth, 3));
    float value;

    if (dist <= 0.5 * smooth) {
        value = 6 * (pow((dist / smooth), 3) - pow((dist / smooth), 2)) + 1;
    } else if (dist > 0.5 * smooth && dist <= smooth) {  
        value = 2 * pow((1 - (dist / smooth)), 3);
    } else {
        value = 0;
    }

    return value * norm;
}

__device__ double CubicSplineDifKernel(float dist, float smooth)
{
    if (dist <= 0.5 * smooth)
    {
		float norm = 8 / (3.14159265f * pow(smooth, 3));
		float value = ((3*pow(dist,2))/pow(smooth,3)) - ((2*dist)/pow(smooth,2));
        return 6 * value * norm;
    }
    else if (dist > 0.5 * smooth && dist <= smooth)
    {
        float norm = 8 / (3.14159265f * pow(smooth, 3));
        float value = pow((1- (dist/smooth)),2);
        return (6 / smooth) * value * norm;
    }
    return 0;
}

__device__ double CubicSplineDif2Kernel(float dist, float smooth)
{
    float norm = 8.0f / (CUDART_PI_F * pow(smooth, 3));
    float q = dist / smooth;
    float value;

    if (q <= 0.5f)
    {
        value = (6.0f * dist / pow(smooth, 3)) - (2.0f / pow(smooth, 2));
        return 6.0f * value * norm;
    }
    else if (q >= 0.5f && q <= 1.0f)
    {
        value = 1.0f - q;
        return (12.0f / pow(smooth, 2)) * value * norm;
    }
    return 0.0f;
}

__device__ double SpikyKernel(float dist, float smooth)
{
    if (dist < smooth)
    {
		float norm = 15 / (3.14159265f * pow(smooth, 6));
		float value = pow((smooth - dist),3);
        return (value * norm);
    }
    return 0;
}

__device__ double SpikyDifKernel(float dist, float smooth)
{
    if (dist < smooth)
    {
        float norm = 15.0f / (CUDART_PI_F * pow(smooth, 6));
        float value = pow(smooth - dist, 2);
        return -3.0f * value * norm;
    }
    return 0.0f;
}

__device__ double SpikyDif2Kernel(float dist, float smooth)
{
    if (dist < smooth)
    {
        float norm = 15.0f / (CUDART_PI_F * pow(smooth, 6));
        float value = smooth - dist;
        return 6.0f * value * norm;
    }
    return 0.0f;
}

__device__ double Poly6Kernel(float dist, float smooth)
{
    if (dist <= smooth)
    {
        float norm = 315.0f / (64.0f * CUDART_PI_F * pow(smooth, 9));
        float value = pow(pow(smooth, 2) - pow(dist, 2), 3);
        return value * norm;
    }
    return 0.0f;
}

__device__ double Poly6DifKernel(float dist, float smooth)
{
    if (dist <= smooth)
    {
        float norm = 315.0f / (64.0f * CUDART_PI_F * pow(smooth, 9));
        float value = pow(pow(smooth, 2) - pow(dist, 2), 2);
        return -6.0f * value * norm * dist;
    }
    return 0.0f;
}

__device__ double Poly6Dif2Kernel(float dist, float smooth)
{
    if (dist <= smooth)
    {
        float norm = 315.0f / (64.0f * CUDART_PI_F * pow(smooth, 9));
        float value = pow(smooth, 2) - pow(dist, 2);
        return 24.0f * value * norm * pow(dist, 2);
    }
    return 0.0f;
}

__device__ double CohesionKernel(float dist, float smooth)
{
    float norm = 32.0f / (3.14159265f * pow(smooth, 6));
    float value;

    if (dist <= 0.5 * smooth) {
        value = 2 * (pow((smooth - dist),3) * pow(dist,3)) - (pow(smooth,6) / 64);
    } else if (dist > 0.5 * smooth && dist <= smooth) {  
        value = pow((smooth - dist),3) * pow(dist,3);
    } else {
        value = 0;
    }

    return (value * norm)/pow(smooth, 3);
}

// __device__ float CohesionKernel(float dist, float smooth)
// {
//     if (dist <= 0.5 * smooth)
// 	{
// 		double norm = 32 / (3.14159265f * powf(smooth, 9));
// 		double value = 2 * powf((smooth-dist),3) * powf(dist,3) - (powf(smooth,6)/64);
//         return float(norm * value);
// 	}
//     else if (dist > 0.5 * smooth && dist <= smooth)
//     {
//         double norm = 32 / (3.14159265f * powf(smooth, 9));
//         double value = powf((smooth-dist),3) * powf(dist,3);
//         return float(value * norm);
//     }
// 	return 0;
// }

__device__ double AdhesionKernel(float dist, float smooth)
{
    if (dist <= smooth)
    {
        float norm = 0.007f / pow(smooth, 3.25f);
        float value = pow(-((4.0f * pow(dist, 2)) / smooth) + 6.0f * dist - 2.0f * smooth, 0.25f);
        return norm * value;
    }
    return 0.0f;
}

__device__ double pressureKernel(float dist, float smooth)
{
	return SpikyKernel(dist, smooth);
}

__device__ double viscosityKernel(float dist, float smooth)
{
	return CubicSplineKernel(dist, smooth);
}

__device__ double densityKernel(float dist, float smooth)
{
	return Poly6Kernel(dist, smooth);
}

__device__ double DensityKernel(float dist, float smooth)
{
	return CubicSplineKernel(dist, smooth);
}

__global__ void DensityKernelParticle(int* neighbor_ids, float* densities, float SMOOTHING_LENGTH,
                        float* distances, float PARTICLE_MASS, int n_particles, int* total_neighbors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_particles) {
		float density = 0.0f;
		for (int j = 0; j < total_neighbors[i]; ++j)
		{
			int neighbor_id = neighbor_ids[i * n_particles + j];
            float dist = distances[i * n_particles + j];
            if (dist <= 0.0f) {
                printf("particle %d with neighbor %d has distance 0\n",i , neighbor_id, dist);
            }
            if (!isfinite(dist)) {
                printf("ERROR: Infinite distance for particle %d and neighbor %d\n", i, j);
            }
			if (dist > 0.0f) {
				float kernel_value = CubicSplineKernel(dist, SMOOTHING_LENGTH);
				density += kernel_value * PARTICLE_MASS;
			}
		}
		densities[i] = density;
	}
}

__global__ void DensityCorrectedParticle(int* neighbor_ids, float* densities,float* densities_corr, float SMOOTHING_LENGTH,
                        float* distances, float PARTICLE_MASS, int n_particles, int* total_neighbors)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_particles) {
		float density_corr = 0.0f;
		for (int j = 0; j < total_neighbors[i]; ++j)
		{
			int neighbor_id = neighbor_ids[i * n_particles + j];
			float density = densities[i];
            float dist = distances[i * n_particles + j];

            if (density <= 0.0f) {
                printf("particle %d has density 0\n",i);
            }
			if ((dist > 0.0f) &&  (density > 0.0f)) {
				float kernel_value = CubicSplineKernel(dist, SMOOTHING_LENGTH);
				density_corr += (PARTICLE_MASS * kernel_value) / ((PARTICLE_MASS/density) * kernel_value);
			}
		}
		densities_corr[i] = density_corr;
	}
}

__global__ void CheckBoundaryParticle(int* neighbor_idsFluidFibre,float* randpart, float* densities, float SMOOTHING_LENGTH_FIBRE,
                        float* distancesFluidFibre, float PARTICLE_MASS_FIBRE, int n_particles, int* d_n_particles, int* total_neighbors)
{
    int n_particles2 = *d_n_particles; //dereference pointer
	int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_particles) {
		float rand = 0.0f;
		for (int j = 0; j < total_neighbors[i]; ++j)
		{
			int neighbor_idFluidFibre = neighbor_idsFluidFibre[i * n_particles2 + j];

            float dist = distancesFluidFibre[i * n_particles2 + j];
			if (dist > 0.0f) {
				float kernel_value = DensityKernel(dist, SMOOTHING_LENGTH_FIBRE);
				rand += kernel_value * PARTICLE_MASS_FIBRE;
			}
		}
		randpart[i] = rand + densities[i];
	}
}

__global__ void ColorFieldKernel(int* neighbor_ids, float* gradColorField, float* position, float* densities, float SMOOTHING_LENGTH,
                        float* distances, float PARTICLE_MASS, int n_particles, int* total_neighbors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_particles) {
		float grad_x = 0.0f;
		float grad_y = 0.0f;
		float grad_z = 0.0f;
		for (int j_in_list = 0; j_in_list < total_neighbors[i]; ++j_in_list)
		{
			int j = neighbor_ids[i * n_particles + j_in_list]; //neighbor of particle i
			float density = densities[j]; //density of the neighbor
			float positionj_x = position[j * 3];  // x-coordinate of the neighbor
            float positionj_y = position[j * 3 + 1];  // y-coordinate of the neighbor
            float positionj_z = position[j * 3 + 2];  // z-coordinate of the neighbor
            float positioni_x = position[i * 3];  // x-coordinate of particle i
            float positioni_y = position[i * 3 + 1];  // y-coordinate of particle i
            float positioni_z = position[i * 3 + 2];  // z-coordinate of particle i

            float dist = distances[i * n_particles + j_in_list];

            if (density <= 0.0f) {
                printf("ERROR: Particle %d has zero density!\n",i);
                return;
            }

            if (dist <= 0.0f) {
                printf("ERROR: Particle %d and neighbor %d have zero/negative distance!\n", i, j);
                continue; // Skip this neighbor
            }

			float kernel_value = CubicSplineDifKernel(dist, SMOOTHING_LENGTH);
			if ((dist > 0.0f) &&  (density > 0.0f)) {
			float factor = kernel_value * SMOOTHING_LENGTH * (PARTICLE_MASS / density);
			grad_x += factor * ((positionj_x - positioni_x) / dist);
			grad_y += factor * ((positionj_y - positioni_y) / dist);
			grad_z += factor * ((positionj_z - positioni_z) / dist);
			}
		}
		gradColorField[i * 3]     = grad_x;
		gradColorField[i * 3 + 1] = grad_y;
		gradColorField[i * 3 + 2] = grad_z;
	}

	//Delete NaN-Values
	// if (i < n_particles) {
	// 	for (int d = 0; d < 3; ++d) {
	// 		if (isnan(gradColorField[i * 3 + d])) {
	// 			gradColorField[i * 3 + d] = 0.0f;
	// 		}
	// 	}
	// }
	if (i < n_particles) {
		gradColorField[i * 3]     = isnan(gradColorField[i * 3]) ? 0.0f : gradColorField[i * 3];
		gradColorField[i * 3 + 1] = isnan(gradColorField[i * 3 + 1]) ? 0.0f : gradColorField[i * 3 + 1];
		gradColorField[i * 3 + 2] = isnan(gradColorField[i * 3 + 2]) ? 0.0f : gradColorField[i * 3 + 2];
	}
}

// __global__ void ColorFieldKernelCuda(const int* d_neighbor_ids, const float* d_positions, const float* d_distances,
// 					const float* d_densities, float* d_gradColorField, int n_particles, int n_neighbors,
// 					float SMOOTHING_LENGTH, float PARTICLE_MASS) 
// {
//     // Shared memory for particle positions and densities
//     extern __shared__ float shared_data[];
//     float* shared_positions = shared_data;
//     float* shared_densities = &shared_data[blockDim.x * 3];

//     int global_id = blockIdx.x * blockDim.x + threadIdx.x;
//     int local_id = threadIdx.x;

//     if (global_id < n_particles) {
//         // Load data into shared memory
//         for (int d = 0; d < 3; ++d) {
//             shared_positions[local_id * 3 + d] = d_positions[global_id * 3 + d];
//         }
//         shared_densities[local_id] = d_densities[global_id];
//         __syncthreads();

//         // Initialize local gradient
//         float grad[3] = {0.0f, 0.0f, 0.0f};

//         // Iterate over neighbors
//         for (int j_in_list = 0; j_in_list < n_neighbors; ++j_in_list) {
//             int neighbor_id = d_neighbor_ids[global_id * n_neighbors + j_in_list];

//             if (neighbor_id >= 0) { // Valid neighbor
//                 float dist = d_distances[global_id * n_neighbors + j_in_list];
//                 float kernel_value = CubicSplineDifKernel(dist, SMOOTHING_LENGTH); // Implement this kernel function separately

//                 // Compute the gradient contribution
//                 float density = shared_densities[local_id];
//                 float factor = kernel_value * SMOOTHING_LENGTH * (PARTICLE_MASS / density);

//                 for (int d = 0; d < 3; ++d) {
//                     float diff = d_positions[neighbor_id * 3 + d] - shared_positions[local_id * 3 + d];
//                     grad[d] += factor * (diff / dist);
//                 }
//             }
//         }

// 		if (global_id < n_particles) {
//         for (int d = 0; d < 3; ++d) {
//             if (isnan(d_gradColorField[global_id * 3 + d])) {
//                 d_gradColorField[global_id * 3 + d] = 0.0f;
//             }
//         }
//     }

//         // Write the gradient to global memory
//         for (int d = 0; d < 3; ++d) {
//             d_gradColorField[global_id * 3 + d] = grad[d];
//         }
//     }
// }

// __global__ void ColorFieldKernelCuda(const int* d_neighbor_ids, const float* d_positions, const float* d_distances,
//                                      const float* d_densities, float* d_gradColorField, int n_particles, int n_neighbors,
//                                      float SMOOTHING_LENGTH, float PARTICLE_MASS) 
// {
//     int global_id = blockIdx.x * blockDim.x + threadIdx.x;

//     if (global_id < n_particles) {
//         // Initialize local gradient
//         float grad[3] = {0.0f, 0.0f, 0.0f};

//         // Iterate over neighbors
//         for (int j_in_list = 0; j_in_list < n_neighbors; ++j_in_list) {
//             int neighbor_id = d_neighbor_ids[global_id * n_neighbors + j_in_list];

//             if (neighbor_id >= 0) { // Valid neighbor
//                 float dist = d_distances[global_id * n_neighbors + j_in_list];
//                 if (dist > 1e-6) { // Avoid division by zero
//                     float kernel_value = CubicSplineDifKernel(dist, SMOOTHING_LENGTH); // Implement this kernel function separately

//                     // Compute the gradient contribution
//                     float factor = kernel_value * SMOOTHING_LENGTH * (PARTICLE_MASS / d_densities[neighbor_id]);

//                     for (int d = 0; d < 3; ++d) {
//                         float diff = d_positions[neighbor_id * 3 + d] - d_positions[global_id * 3 + d];
//                         grad[d] += factor * (diff / dist);
//                     }
//                 }
//             }
//         }

//         // Write the gradient to global memory, handling NaNs
//         for (int d = 0; d < 3; ++d) {
//             d_gradColorField[global_id * 3 + d] = isnan(grad[d]) ? 0.0f : grad[d];
//         }
//     }
// }


