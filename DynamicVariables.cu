#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include "kernel.h"

__global__ void ComputePressures(float* d_pressures, float* d_densities, float isotropic_exponent, float base_density, int n_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        d_pressures[i] = isotropic_exponent * (d_densities[i] - base_density);
    }
}

// #include <cblas.h> // OpenBLAS
// #include <cmath>
// void ForcesFluid(
//     int n_particles, 
//     const std::vector<std::vector<int>>& neighbor_ids, 
//     const std::vector<std::vector<float>>& distances, 
//     const std::vector<std::vector<float>>& positions, 
//     const std::vector<float>& pressures, 
//     const std::vector<float>& densities, 
//     const std::vector<std::vector<float>>& velocities, 
//     std::vector<std::vector<float>>& forces, 
//     float SMOOTHING_LENGTH, float PARTICLE_MASS, float viscosity_artificial, 
//     float DYNAMIC_VISCOSITY, float BASE_DENSITY, float artificial_surface_tension_cohesion, 
//     float artificial_surface_tension_curvature, float SURFACE_TENSION, 
//     float density_gas, float cw, 
//     const std::vector<float>& CONSTANT_FORCE, 
//     const std::vector<float>& velocity_gas, 
//     std::vector<std::vector<float>>& gradColorField, 
//     int count) 
// {
//     for (int i = 0; i < n_particles; ++i) {
//         for (size_t j_in_list = 0; j_in_list < neighbor_ids[i].size(); ++j_in_list) {
//             int j = neighbor_ids[i][j_in_list];

//             // Pressure force
//             float kernel_value = pressureKernel(distances[i][j_in_list], SMOOTHING_LENGTH);
//             std::vector<float> pressure_force(3, 0.0f);
//             for (int d = 0; d < 3; ++d) {
//                 pressure_force[d] = kernel_value * ((positions[j][d] - positions[i][d]) / distances[i][j_in_list]) * 
//                                     ((pressures[i] + pressures[j]) / 2 * densities[j]);
//                 forces[i][d] -= pressure_force[d];
//             }

//             // Viscous force
//             kernel_value = viscosityKernel(distances[i][j_in_list], SMOOTHING_LENGTH);
//             std::vector<float> viscous_force(3, 0.0f);
//             for (int d = 0; d < 3; ++d) {
//                 viscous_force[d] = viscosity_artificial * kernel_value * DYNAMIC_VISCOSITY * PARTICLE_MASS * 
//                                    (velocities[j][d] - velocities[i][d]) / (densities[j] * densities[j]);
//                 forces[i][d] += viscous_force[d];
//             }

//             // Surface tension - Cohesion
//             float freeSurfaceIndicator = ((2 * BASE_DENSITY) / (densities[i] + densities[j]));
//             kernel_value = CohesionKernel(distances[i][j_in_list], SMOOTHING_LENGTH);
//             std::vector<float> surface_tension_cohesion(3, 0.0f);
//             for (int d = 0; d < 3; ++d) {
//                 surface_tension_cohesion[d] = -artificial_surface_tension_cohesion * 2 * PARTICLE_MASS * 
//                                               ((positions[i][d] - positions[j][d]) / distances[i][j_in_list]) * kernel_value;
//                 forces[i][d] += surface_tension_cohesion[d];
//             }

//             // Surface tension - Curvature
//             std::vector<float> surface_tension_curvature(3, 0.0f);
//             for (int d = 0; d < 3; ++d) {
//                 surface_tension_curvature[d] = -artificial_surface_tension_curvature * SURFACE_TENSION * PARTICLE_MASS * 
//                                                freeSurfaceIndicator * (gradColorField[j][d] - gradColorField[i][d]);
//                 forces[i][d] += surface_tension_curvature[d];
//             }
//         }

//         // Gravitational force (Using OpenBLAS saxpy for vector addition)
//         cblas_saxpy(3, 1.0f / PARTICLE_MASS, CONSTANT_FORCE.data(), 1, forces[i].data(), 1);

//         // Drag force
//         float radius = std::pow((6 * PARTICLE_MASS) / (M_PI * densities[i]), 1.0f / 3.0f);
//         float drag_coeff = cw * 0.5f * density_gas * (M_PI / 4) * (radius * radius);
//         std::vector<float> velocity_diff(3, 0.0f);
//         std::vector<float> drag_force(3, 0.0f);

//         for (int d = 0; d < 3; ++d) {
//             velocity_diff[d] = velocity_gas[d] - velocities[i][d];
//         }

//         float velocity_norm = cblas_snrm2(3, velocity_diff.data(), 1);
//         for (int d = 0; d < 3; ++d) {
//             drag_force[d] = drag_coeff * velocity_diff[d] * velocity_norm / PARTICLE_MASS;
//             forces[i][d] += drag_force[d];
//         }
//     }
// }

__global__ void ForcesFluid( int* neighbor_ids, float* distances, float* positions, float* pressures, float* densities, float* velocities, float* forces, float* gradColorField, 
                                    float SMOOTHING_LENGTH, float PARTICLE_MASS, float viscosity_artificial, float DYNAMIC_VISCOSITY, float BASE_DENSITY, float artificial_surface_tension_cohesion,
                                    float artificial_surface_tension_curvature, float SURFACE_TENSION, float density_gas, float cw,
                                    float* CONSTANT_FORCE, float* velocity_gas, int n_particles, int* total_neighbors, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_particles) {
        float force_x = 0.0f, force_y = 0.0f, force_z = 0.0f;

        for (int j_in_list = 0; j_in_list < total_neighbors[i]; ++j_in_list) {
            int j = neighbor_ids[i * n_particles + j_in_list]; //neighbor j of particle i
            float dist = distances[i * n_particles + j_in_list]; //distance between particle i and j

            // Load particle data
            float px_i = positions[i * 3], py_i = positions[i * 3 + 1], pz_i = positions[i * 3 + 2];
            float px_j = positions[j * 3], py_j = positions[j * 3 + 1], pz_j = positions[j * 3 + 2];
            float vx_i = velocities[i * 3], vy_i = velocities[i * 3 + 1], vz_i = velocities[i * 3 + 2];
            float vx_j = velocities[j * 3], vy_j = velocities[j * 3 + 1], vz_j = velocities[j * 3 + 2];

            if (densities[i] <= 0.0f) {
                printf("ERROR: Particle %d has zero density!\n", i);
                return;
            }

            if (densities[j] <= 0.0f) {
                printf("ERROR: Neighbour %d of Particle %d has zero density!\n", j, i);
                return;
            }

            if (dist <= 0.0f) {
                printf("ERROR: Particle %d and neighbor %d have very less distance!\n", i, j);
                continue; // Skip this neighbor
            }

            if ((dist > 0.0f) && (densities[i] > 0.0f) && (densities[j] > 0.0f)) {

                // Pressure force
                double kernel_value = pressureKernel(dist, SMOOTHING_LENGTH);
                //need to check j should be neighbor id 
                float pressure_force_x = kernel_value * ((px_j - px_i) / dist) * ((pressures[i] + pressures[j]) / (2.0f * densities[j]));
                // if ((i ==100) && (j < 5) && (count <10)) {
                //     printf("Pressure Parameters 1 x: kernel %e, posi %f, posj %f, dist %f\n", kernel_value, px_i, px_j, dist);
                //     printf("Pressure Parameters 2 x: pressi %e, pressj %e, densj %f\n", pressures[i], pressures[j], densities[j]);
                //     printf("Pressure Parameters 3 x: pos_diff/dis %f, Pressure / density %e\n", ((px_j - px_i) / dist)), ((pressures[i] + pressures[j]) / (2.0f * densities[j]));
                // }
                float pressure_force_y = kernel_value * (py_j - py_i) / dist * ((pressures[i] + pressures[j]) / (2.0f * densities[j]));
                float pressure_force_z = kernel_value * (pz_j - pz_i) / dist * ((pressures[i] + pressures[j]) / (2.0f * densities[j]));
                force_x -= pressure_force_x;
                force_y -= pressure_force_y;
                force_z -= pressure_force_z;

                // if ((i ==100) && (j < 5) && (count <5)) {
                //     printf("Pressure force x: %e, y: %e, z: %e\n", pressure_force_x, pressure_force_y, pressure_force_z);
                // }

                // if ((i ==10) && (j < 5) && (count <10)) {
                //     printf("i %d , j %d and count %d\n", i, j, count);
                //     printf("After Pressure force x: %e, y: %e, z: %e\n", force_x, force_y, force_z);
                //     printf("*****************\n");
                // }

                // Viscous force
                kernel_value = viscosityKernel(dist, SMOOTHING_LENGTH);
                float viscous_force_x = viscosity_artificial * kernel_value * DYNAMIC_VISCOSITY * PARTICLE_MASS *
                                        (vx_j - vx_i) / powf(densities[j],2);
                // if ((i ==0) && (j < 5) && (count <3)) {
                //     printf("Viscous Parameters 1 x: cosnt %f, mass %e, posi %f, posj %f, dist %f\n", viscosity_artificial, PARTICLE_MASS, px_i, px_j, dist);
                //     printf("Viscous Parameters 2 x: vel_diff/dens %f, kernel %e\n", ((vx_j - vx_i) / powf(densities[j],2)), kernel_value);
                // }
                float viscous_force_y = viscosity_artificial * kernel_value * DYNAMIC_VISCOSITY * PARTICLE_MASS *
                                        (vy_j - vy_i) / powf(densities[j],2);
                float viscous_force_z = viscosity_artificial * kernel_value * DYNAMIC_VISCOSITY * PARTICLE_MASS *
                                        (vz_j - vz_i) / powf(densities[j],2);
                force_x += viscous_force_x;
                force_y += viscous_force_y;
                force_z += viscous_force_z;

                // if ((i ==0) && (j < 5) && (count <3)) {
                //     printf("Viscous force x: %e, y: %e, z: %e\n", viscous_force_x, viscous_force_y, viscous_force_z);
                // }

                // if ((i ==10) && (j < 5) && (count <5)) {
                //     printf("i %d , j %d and count %d\n", i, j, count);
                //     printf("After Viscous force x: %e, y: %e, z: %e\n", force_x, force_y, force_z);
                //     printf("*****************\n");
                // }

                // Surface tension (Cohesion[Akinci])
                float freeSurfaceIndicator = ((2 * BASE_DENSITY) / (densities[i] + densities[j]));
                kernel_value = CohesionKernel(dist, SMOOTHING_LENGTH);
                float surface_tension_cohesion_x = -artificial_surface_tension_cohesion * 2 * PARTICLE_MASS *
                                                ((px_i - px_j) / dist) * kernel_value;
                // if ((i ==199) && (j_in_list < 5) && (count <10)) {
                //     printf("Cohesion Parameters 1 x: cosnt %f, mass %e, posi %f, posj %f, dist %f\n", artificial_surface_tension_cohesion, PARTICLE_MASS, py_i, py_j, dist);
                //     printf("Cohesion Parameters 2 x: pos_diff/dis %f, kernel %e\n", ((py_i - py_j) / dist), kernel_value);
                // }
                float surface_tension_cohesion_y = -artificial_surface_tension_cohesion * 2 * PARTICLE_MASS *
                                                ((py_i - py_j) / dist) * kernel_value;
                float surface_tension_cohesion_z = -artificial_surface_tension_cohesion * 2 * PARTICLE_MASS *
                                                ((pz_i - pz_j) / dist) * kernel_value;
                force_x += surface_tension_cohesion_x;
                force_y += surface_tension_cohesion_y;
                force_z += surface_tension_cohesion_z;

                // if ((i ==199) && (j_in_list < 5) && (count <10)) {
                //     printf("Cohesion force x: %e, y: %e, z: %e\n", surface_tension_cohesion_x, surface_tension_cohesion_y, surface_tension_cohesion_z);
                // }

                // if ((i ==100) && (j < 5) && (count <10)) {
                //     printf("i %d , j %d and count %d\n", i, j, count);
                //     printf("After Cohesion force x: %e, y: %e, z: %e\n", force_x, force_y, force_z);
                //     printf("*****************\n");
                // }

                // Surface tension (Curvature[Akinci])
                float surface_tension_curvature_x = - artificial_surface_tension_curvature * SURFACE_TENSION * PARTICLE_MASS * freeSurfaceIndicator *
                                                    (gradColorField[j * 3] - gradColorField[i * 3]);
                // if ((i ==0) && (j < 5) && (count <3)) {
                //     printf("Curvature Parameters 1 x: curvature cosnt %f, surface tension %f, mass %e, freeSurfaceIndicator %f, gradj %f, gradi %f\n", artificial_surface_tension_curvature, SURFACE_TENSION, PARTICLE_MASS, freeSurfaceIndicator, gradColorField[j * 3], gradColorField[i * 3]);
                //     printf("Curvature Parameters 2 x: grad_diff_x, grad_diff_y, grad_diff_z %f, %f, %f\n", (gradColorField[j * 3] - gradColorField[i * 3]), (gradColorField[j * 3 + 1] - gradColorField[i * 3 + 1]), (gradColorField[j * 3 + 2] - gradColorField[i * 3 + 2]));
                // }
                float surface_tension_curvature_y = -artificial_surface_tension_curvature * SURFACE_TENSION * PARTICLE_MASS * freeSurfaceIndicator *
                                                    (gradColorField[j * 3 + 1] - gradColorField[i * 3 + 1]);
                float surface_tension_curvature_z = -artificial_surface_tension_curvature * SURFACE_TENSION * PARTICLE_MASS * freeSurfaceIndicator *
                                                    (gradColorField[j * 3 + 2] - gradColorField[i * 3 + 2]);
                // if ((i ==0) && (j < 5) && (count <3)) {
                //     printf("Curvature force x: %e, y: %e, z: %e\n", surface_tension_curvature_x, surface_tension_curvature_y, surface_tension_curvature_z);
                // }
                force_x += surface_tension_curvature_x;
                force_y += surface_tension_curvature_y;
                force_z += surface_tension_curvature_z;

                // if ((i ==10) && (j < 5) && (count <5)) {
                //     printf("i %d , j %d and count %d\n", i, j, count);
                //     printf("After Curvature force x: %e, y: %e, z: %e\n", force_x, force_y, force_z);
                //     printf("*****************\n");
                // }

                // if (!isfinite(force_x) || !isfinite(force_y) || !isfinite(force_z)) {
                //     printf("Curvature force\n");
                //     printf("ERROR: Infinite or non defenite value of force for particle %d with x,y,z values %f , %f , %f\n", i, force_x, force_y, force_z);
                // }
            }
        }

        // Gravitational Force
        float Gravitational_force_x = CONSTANT_FORCE[0] / PARTICLE_MASS;
        // if ((i ==0) && (count <3)) {
        //     printf("Gravitational Paramters x: cosnt %e, %e, %e, mass %e\n", CONSTANT_FORCE[0], CONSTANT_FORCE[1], CONSTANT_FORCE[2], PARTICLE_MASS);
        // }
        float Gravitational_force_y = CONSTANT_FORCE[1] / PARTICLE_MASS;
        float Gravitational_force_z = CONSTANT_FORCE[2] / PARTICLE_MASS;

        // if ((i ==0) && (count <3)) {
        //     printf("Gravitational force x: %f, y: %f, z: %f\n", Gravitational_force_x, Gravitational_force_y, Gravitational_force_z);
        // }

        force_x += Gravitational_force_x;
        force_y += Gravitational_force_y;
        force_z += Gravitational_force_z;

        // if ((i ==100) && (count <5)) {
        //     printf("i %d and count %d\n", i, count);
        //     printf("After Gravitational force x: %e, y: %e, z: %e\n", force_x, force_y, force_z);
        //     printf("*****************\n");
        // }

        if (densities[i] > 0.0f) {
            // Drag force because of surrounding gas flow

            float Drag_force_x = (cw * 0.5 * density_gas * (3.14159265 / 4) * powf((powf((6 * PARTICLE_MASS) / (3.14159265 * densities[i]), 0.33333)), 2) * 
                (velocity_gas[0] - velocities[i * 3]) * (sqrtf(velocity_gas[0] * velocity_gas[0] + velocity_gas[1] * velocity_gas[1] + velocity_gas[2] * velocity_gas[2]) - 
                sqrtf(velocities[i * 3] * velocities[i * 3] + velocities[i * 3 + 1] * velocities[i * 3 + 1] + velocities[i * 3 + 2] * velocities[i * 3 + 2]))) / PARTICLE_MASS;
            // if ((i ==0) && (count <3)) {
            //     printf("Drag Parameters 1 x: cw %f, mass %e, density %f, density_gas %f\n", cw, PARTICLE_MASS, densities[i], density_gas);
            //     printf("Drag Parameters 2 x: vel_gas %f, vel_i_x %f, parameter 1 %f\n", velocity_gas[0], velocities[i * 3], powf((powf((6 * PARTICLE_MASS) / (3.14159265 * densities[i]), 0.33333)), 2));
            //     printf("Drag Parameters 3 x: parameter 2 %f\n", ((velocity_gas[0] - velocities[i * 3]) * (sqrtf(velocity_gas[0] * velocity_gas[0] + velocity_gas[1] * velocity_gas[1] + velocity_gas[2] * velocity_gas[2]) - 
            //             sqrtf(velocities[i * 3] * velocities[i * 3] + velocities[i * 3 + 1] * velocities[i * 3 + 1] + velocities[i * 3 + 2] * velocities[i * 3 + 2]))) / PARTICLE_MASS);
            //     printf("Drag Parameters y & z: vel_gas_1 %f, vel_gas_2 vel_i_y %f, vel_i_z %f\n", velocity_gas[1], velocity_gas[2], velocities[i * 3 + 1], velocities[i * 3 + 2]);
            // }
            float Drag_force_y = (cw * 0.5 * density_gas * (3.14159265 / 4) * powf((powf((6 * PARTICLE_MASS) / (3.14159265 * densities[i]), 0.33333)), 2) * 
                (velocity_gas[1] - velocities[i * 3 + 1]) * (sqrtf(velocity_gas[0] * velocity_gas[0] + velocity_gas[1] * velocity_gas[1] + velocity_gas[2] * velocity_gas[2]) - 
                sqrtf(velocities[i * 3] * velocities[i * 3] + velocities[i * 3 + 1] * velocities[i * 3 + 1] + velocities[i * 3 + 2] * velocities[i * 3 + 2]))) / PARTICLE_MASS;

            float Drag_force_z = (cw * 0.5 * density_gas * (3.14159265 / 4) * powf((powf((6 * PARTICLE_MASS) / (3.14159265 * densities[i]), 0.33333)), 2) * 
                (velocity_gas[2] - velocities[i * 3 + 2]) * (sqrtf(velocity_gas[0] * velocity_gas[0] + velocity_gas[1] * velocity_gas[1] + velocity_gas[2] * velocity_gas[2]) - 
                sqrtf(velocities[i * 3] * velocities[i * 3] + velocities[i * 3 + 1] * velocities[i * 3 + 1] + velocities[i * 3 + 2] * velocities[i * 3 + 2]))) / PARTICLE_MASS;
            
            // if ((i ==0) && (count <3)) {
            //     printf("Drag force x: %e, y: %e, z: %e\n", Drag_force_x, Drag_force_y, Drag_force_z);
            // }

            force_x += Drag_force_x;
            force_y += Drag_force_y;
            force_z += Drag_force_z;

            // if ((i ==100) && (count <5)) {
            //     printf("i %d and count %d\n", i, count);
            //     printf("After Drag force x: %e, y: %e, z: %e\n", force_x, force_y, force_z);
            //     printf("*****************\n");
            // }
        }

        // forces
        forces[i * 3] = force_x;
        forces[i * 3 + 1] = force_y;
        forces[i * 3 + 2] = force_z;

        // if ((i ==1000) && (count <10)) {
        //     printf("i %d and count %d\n", i, count);
        //     printf("All forces x: %f, y: %f, z: %f\n", forces[i * 3], forces[i * 3 + 1], forces[i * 3 + 2]);
        //     printf("*****************\n");
        // }

    }
}

__global__ void ForcesGeometry( int* neighbor_idsFluidFibre, float* distancesFluidFibre, float* positions, float* positionsFibre, 
                                        float* velocities, float* velocitiesFibre, float* densities, float* forces, float* adhesion, float SMOOTHING_LENGTH, 
                                        float SMOOTHING_LENGTH_FIBRE, float PARTICLE_MASS,float PARTICLE_MASS_FIBRE, float viscosity_artificial,
                                        float DYNAMIC_VISCOSITY_FIBRE, float BASE_DENSITY, float artificial_surface_tension_adhesion, float LJP_DISTANCE, 
                                        int LJP_P1, int LJP_P2, float LJP_COEF, int* total_neighbors, int n_particles, int* d_n_particlesFibre, int count) {
    int n_particlesFibre = *d_n_particlesFibre;// Dereference device pointer
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_particles) {
        float force_x = 0.0f;
        float force_y = 0.0f;
        float force_z = 0.0f;
        float adhesion_z = 0.0f;

        for (int j_in_list = 0; j_in_list < total_neighbors[i]; ++j_in_list) {

            int j = neighbor_idsFluidFibre[i * n_particlesFibre + j_in_list];
            float dist = distancesFluidFibre[i * n_particlesFibre + j_in_list];

            if (densities[i] <= 0.0f) {
                printf("ERROR: Particle Fibre %d has zero density!\n", i);
                return;
            }

            if (dist <= 0.0f) {
                printf("ERROR: Particle %d and Fibre %d have zero/negative distance!\n", i, j);
                return;
            }

            if ((dist > 0.0f) && (densities[i] > 0.0f)) {

                // Fluid-Fiber Viscous Force
                float kernel_viscosity = viscosityKernel(dist, SMOOTHING_LENGTH_FIBRE);
                force_x += viscosity_artificial * kernel_viscosity * DYNAMIC_VISCOSITY_FIBRE * PARTICLE_MASS * 
                        (velocitiesFibre[j * 3] - velocities[i * 3]) / powf(densities[i], 2);
                // if ((i ==0) && (j < 5) && (count <5)) {
                //     printf("Fibre Viscous Parameters 1 x: cosnt %f, mass %e, Dynamic Viscosity Fibre %f, dist %f\n", viscosity_artificial, PARTICLE_MASS, DYNAMIC_VISCOSITY_FIBRE, dist);
                //     printf("Fibre Viscous Parameters 2 x: vel_diff/dens %f, kernel %f\n", (velocitiesFibre[j * 3] - velocities[i * 3]) / powf(densities[i], 2), kernel_viscosity);
                // }       
                
                force_y += viscosity_artificial * kernel_viscosity * DYNAMIC_VISCOSITY_FIBRE * PARTICLE_MASS * 
                        (velocitiesFibre[j * 3 + 1] - velocities[i * 3 + 1]) / powf(densities[i], 2);
                force_z += viscosity_artificial * kernel_viscosity * DYNAMIC_VISCOSITY_FIBRE * PARTICLE_MASS * 
                        (velocitiesFibre[j * 3 + 2] - velocities[i * 3 + 2]) / powf(densities[i], 2);

                // if (!isfinite(force_x) || !isfinite(force_y) || !isfinite(force_z)) {
                //     printf("Fluid-Fibre Viscous force\n");
                //     printf("ERROR: Infinite or non defenite value of force for particle %d with x,y,z values %f , %f , %f\n", i, force_x, force_y, force_z);
                // }


                // if ((i ==0) && (j < 5) && (count <5)) {
                //     printf("After Fiber Viscous force\n");
                //     printf("Fiber Viscous Force x: %f\n", force_x);
                //     printf("Fiber Viscous Force y: %f\n", force_y);
                //     printf("Fiber Viscous Force z: %f\n", force_z);
                // }  

                // Fluid-Fiber Adhesive Force (Akinci)
                float kernel_cohesion = CohesionKernel(dist, SMOOTHING_LENGTH);

                float adhesion_factor = -artificial_surface_tension_adhesion * PARTICLE_MASS * PARTICLE_MASS_FIBRE * kernel_cohesion;
                force_x += adhesion_factor * (positions[i * 3] - positionsFibre[j * 3] )/ dist;
                // if ((i ==0) && (j < 5) && (count <5)) {
                //     printf("Fibre Adhesion Parameters 1 x: cosnt %f, mass %e, mass_fibre %e, kernel %f\n", artificial_surface_tension_adhesion, PARTICLE_MASS, PARTICLE_MASS_FIBRE, kernel_cohesion);
                //     printf("Fibre Adhesion Parameters 2 x: pos_diff/dis %f\n", (positions[i * 3] - positionsFibre[j * 3] )/ dist);
                // }
                force_y += adhesion_factor * (positions[i * 3 + 1] - positionsFibre[j * 3 + 1]) / dist;
                force_z += adhesion_factor * (positions[i * 3 + 2] - positionsFibre[j * 3 + 2]) / dist;
                adhesion_z += adhesion_factor * (positions[i * 3 + 2] - positionsFibre[j * 3 + 2]) / dist;

                // if (!isfinite(force_x) || !isfinite(force_y) || !isfinite(force_z) || !isfinite(adhesion_z)) {
                //     printf("Fluid-Fibre Adhesive force\n");
                //     printf("ERROR: Infinite or non defenite value of force for particle %d with x,y,z values %f , %f , %f, %f\n", i, force_x, force_y, force_z, adhesion_z);
                // }

                // if ((i ==0) && (j < 5) && (count <5)) {
                //     printf("After Fiber Adhesive force\n");
                //     printf("Fiber Adhesive Force x: %f\n", force_x);
                //     printf("Fiber Adhesive Force y: %f\n", force_y);
                //     printf("Fiber Adhesive Force z: %f\n", force_z);
                // }  

                // Lennard-Jones Potential Force
                if (dist < LJP_DISTANCE) {
                    float ljp_factor = LJP_COEF *
                                    (powf(LJP_DISTANCE / dist, LJP_P1) - powf(LJP_DISTANCE / dist, LJP_P2));

                    force_x -= ljp_factor * (positionsFibre[j * 3] - positions[i * 3]) / (dist * dist ) / densities[i];
                    // if ((i ==0) && (j < 5) && (count <5))
                    // {
                    //     printf("Lennard-Jones Parameters 1 x: ljp_factor %f, dist %f, pos_diff %f, dens %f\n", ljp_factor, dist, (positionsFibre[j * 3] - positions[i * 3]), densities[i]);
                    // }
                    force_y -= ljp_factor * (positionsFibre[j * 3 + 1] - positions[i * 3 + 1]) / (dist * dist ) / densities[i];
                    force_z -= ljp_factor * (positionsFibre[j * 3 + 2] - positions[i * 3 + 2]) / (dist * dist ) / densities[i];

                    // if (!isfinite(force_x) || !isfinite(force_y) || !isfinite(force_z)) {
                    //     printf("Lennard-Jones Potential force\n");
                    //     printf("ERROR: Infinite or non defenite value of force for particle %d with x,y,z values %f , %f , %f\n", i, force_x, force_y, force_z);
                    // }

                    // if ((i ==0) && (j < 5) && (count <5)) {
                    //     printf("After Fiber Potential force\n");
                    //     printf("Fiber Potential Force x: %f\n", force_x);
                    //     printf("Fiber Potential Force y: %f\n", force_y);
                    //     printf("Fiber Potential Force z: %f\n", force_z);
                    // }  
                }
            }
        }

        // forces and adhesion
        forces[i * 3] += force_x;
        forces[i * 3 + 1] += force_y;
        forces[i * 3 + 2] += force_z;
        adhesion[i] += adhesion_z;
    }
}

__global__ void deleteunneccessaryparticles( float* velocities, float* forces, int n_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_particles) {
        for (int j = 0; j < 3; ++j) {
            if (isnan(velocities[i * 3 + j]) || fabs(velocities[i * 3 + j]) > 1.0e20 || fabs(velocities[i * 3 + j]) < 1e-4) {
                velocities[i * 3 + j] = 0.0f;
            }
            if (isnan(forces[i * 3 + j]) || fabs(forces[i * 3 + j]) > 1.0e20) {
                forces[i * 3 + j] = 0.0f;
            }
        }
    }
}
