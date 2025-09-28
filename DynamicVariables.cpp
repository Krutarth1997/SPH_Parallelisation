#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <thread>
#include "DynamicVariables.h"
#include "kernel.h"
#include <mpi.h>

void ForcesFluid(const std::vector<std::vector<int>>& neighbor_ids, const std::vector<std::vector<float>>& distances, const Eigen::MatrixXf& positions, 
                const Eigen::VectorXf& pressures, const Eigen::VectorXf& densities, const Eigen::MatrixXf& velocities, Eigen::MatrixXf& forces, float SMOOTHING_LENGTH,
                float PARTICLE_MASS, float viscosity_artificial, float DYNAMIC_VISCOSITY, float BASE_DENSITY, float artificial_surface_tension_cohesion, float artificial_surface_tension_curvature,
                float SURFACE_TENSION, float density_gas, float cw, const Eigen::Matrix<float, 1, 3>& CONSTANT_FORCE, const Eigen::Matrix<float, 1, 3>& velocity_gas, Eigen::MatrixXf gradColorField, int start, int end)
{
    for (int i = start; i <= end; ++i) {
        int local_index = i - start;

        for (int j_in_list = 0; j_in_list < neighbor_ids[i].size(); ++j_in_list) {
            int j = neighbor_ids[i][j_in_list];

            // Pressure force
            float kernel_value = pressureKernel(distances[i][j_in_list], SMOOTHING_LENGTH);
            Eigen::RowVectorXf pressure_force = kernel_value * ((positions.row(j).array() - positions.row(i).array()) / distances[i][j_in_list]) * ((pressures(i) + pressures(j)) / 2 * densities(j));
            forces.row(local_index) -= pressure_force;

            // Viscous force
            kernel_value = viscosityKernel(distances[i][j_in_list], SMOOTHING_LENGTH);
            Eigen::RowVectorXf viscous_force = viscosity_artificial * kernel_value * DYNAMIC_VISCOSITY * PARTICLE_MASS * (velocities.row(j) - velocities.row(i)) / pow(densities(j), 2);
            forces.row(local_index) += viscous_force;

            // Surface tension (Cohesion [Akinci])
            float freeSurfaceIndicator = ((2 * BASE_DENSITY) / (densities(i) + densities(j)));
            kernel_value = CohesionKernel(distances[i][j_in_list], SMOOTHING_LENGTH);
            Eigen::RowVectorXf surface_tension_cohesion = -artificial_surface_tension_cohesion * 2 * PARTICLE_MASS * ((positions.row(i).array() - positions.row(j).array()) / distances[i][j_in_list]) * kernel_value;
            forces.row(local_index) += surface_tension_cohesion;

            // Surface tension (Curvature [Akinci])
            Eigen::RowVectorXf surface_tension_curvature = -artificial_surface_tension_curvature * SURFACE_TENSION * PARTICLE_MASS * freeSurfaceIndicator * (gradColorField.row(j).array() - gradColorField.row(i).array());
            forces.row(local_index) += surface_tension_curvature;
        }

        // Gravitational force
        forces.row(local_index) += CONSTANT_FORCE.row(0) / PARTICLE_MASS;

        // Drag force because of surrounding gas flow
        forces.row(local_index) += (cw * 0.5 * density_gas * (3.14159265 / 4) * pow((pow((6 * PARTICLE_MASS) / (3.14159265 * densities(i)), 0.33333)), 2) * (velocity_gas.row(0) - velocities.row(i)) * (velocity_gas.row(0).norm() - velocities.row(i).norm())) / PARTICLE_MASS;
    }
}

void ForcesFluidGeometry(const std::vector<std::vector<int>>& neighbor_idsFluidFibre, const std::vector<std::vector<float>>& distancesFluidFibre, const Eigen::MatrixXf& positions, 
                        const Eigen::MatrixXf& positionsFibre, const Eigen::MatrixXf& velocities, const Eigen::MatrixXf& velocitiesFibre, const Eigen::VectorXf& densities, 
                        Eigen::MatrixXf& forces, Eigen::VectorXf& adhesion, float SMOOTHING_LENGTH, float SMOOTHING_LENGTH_FIBRE, float PARTICLE_MASS, float PARTICLE_MASS_FIBRE, 
                        float viscosity_artificial, float DYNAMIC_VISCOSITY_FIBRE, float BASE_DENSITY, float artificial_surface_tension_adhesion, float LJP_DISTANCE, int LJP_P1,
                         int LJP_P2, float LJP_COEF, int start, int end)
{
    for (int i = start; i <= end; ++i) {
        int local_index = i - start;

        for (int j_in_list = 0; j_in_list < neighbor_idsFluidFibre[i].size(); ++j_in_list) {
            int j = neighbor_idsFluidFibre[i][j_in_list];
            
            // Viscous force (wall friction)
            float kernel_value = viscosityKernel(distancesFluidFibre[i][j_in_list], SMOOTHING_LENGTH_FIBRE);
            Eigen::RowVectorXf viscous_force = viscosity_artificial * kernel_value * DYNAMIC_VISCOSITY_FIBRE * PARTICLE_MASS * (velocitiesFibre.row(j) - velocities.row(i)) / pow(densities(i), 2);
            forces.row(local_index) += viscous_force;
            
            // Adhesive force [Akinci]
            kernel_value = CohesionKernel(distancesFluidFibre[i][j_in_list], SMOOTHING_LENGTH);
            Eigen::RowVectorXf adhesion_force = -artificial_surface_tension_adhesion * PARTICLE_MASS * PARTICLE_MASS_FIBRE * kernel_value * ((positions.row(i).array() - positionsFibre.row(j).array()) / distancesFluidFibre[i][j_in_list]);
            forces.row(local_index) += adhesion_force;
            adhesion(local_index) += adhesion_force(2);

            // Lennard-Jones Potential (Repulsive forces between fibre and fluid when LJP_DISTANCE is reached [Monaghan94])
            if (distancesFluidFibre[i][j_in_list] < LJP_DISTANCE) {
                Eigen::RowVectorXf ljp_force = (LJP_COEF * (pow(LJP_DISTANCE / distancesFluidFibre[i][j_in_list], LJP_P1) - pow(LJP_DISTANCE / distancesFluidFibre[i][j_in_list], LJP_P2)) * ((positionsFibre.row(j) - positions.row(i)) / pow(distancesFluidFibre[i][j_in_list], 2))) / densities(i);
                forces.row(local_index) -= ljp_force;
            }
        }
    }
}

void deleteunneccessaryparticles(Eigen::MatrixXf& velocities, Eigen::MatrixXf& forces, int start, int end)
{
    for (int i = start; i <= end; ++i)
    {
        int local_i = i - start;
        for (int j = 0; j < 3; j++)
        {

            if (std::isnan(velocities.coeff(local_i, j)) || abs(velocities(local_i, j)) > 1.0e20 || abs(velocities(local_i, j)) < 1e-4)
            {
                velocities(local_i, j) = 0.0f;
            }
            if (std::isnan(forces.coeff(i, j)) || abs(forces(i, j)) > 1.0e20)
            {
                forces(i, j) = 0.0f;
            }
        }
    }
}