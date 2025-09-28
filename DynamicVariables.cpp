#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <thread>
#include "DynamicVariables.h"
#include "kernel.h"

void ForcesFluid(int n_particles, const std::vector<std::vector<int>>& neighbor_ids, const std::vector<std::vector<float>>& distances, const Eigen::MatrixXf& positions, 
                const Eigen::VectorXf& pressures, const Eigen::VectorXf& densities, const Eigen::MatrixXf& velocities, Eigen::MatrixXf& forces, float SMOOTHING_LENGTH,
                float PARTICLE_MASS, float viscosity_artificial,float DYNAMIC_VISCOSITY, float BASE_DENSITY, float artificial_surface_tension_cohesion, float artificial_surface_tension_curvature,
                float SURFACE_TENSION, float density_gas, float cw,const Eigen::Matrix<float, 1, 3>& CONSTANT_FORCE,const Eigen::Matrix<float, 1, 3>& velocity_gas, Eigen::MatrixXf gradColorField, int count)
{
    for (int i = 0; i < n_particles; ++i) {
        for (int j_in_list = 0; j_in_list < neighbor_ids[i].size(); ++j_in_list)
        {
            int j = neighbor_ids[i][j_in_list];

            //Pressure force
            float kernel_value = pressureKernel(distances[i][j_in_list], SMOOTHING_LENGTH);
            Eigen::RowVectorXf pressure_force = kernel_value * ((positions.row(j).array() - positions.row(i).array()) / distances[i][j_in_list]) * ((pressures(i) + pressures(j)) / 2 * densities(j));
            // if ((i ==100) && (j < 5) && (count <10)) {
            //     printf("Pressure Parameters 1 x: kernel %e, posi %f, posj %f, dist %f\n", kernel_value, positions(i, 0), positions(j, 0), distances[i][j_in_list]);
            //     printf("Pressure Parameters 2 x: pressi %e, pressj %e, densj %f\n", pressures(i) , pressures(j), densities(j));
            //     printf("Pressure Parameters 3 x: pos_diff/dis %f, Pressure / density %e\n",((positions(j, 0) - positions(i, 0)) / distances[i][j_in_list]), ((pressures(i) + pressures(j)) / 2 * densities(j)));
            // }
            forces.row(i) -= pressure_force;

            // if ((i ==0) && (j < 5) && (count <5)) {
            //     printf("Particle %d, Pressure Force: (%e, %e, %e)\n", i, pressure_force(i, 0), pressure_force(i, 1), pressure_force(i, 2));
            // }

            // if ((i ==0) && (j < 5) && (count <10)) {
            //     printf("i %d , j %d and count %d\n", i, j, count);
            //     printf("Particle %d, After Pressure Force: (%e, %e, %e)\n", i, forces(i, 0), forces(i, 1), forces(i, 2));
            //     printf("*****************\n");
            // }
            
            //viscous force
            kernel_value = viscosityKernel(distances[i][j_in_list], SMOOTHING_LENGTH);
            Eigen::RowVectorXf viscous_force = viscosity_artificial * kernel_value * DYNAMIC_VISCOSITY * PARTICLE_MASS * (velocities.row(j)- velocities.row(i)) / pow(densities(j),2);
            forces.row(i) += viscous_force;

            // if ((i ==0) && (j < 5) && (count <3)) {
            //     printf("Particle %d, Viscous Force: (%e, %e, %e)\n", i, viscous_force(i, 0), viscous_force(i, 1), viscous_force(i, 2));
            // }

            // if ((i ==0) && (j < 5) && (count <5)) {
            //     printf("i %d , j %d and count %d\n", i, j, count);
            //     printf("Particle %d, After Viscous Force: (%e, %e, %e)\n", i, forces(i, 0), forces(i, 1), forces(i, 2));
            //     printf("*****************\n");
            // }

            //Surfacetension (Cohesion [Akinci])
            float freeSurfaceIndicator = ((2*BASE_DENSITY)/(densities(i)+densities(j)));
            kernel_value = CohesionKernel(distances[i][j_in_list], SMOOTHING_LENGTH);
            Eigen::RowVectorXf surface_tension_cohesion = - artificial_surface_tension_cohesion * 2 * PARTICLE_MASS * ((positions.row(i).array() - positions.row(j).array()) / distances[i][j_in_list]) * kernel_value;
            // if ((i == 100) && (j_in_list < 5) && (count <10)) {
            //     printf("Cohesion Parameters 1 x: cosnt %f, mass %e, posi %f, posj %f, dist %f\n", artificial_surface_tension_cohesion, PARTICLE_MASS, positions(i, 1), positions(j, 1), distances[i][j_in_list]);
            //     printf("Cohesion Parameters 2 x: pos_diff/dis %f, kernel %e\n", ((positions(i, 1) - positions(j, 1)) / distances[i][j_in_list]), kernel_value);
            // }
            forces.row(i) += surface_tension_cohesion;

            // if ((i == 0) && (j_in_list < 5) && (count <10)) {
            //     printf("Particle %d, Cohesion Force: (%e, %e, %e)\n", i, surface_tension_cohesion(i, 0), surface_tension_cohesion(i, 1), surface_tension_cohesion(i, 2));
            // }
            
            // if ((i ==0) && (j < 5) && (count <10)) {
            //     printf("i %d , j %d and count %d\n", i, j, count);
            //     printf("Particle %d, After Cohesion Force: (%e, %e, %e)\n", i, forces(i, 0), forces(i, 1), forces(i, 2));
            //     printf("*****************\n");
            // }

            //Surfacetension (Curvature [Akinci])
            Eigen::RowVectorXf surface_tension_curvature = - artificial_surface_tension_curvature * SURFACE_TENSION * PARTICLE_MASS * freeSurfaceIndicator * (gradColorField.row(j).array()-gradColorField.row(i).array());
            forces.row(i) += surface_tension_curvature;

            // if ((i ==0) && (j < 5) && (count <3)) {
            //     printf("Particle %d, Curvature Force: (%e, %e, %e)\n", i, surface_tension_curvature(i, 0), surface_tension_curvature(i, 1), surface_tension_curvature(i, 2));
            // }

            // if ((i ==0) && (j < 5) && (count <5)) {
            //     printf("i %d , j %d and count %d\n", i, j, count);
            //     printf("Particle %d, After Curvature Force: (%e, %e, %e)\n", i, forces(i, 0), forces(i, 1), forces(i, 2));
            //     printf("*****************\n");
            // }
        }

        //gravitaional force
        Eigen::RowVectorXf gravitational_force = CONSTANT_FORCE.row(0) / PARTICLE_MASS;
        forces.row(i) += gravitational_force;

        // if ((i ==0) && (count <3)) {
        //     printf("Particle %d, Gravitational Force: (%f, %f, %f)\n", i, gravitational_force(i, 0), gravitational_force(i, 1), gravitational_force(i, 2));
        // }

        // if ((i ==0) && (count <5)) {
        //     printf("i %d and count %d\n", i, count);
        //     printf("Particle %d, After Gravitational Force: (%e, %e, %e)\n", i, forces(i, 0), forces(i, 1), forces(i, 2));
        //     printf("*****************\n");
        // }

        //drag force because of sorrounding gas flow
        Eigen::RowVectorXf drag_force= (cw * 0.5* density_gas * (3.14159265/4) * pow((pow((6*PARTICLE_MASS)/(3.14159265*densities[i]),0.33333)),2) * (velocity_gas.row(0)-velocities.row(i)) * (velocity_gas.row(0).norm()-velocities.row(i).norm())) / PARTICLE_MASS;
        forces.row(i) += drag_force;

        // if ((i ==0) && (count <5)) {
        //     printf("Particle %d, Drag Force: (%f, %f, %f)\n", i, drag_force(i, 0), drag_force(i, 1), drag_force(i, 2));
        // }

        // if ((i ==0) && (count <5)) {
        //     printf("i %d and count %d\n", i, count);
        //     printf("Particle %d, After Drag Force: (%e, %e, %e)\n", i, forces(i, 0), forces(i, 1), forces(i, 2));
        //     printf("*****************\n");
        // }

        // if ((i ==0) && (count <10)) {
        //     printf("i %d and count %d\n", i, count);
        //     printf("Particle %d, After all Forces: (%f, %f, %f)\n", i, forces(i, 0), forces(i, 1), forces(i, 2));
        //     printf("*****************\n");
        // }
    }
}

void ForcesFluidGeometry(int n_particles, const std::vector<std::vector<int>>& neighbor_idsFluidFibre, const std::vector<std::vector<float>>& distancesFluidFibre, const Eigen::MatrixXf& positions, 
                        const Eigen::MatrixXf& positionsFibre, const Eigen::MatrixXf& velocities, const Eigen::MatrixXf& velocitiesFibre, const Eigen::VectorXf& densities, 
                        Eigen::MatrixXf& forces, Eigen::VectorXf& adhesion, float SMOOTHING_LENGTH, float SMOOTHING_LENGTH_FIBRE, float PARTICLE_MASS, float PARTICLE_MASS_FIBRE, 
                        float viscosity_artificial, float DYNAMIC_VISCOSITY_FIBRE, float BASE_DENSITY, float artificial_surface_tension_adhesion, float LJP_DISTANCE, int LJP_P1, int LJP_P2, float LJP_COEF, int count)
{
    for (int i = 0; i < n_particles; ++i)
    {
        for (int j_in_list = 0; j_in_list < neighbor_idsFluidFibre[i].size(); ++j_in_list) {
            int j = neighbor_idsFluidFibre[i][j_in_list];
            
            //viscous force (wall friction)
            float kernel_value = viscosityKernel(distancesFluidFibre[i][j_in_list], SMOOTHING_LENGTH_FIBRE);
            Eigen::RowVectorXf viscous_force = viscosity_artificial * kernel_value * DYNAMIC_VISCOSITY_FIBRE * PARTICLE_MASS * (velocitiesFibre.row(j)- velocities.row(i)) / pow(densities(i),2);
            forces.row(i) += viscous_force;
            
            //adhesive force [Akinci]
            kernel_value = CohesionKernel(distancesFluidFibre[i][j_in_list], SMOOTHING_LENGTH);
            Eigen::RowVectorXf adhesion_force = - artificial_surface_tension_adhesion * PARTICLE_MASS * PARTICLE_MASS_FIBRE * kernel_value * ((positions.row(i).array() - positionsFibre.row(j).array()) / distancesFluidFibre[i][j_in_list]);
            forces.row(i) += adhesion_force;
            adhesion(i) += adhesion_force(2);

            //Lennard Jones Potential (Repulsive forces between fibre and fluid when LJP_DISTANCE is reached [Monaghan94])
            if (distancesFluidFibre[i][j_in_list] < LJP_DISTANCE)
            {
                Eigen::RowVectorXf ljp_force = (LJP_COEF * (pow(LJP_DISTANCE / distancesFluidFibre[i][j_in_list], LJP_P1) - pow(LJP_DISTANCE / distancesFluidFibre[i][j_in_list], LJP_P2)) * ((positionsFibre.row(j)-positions.row(i)) / pow(distancesFluidFibre[i][j_in_list],2))) / densities(i);
                forces.row(i) -= ljp_force;
            }
        }
    }
}

void deleteunneccessaryparticles(int n_particles, Eigen::MatrixXf& velocities, Eigen::MatrixXf& forces)
{
    for (int i=0; i<n_particles;i++)
    {
        for (int j =0; j<3;j++)
        {
            if (std::isnan(velocities.coeff(i, j)) || abs(velocities(i,j)) > 1.0e20 || abs(velocities(i,j)) < 1e-4)
            {
                velocities(i,j) = 0.0f;
            }
            if (std::isnan(forces.coeff(i, j)) || abs(forces(i,j)) > 1.0e20)
            {
                forces(i,j) = 0.0f;
            }
        }
    }
}