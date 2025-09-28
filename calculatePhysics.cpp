#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include "calculatePhysics.h"
#include "kernel.h"

void calculateDensities(int int_particles, std::vector<std::vector<int>>& int_neighbor, Eigen::VectorXf& int_densities, float int_smoothing_length,
std::vector<std::vector<float>>& int_distances, float int_particle_mass, int start, int end)
{
    for (int i = start; i < end; ++i)
    {
        for (int j_in_list = 0; j_in_list < int_neighbor[i].size(); ++j_in_list)
        {
            int j = int_neighbor[i][j_in_list];
            int_densities(i) += densityKernel(int_distances[i][j_in_list], int_smoothing_length) * int_particle_mass;
        }
    }
}

void updateVelocities(Eigen::MatrixXf& int_velocities, Eigen::MatrixXf& int_forces, int int_particles, float int_timestep, int start, int end)
{
    for (int i = start; i < end; ++i)
    {
        int_velocities.row(i) = int_velocities.row(i).array() + int_timestep * int_forces.row(i).array();
    }
}

void updatePositions(Eigen::MatrixXf& int_positions, Eigen::MatrixXf& int_velocities, int int_particles, float int_timestep, int start, int end)
{
    for (int i = start; i < end; ++i)
    {
        int_positions.row(i) = int_positions.row(i).array() + int_timestep * int_velocities.row(i).array();
    }
}

float calculateTimestepLength(Eigen::MatrixXf& int_velocities, Eigen::MatrixXf& int_forces, int int_particles, float int_smooth, float Co)
{
	float timestep = 0.000005f;
	
	//Timestep durch Geschwindigkeit
	Eigen::VectorXf velocities_res = int_velocities.rowwise().norm();
    float max_velocity = velocities_res.maxCoeff();
	float timestep_v = abs(int_smooth / max_velocity);
	
	//Timestep durch Beschleunigung
	Eigen::VectorXf acceleration_res = int_forces.rowwise().norm();
	float max_acc = acceleration_res.maxCoeff();
	float timestep_a = abs(pow((int_smooth/max_acc),0.5));
	
	timestep = 0.1 * Co * std::min({timestep_v, timestep_a});	
	
	return timestep;
}
