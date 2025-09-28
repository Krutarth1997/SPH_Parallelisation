#ifndef DYNAMICVARIABLES_H
#define DYNAMICVARIABLES_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>

void ForcesFluid(const std::vector<std::vector<int>>& neighbor_ids, const std::vector<std::vector<float>>& distances, const Eigen::MatrixXf& positions, 
                const Eigen::VectorXf& pressures, const Eigen::VectorXf& densities, const Eigen::MatrixXf& velocities, Eigen::MatrixXf& forces,
                float SMOOTHING_LENGTH, float PARTICLE_MASS, float viscosity_artificial,float DYNAMIC_VISCOSITY, float BASE_DENSITY, float artificial_surface_tension_cohesion,
                float artificial_surface_tension_curvature, float SURFACE_TENSION, float density_gas, float cw,const Eigen::Matrix<float, 1, 3>& CONSTANT_FORCE,const Eigen::Matrix<float, 1, 3>& velocity_gas,
                Eigen::MatrixXf gradColorField, int start, int end);

void ForcesFluidGeometry(const std::vector<std::vector<int>>& neighbor_idsFluidFibre, const std::vector<std::vector<float>>& distancesFluidFibre, const Eigen::MatrixXf& positions, 
                        const Eigen::MatrixXf& positionsFibre, const Eigen::MatrixXf& velocities, const Eigen::MatrixXf& velocitiesFibre, const Eigen::VectorXf& densities, 
                        Eigen::MatrixXf& forces, Eigen::VectorXf& adhesion, float SMOOTHING_LENGTH, float SMOOTHING_LENGTH_FIBRE, float PARTICLE_MASS, float PARTICLE_MASS_FIBRE, 
                        float viscosity_artificial, float DYNAMIC_VISCOSITY_FIBRE, float BASE_DENSITY, float artificial_surface_tension_adhesion, float LJP_DISTANCE, int LJP_P1, int LJP_P2, float LJP_COEF , int start, int end);

void deleteunneccessaryparticles(Eigen::MatrixXf& velocities, Eigen::MatrixXf& forces, int start, int end);

#endif 