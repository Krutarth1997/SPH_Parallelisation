#ifndef MATRIXOPERATION_H
#define MATRIXOPERATION_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>

void calculateNeighborhood(const Eigen::MatrixXf& positions1, const Eigen::MatrixXf& positions2, float SMOOTHING_LENGTH, std::vector<std::vector<int>>& neighbor_idsFluidFibre, std::vector<std::vector<float>>& distancesFluidFibre);

void removeRow(Eigen::MatrixXf& matrix, unsigned int rowToRemove);
float euclideanDistance(const Eigen::VectorXf& p1, const Eigen::VectorXf& p2);

void removeOwnElement(std::vector<std::vector<int>>& int_neighbor, std::vector<std::vector<float>>& int_distances, int start, int end);
float removeParticle(Eigen::MatrixXf& velocities, Eigen::MatrixXf& positions, float boundary_x0, float boundary_x1, float boundary_y0, float boundary_y1, float boundary_z0,float boundary_z1);

#endif 