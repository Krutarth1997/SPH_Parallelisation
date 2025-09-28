#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <thread>
#include "matrixOperation.h"
#include <mpi.h>

void neighborhoodMPI(const Eigen::MatrixXf& positions1, const Eigen::MatrixXf& positions2, float SMOOTHING_LENGTH, std::vector<std::vector<int>>& int_neighbor, std::vector<std::vector<float>>& int_distances, int start, int end)
{
    int n_particles1 = positions1.rows();
    int n_particles2 = positions2.rows();
 
    for (int i = start; i <= end; ++i) {
        std::vector<int> neighbors;
        std::vector<float> dists;

        for (int j = 0; j < n_particles2; ++j) {
            float distance = (positions1.row(i) - positions2.row(j)).norm();

            if (distance < SMOOTHING_LENGTH) {
                neighbors.push_back(j);
                dists.push_back(distance);
            }
        }

        int_neighbor[i] = neighbors;
        int_distances[i] = dists;
    }
}

float euclideanDistance(const Eigen::VectorXf& p1, const Eigen::VectorXf& p2) {
    if (p1.size() != p2.size()) {
        return 0.0f;  // Sicherheitsprüfung auf gleiche Dimensionen
    }

    float distanceSquared = 0.0f;
    for (size_t i = 0; i < p1.size(); ++i) {
        float diff = p1[i] - p2[i];
        distanceSquared += diff * diff;
    }

    return std::sqrt(distanceSquared);
}

void removeOwnElement(std::vector<std::vector<int>>& int_neighbor, std::vector<std::vector<float>>& int_distances, int start, int end)
{
    for (int i = start; i <= end; ++i) {
        int currentValue = i;
        for (int j = 0; j < int_neighbor[i].size(); ++j) {
            if (int_neighbor[i][j] == currentValue) {
                int_neighbor[i].erase(int_neighbor[i].begin() + j);
                int_distances[i].erase(int_distances[i].begin() + j);
                --j;
            }
        }
    }
}

bool compareEigenVector(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
    if (a.x() != b.x()) return a.x() < b.x();
    if (a.y() != b.y()) return a.y() < b.y();
    return a.z() < b.z();
}

float removeParticle(Eigen::MatrixXf& velocities, Eigen::MatrixXf& positions, float boundary_x0, float boundary_x1, float boundary_y0, float boundary_y1, float boundary_z0,float boundary_z1)
{
    std::vector<int> delete_list;

    for (int i = 0; i < velocities.rows(); i++)
    {
        if ((abs(positions(i,0)) < boundary_x0) || (abs(positions(i,0)) > boundary_x1) ||
        (abs(positions(i,1)) < boundary_y0) || (abs(positions(i,1)) > boundary_y1) ||
        (abs(positions(i,2)) < boundary_z0) || (abs(positions(i,2)) > boundary_z1))
        {
            delete_list.push_back(i);
        }
    }

    for (int index : delete_list) {
        positions.row(index).swap(positions.row(positions.rows() - 1));
        positions.conservativeResize(positions.rows() - 1, Eigen::NoChange);

        velocities.row(index).swap(velocities.row(velocities.rows() - 1));
        velocities.conservativeResize(velocities.rows() - 1, Eigen::NoChange);
    }


    if (!delete_list.empty())
    {
        std::cout << "Partikel außerhalb der festgelegten Grenzen: " << delete_list.size() << std::endl << std::endl;
    }

    int num_Particles = positions.rows();

    return num_Particles;
}