#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include "kernel.h"

float CubicSplineKernel(float dist, float smooth)
{
	float norm = 8 / (3.14159265f * std::pow(smooth, 3));
    float value;

    if (dist <= 0.5 * smooth) {
		value = 6 * (std::pow((dist / smooth), 3) - std::pow((dist / smooth), 2)) + 1;
    } else if (dist > 0.5 * smooth && dist <= smooth) {
		value = 2 * std::pow((1 - (dist / smooth)), 3);
    } else {
		value = 0;
    }

	return value * norm;
}

float CubicSplineDifKernel(float dist, float smooth)
{
	if (dist <= 0.5 * smooth)
	{
		float norm = 8 / (3.14159265f * std::pow(smooth, 3));
		float value = ((3*pow(dist,2))/pow(smooth,3)) - ((2*dist)/pow(smooth,2));
        return 6 * value * norm;
	}
    else if (dist > 0.5 * smooth && dist <= smooth)
    {
        float norm = 8 / (3.14159265f * std::pow(smooth, 3));
        float value = pow((1- (dist/smooth)),2);
        return (6 / smooth) * value * norm;
    }
	return 0;
}

float CubicSplineDif2Kernel(float dist, float smooth)
{
	if (dist <= 0.5 * smooth)
	{
		float norm = 8 / (3.14159265f * std::pow(smooth, 3));
		float value = ((6*dist)/pow(smooth,3)) - (2/pow(smooth,2));
        return 6 * value * norm;
	}
    else if (dist > 0.5 * smooth && dist <= smooth)
    {
        float norm = 8 / (3.14159265f * std::pow(smooth, 3));
        float value = 1- (dist/smooth);
        return (12 / pow(smooth,2)) * value * norm;
    }
	return 0;
}

float SpikyKernel(float dist, float smooth)
{
	if (dist < smooth)
	{
		float norm = 15 / (3.14159265f * std::pow(smooth, 6));
		float value = pow((smooth - dist),3);
        return value * norm;
	}
	return 0;
}

float SpikyDifKernel(float dist, float smooth)
{
	if (dist < smooth)
	{
		float norm = 15 / (3.14159265f * std::pow(smooth, 6));
		float value = pow((smooth - dist),2);
        return -3 * value * norm;
	}
	return 0;
}

float SpikyDif2Kernel(float dist, float smooth)
{
	if (dist < smooth)
	{
		float norm = 15 / (3.14159265f * std::pow(smooth, 6));
		float value = smooth - dist;
        return 6 * value * norm;
	}
	return 0;
}

float Poly6Kernel(float dist, float smooth)
{
	if (dist <= smooth)
	{
		float norm = 315.0f / (64.0f * 3.14159265f * std::pow(smooth, 9));
		float value = pow((pow(smooth,2) - pow(dist,2)),3);
        return value * norm;
	}
	return 0;
}

float Poly6DifKernel(float dist, float smooth)
{
	if (dist <= smooth)
	{
		float norm = 315.0f / (64.0f * 3.14159265f * std::pow(smooth, 9));
		float value = pow((pow(smooth,2) - pow(dist,2)),2);
        return -6 * value * norm * dist;
	}
	return 0;
}

float Poly6Dif2Kernel(float dist, float smooth)
{
	if (dist <= smooth)
	{
		float norm = 315.0f / (64.0f * 3.14159265f * std::pow(smooth, 9));
		float value = pow(smooth,2) - pow(dist,2);
        return 24 * value * norm * pow(dist,2);
	}
	return 0;
}

float CohesionKernel(float dist, float smooth)
{
	if (dist <= 0.5 * smooth)
	{
		double norm = 32 / (3.14159265f * std::pow(smooth, 9));
		double value = 2 * pow((smooth-dist),3) * pow(dist,3) - (pow(smooth,6)/64);
        return float(norm * value);
	}
    else if (dist > 0.5 * smooth && dist <= smooth)
    {
        double norm = 32 / (3.14159265f * std::pow(smooth, 9));
        double value = pow((smooth-dist),3) * pow(dist,3);
        return float(value * norm);
    }
	return 0;
}

float AdhesionKernel(float dist, float smooth)
{
	//if (dist > 0.5 * smooth && dist <= smooth)
	if (dist <= smooth)
	{
		double norm = 0.007 / pow(smooth,3.25);
		double value = pow(-((4*pow(dist,2))/smooth)+6*dist-2*smooth,(1/4));
		return float(norm*value);
	}
	return 0;
}


float DensityKernel(float dist, float smooth)
{
	return CubicSplineKernel(dist, smooth);
}

Eigen::MatrixXf DensityKernelParticle(int n_particles, const std::vector<std::vector<int>>& neighbor_ids, const std::vector<std::vector<float>>& distances, float SMOOTHING_LENGTH, float PARTICLE_MASS)
{
    Eigen::VectorXf densities(n_particles);
    densities.setZero();

    for (int i = 0; i < n_particles; ++i)
    {
        for (int j_in_list = 0; j_in_list < neighbor_ids[i].size(); ++j_in_list)
        {
            float kernel_value = CubicSplineKernel(distances[i][j_in_list], SMOOTHING_LENGTH);
            densities(i) += kernel_value * PARTICLE_MASS;
        }
    }

	return densities;
}

Eigen::MatrixXf DensityCorrectedParticle(int n_particles, const std::vector<std::vector<int>>& neighbor_ids, const std::vector<std::vector<float>>& distances, float SMOOTHING_LENGTH, float PARTICLE_MASS, Eigen::VectorXf densities)
{
	Eigen::VectorXf densities_corr(n_particles);
    densities_corr.setZero();

    for (int i = 0; i < n_particles; ++i)
	{
        for (int j_in_list = 0; j_in_list < neighbor_ids[i].size(); ++j_in_list)
		{
            float kernel_value = CubicSplineKernel(distances[i][j_in_list], SMOOTHING_LENGTH);
            densities_corr(i) += (PARTICLE_MASS * kernel_value) / ((PARTICLE_MASS/densities[i]) * kernel_value);
        }
    }

	return densities_corr;
}

Eigen::MatrixXf CheckBoundaryParticle(int n_particles, const std::vector<std::vector<int>>& neighbor_idsFluidFibre, const std::vector<std::vector<float>>& distancesFluidFibre, float SMOOTHING_LENGTH_FIBRE, float PARTICLE_MASS_FIBRE)
{
	Eigen::VectorXf randpart(n_particles);
	randpart.setZero();

	for (int i = 0; i < n_particles; ++i)
	{
		for (int j_in_list = 0; j_in_list < neighbor_idsFluidFibre[i].size(); ++j_in_list)
		{
			float kernel_value = DensityKernel(distancesFluidFibre[i][j_in_list], SMOOTHING_LENGTH_FIBRE);
			randpart(i) += kernel_value * PARTICLE_MASS_FIBRE;
		}
	}

	return randpart;
 //Here we have to change randpart += densities; and return something
}

float DensityCorrectedKernel(float dist, float smooth)
{
	return CubicSplineKernel(dist, smooth);
}

Eigen::MatrixXf ColorFieldKernel(int n_particles, const std::vector<std::vector<int>>& neighbor_ids, const Eigen::MatrixXf& positions, const std::vector<std::vector<float>>& distances, float SMOOTHING_LENGTH, float PARTICLE_MASS, Eigen::VectorXf densities)
{
	Eigen::MatrixXf gradColorField(n_particles, 3);
    gradColorField.setZero();
    for (int i = 0; i < n_particles; ++i)
	{
		for (int j_in_list = 0; j_in_list < neighbor_ids[i].size(); ++j_in_list)
		{
			int j = neighbor_ids[i][j_in_list];
			float kernel_value = CubicSplineDifKernel(distances[i][j_in_list], SMOOTHING_LENGTH);
			gradColorField.row(i) += kernel_value * SMOOTHING_LENGTH * (PARTICLE_MASS / densities(j)) * ((positions.row(j) - positions.row(i)) / distances[i][j_in_list]);
		}
	}

    //Delete NaN-Values
    for (int i=0; i<n_particles;i++)
	{
		for (int j =0; j<3;j++)
		{
			if (std::isnan(gradColorField.coeff(i, j)))
			{
				gradColorField(i,j) = 0.0f;
			}
		}
	}

	return gradColorField;
}

float pressureKernel(float dist, float smooth)
{
	return SpikyKernel(dist, smooth);
}

float viscosityKernel(float dist, float smooth)
{
	return CubicSplineKernel(dist, smooth);
}

float densityKernel(float dist, float smooth)
{
	return Poly6Kernel(dist, smooth);
}
