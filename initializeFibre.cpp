#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include "initializeFibre.h"
#include <algorithm>
#include <mpi.h>

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> generateVerticalFibre(Eigen::MatrixXf& positions_fibre, Eigen::MatrixXf& velocities_fibre)
{
    //-----------Geometrieeinstellungen vertikale Faser (Fluid)----------
    int num_height = 25;
    int num_radius = 2;
    int num_circumference[num_radius] = {46,50};
    float inner_radius = 7.6f;
    float outer_radius = 8.0f;
    float height = 20;
    float offset_z = 0;
    //-----------Geometrieeinstellungen vertikale Faser (Fluid)----------


    std::cout << "\nGeometrie (vertikale Faser) wird initialisiert...\n";
    int index_nr = 0;

    for (int i = 0; i < num_height; ++i) {
        for (int k = 0; k < num_radius; ++k) {
            for (int j = 0; j < num_circumference[k]; ++j) {
                float phi = 2* M_PI * (j / static_cast<float>(num_circumference[k]));
                float r = inner_radius + (outer_radius - inner_radius) * (k /static_cast<float>(num_radius-1));
                float x = r * std::cos(phi);
                float y = r * std::sin(phi);
                float z = height * (i / static_cast<float>(num_height - 1));

                positions_fibre(index_nr, 0) = x;
                positions_fibre(index_nr, 1) = y;
                positions_fibre(index_nr, 2) = z + offset_z;
                velocities_fibre.row(index_nr) << 0.0f, 0.0f, 0.0f;

                //std::cout << "\nPositionsmatrix:\n" << positions_fibre << std::endl;

                ++index_nr;
            }
        }
    }

    return std::make_pair(positions_fibre, velocities_fibre);
}

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> generateHorizontalFibre(Eigen::MatrixXf& positions_fibre, Eigen::MatrixXf& velocities_fibre)
{
    //-----------Geometrieeinstellungen vertikale Faser (Fluid)----------
    int num_height = 85;
    int num_circumference = 20;
    float radius = 0.5e-6;
    float height = 15.0e-6;
    float offset_y = 0;
    //-----------Geometrieeinstellungen vertikale Faser (Fluid)----------


    std::cout << "\nGeometrie (vertikale Faser) wird initialisiert...\n";
    int index_nr = 0;

    for (int i = 0; i < num_height; ++i) {
        for (int j = 0; j < num_circumference; ++j) {
            float phi = 2* M_PI * (j / static_cast<float>(num_circumference));
            float x = radius * std::cos(phi);
            float y = height * (i / static_cast<float>(num_height - 1));
            float z = radius * std::sin(phi);

            positions_fibre(index_nr, 0) = x;
            positions_fibre(index_nr, 1) = y + offset_y;
            positions_fibre(index_nr, 2) = z;
            velocities_fibre.row(index_nr) << 0.0f, 0.0f, 0.0f;

            ++index_nr;
        }
    }

    return std::make_pair(positions_fibre, velocities_fibre);
}

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> generateFlatSurface(Eigen::MatrixXf& positions_fibre, Eigen::MatrixXf& velocities_fibre)
{
    float height = 0;
    int n_rows = 40;
    int n_columns = 40; //40
    int n_height = 1;
    Eigen::RowVector2f SPAWN_X_LIM(-3.0e-6,3.0e-6);
    Eigen::RowVector2f SPAWN_Y_LIM(-3.0e-6,3.0e-6); //3.5
    Eigen::RowVector2f SPAWN_Z_LIM(0,-1e-6);

    int index_nr = 0;
    std::cout << "\nInitialisiere Geometrie(Ebene)...\n";
        for (int i = 0; i < n_rows; ++i)
        {
            for (int j = 0; j < n_columns; ++j)
            {
                for (int k = 0; k < n_height; k++)
                {
                    float x = SPAWN_X_LIM(0) + (SPAWN_X_LIM(1) - SPAWN_X_LIM(0)) * (i / static_cast<float>(n_rows - 1));
                    float y = SPAWN_Y_LIM(0) + (SPAWN_Y_LIM(1) - SPAWN_Y_LIM(0)) * (j / static_cast<float>(n_columns - 1));
                    float z = SPAWN_Z_LIM(0);

                    positions_fibre(index_nr, 0) = x;
                    positions_fibre(index_nr, 1) = y;
                    positions_fibre(index_nr, 2) = z + height;
                    velocities_fibre.row(index_nr) << 0.0f, 0.0f, 0.0f;

                    ++index_nr;
                }
            }  
        }
    return std::make_pair(positions_fibre, velocities_fibre);
}

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> generateSTLFibre(Eigen::MatrixXf& positions_fibre, Eigen::MatrixXf& velocities_fibre, int* n_part_fibre)
{
    std::cout << "\n\nRead STL data...\n\n";
    std::ifstream file("Fibre.stl");

    if (!file.is_open()) {
        std::cout << "\n\nFehler beim Öffnen der Datei!\n\n" << std::endl;
        return std::make_pair(positions_fibre, velocities_fibre);
    }

    std::vector<Eigen::Vector3f> vertices;
    std::string line;
    int line_counter = 0;

    //Search for "vertex" / points in STL
    while (std::getline(file, line)) {
        ++ line_counter;
        if (line.find("vertex") != std::string::npos) {
            float x, y, z;
            std::sscanf(line.c_str(), " vertex %f %f %f", &x, &y, &z);
            vertices.push_back(Eigen::Vector3f(x, y, z));
        }
    }
	
	//Delete dublicates
    std::sort(vertices.begin(), vertices.end(), compareEigenVector);
    vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());

    int n_particles = vertices.size();
    *n_part_fibre = n_particles;

    //convert from mm to m
    for (auto& vec : vertices) {
        vec /= 1000.0;
    }

    //save as Matrix
    for (int i = 0; i < n_particles; ++i) {
        positions_fibre.row(i) = vertices[i];
        velocities_fibre.row(i) << 0.0f, 0.0f, 0.0f;
    }

	//Change size of fibre_Matrix to actually necessary fibre particles
    positions_fibre.conservativeResize(n_particles, Eigen::NoChange);
    velocities_fibre.conservativeResize(n_particles, Eigen::NoChange);
    return std::make_pair(positions_fibre, velocities_fibre);
}
