#include <iostream>
#include <fstream>
#include <vector>
#include <vector>
#include <filesystem>
#include <string>
#include <tuple>
#include "initializeFibre.h"
#include <algorithm>

std::pair<std::vector<float>, std::vector<float>> generateVerticalFibre(std::vector<float>& positions_fibre, std::vector<float>& velocities_fibre)
{
    //-----------Geometrieeinstellungen vertikale Faser (Fluid)----------
    int num_height = 25;
    const int num_radius = 2;
    int num_circumference[num_radius] = {46,50};
    float inner_radius = 7.6f;
    float outer_radius = 8.0f;
    float height = 20;
    float offset_z = 0;
    float M_PI = 3.14;
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

                positions_fibre[index_nr * 3 + 0] = x; // X-coordinate
                positions_fibre[index_nr * 3 + 1] = y; // Y-coordinate
                positions_fibre[index_nr * 3 + 2] = z + offset_z; // Z-coordinate

                velocities_fibre[index_nr * 3 + 0] = 0.0f; // X-velocity
                velocities_fibre[index_nr * 3 + 1] = 0.0f;  // Y-velocity
                velocities_fibre[index_nr * 3 + 2] = 0.0f;  // Z-velocity

                ++index_nr;
            }
        }
    }

    return {positions_fibre, velocities_fibre};
}

std::pair<std::vector<float>, std::vector<float>> generateHorizontalFibre(std::vector<float>& positions_fibre, std::vector<float>& velocities_fibre)
{
    //-----------Geometrieeinstellungen vertikale Faser (Fluid)----------
    int num_height = 85;
    int num_circumference = 20;
    float radius = 0.5e-6;
    float height = 15.0e-6;
    float offset_y = 0;
    float M_PI = 3.14;
    //-----------Geometrieeinstellungen vertikale Faser (Fluid)----------


    std::cout << "\nGeometrie (vertikale Faser) wird initialisiert...\n";
    int index_nr = 0;

    for (int i = 0; i < num_height; ++i) {
        for (int j = 0; j < num_circumference; ++j) {
            float phi = 2* M_PI * (j / static_cast<float>(num_circumference));
            float x = radius * std::cos(phi);
            float y = height * (i / static_cast<float>(num_height - 1));
            float z = radius * std::sin(phi);

            positions_fibre[index_nr * 3 + 0] = x; // X-coordinate
            positions_fibre[index_nr * 3 + 1] = y + offset_y; // Y-coordinate
            positions_fibre[index_nr * 3 + 2] = z; // Z-coordinate

            velocities_fibre[index_nr * 3 + 0] = 0.0f; // X-velocity
            velocities_fibre[index_nr * 3 + 1] = 0.0f;  // Y-velocity
            velocities_fibre[index_nr * 3 + 2] = 0.0f;  // Z-velocity

            ++index_nr;
        }
    }

    return {positions_fibre, velocities_fibre};
}

std::pair<std::vector<float>, std::vector<float>> generateFlatSurface(std::vector<float>& positions_fibre, std::vector<float>& velocities_fibre)
{
    float height = 0;
    int n_rows = 40;
    int n_columns = 40; //40
    int n_height = 1;

    float SPAWN_X_LIM[] = {-3.0e-6,3.0e-6};
    float SPAWN_Y_LIM[] = {-3.0e-6,3.0e-6}; //3.5
    float SPAWN_Z_LIM[] = {0,-1e-6};

    int index_nr = 0;
    std::cout << "\nInitialisiere Geometrie(Ebene)...\n";
        for (int i = 0; i < n_rows; ++i)
        {
            for (int j = 0; j < n_columns; ++j)
            {
                for (int k = 0; k < n_height; k++)
                {
                    float x = SPAWN_X_LIM[0] + (SPAWN_X_LIM[1] - SPAWN_X_LIM[0]) * (i / static_cast<float>(n_rows - 1));
                    float y = SPAWN_Y_LIM[0] + (SPAWN_Y_LIM[1] - SPAWN_Y_LIM[0]) * (j / static_cast<float>(n_columns - 1));
                    float z = SPAWN_Z_LIM[0];

                    positions_fibre[index_nr * 3 + 0] = x; // X-coordinate
                    positions_fibre[index_nr * 3 + 1] = y; // Y-coordinate
                    positions_fibre[index_nr * 3 + 2] = z + height; // Z-coordinate

                    velocities_fibre[index_nr * 3 + 0] = 0.0f; // X-velocity
                    velocities_fibre[index_nr * 3 + 1] = 0.0f;  // Y-velocity
                    velocities_fibre[index_nr * 3 + 2] = 0.0f;  // Z-velocity

                    ++index_nr;
                }
            }  
        }
    return {positions_fibre, velocities_fibre};
}

// Comparison function for sorting
bool compareVector(const std::vector<float>& a, const std::vector<float>& b) {
    if (a[0] != b[0]) return a[0] < b[0];
    if (a[1] != b[1]) return a[1] < b[1];
    return a[2] < b[2];
}

std::tuple<std::vector<float>, std::vector<float>, int> generateSTLFibre(std::vector<float>& positions_fibre, std::vector<float>& velocities_fibre, int n_part_fibre)
{
    std::cout << "\n\nRead STL data...\n\n";
    std::ifstream file("Fibre.stl");

    if (!file.is_open()) {
        std::cout << "\n\nFehler beim Öffnen der Datei!\n\n" << std::endl;
        return {positions_fibre, velocities_fibre,n_part_fibre};
    }

    std::vector<std::vector<float>> vertices;
    std::string line;
    int line_counter = 0;

    //Search for "vertex" / points in STL
    while (std::getline(file, line)) {
        ++ line_counter;
        if (line.find("vertex") != std::string::npos) {
            float x, y, z;
            std::sscanf(line.c_str(), " vertex %f %f %f", &x, &y, &z);
            vertices.push_back({x, y, z});
        }
    }
	
	//Delete dublicates
    std::sort(vertices.begin(), vertices.end(), compareVector);
    vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());

    int n_particles = vertices.size();
    n_part_fibre = n_particles;

    //convert from mm to m
    for (auto& vec : vertices) {
        vec[0] /= 1000.0f;
        vec[1] /= 1000.0f;
        vec[2] /= 1000.0f;
    }

    //Change size of fibre_Matrix to actually necessary fibre particles
    positions_fibre.resize(n_particles * 3);
    velocities_fibre.resize(n_particles * 3);

    // Save data into positions_fibre and velocities_fibre
    for (int i = 0; i < n_particles; ++i) {
        positions_fibre[i * 3 + 0] = vertices[i][0];
        positions_fibre[i * 3 + 1] = vertices[i][1];
        positions_fibre[i * 3 + 2] = vertices[i][2];

        velocities_fibre[i * 3 + 0] = 0.0f;
        velocities_fibre[i * 3 + 1] = 0.0f;
        velocities_fibre[i * 3 + 2] = 0.0f;
    }

    return {positions_fibre, velocities_fibre,n_part_fibre};
}
