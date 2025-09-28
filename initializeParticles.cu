#include <iostream>
#include <vector>
#include "initializeParticles.h"

std::pair<std::vector<float>, std::vector<float>> generateBoxFlow(std::vector<float>& positions, std::vector<float>& velocities)
{
    float offset_z = 7.0e-6;
    int n_height = 5; //6
    int n_rows = 5; //10
    int n_columns = 5; //10

    float SPAWN_X_LIM[] = {-1.5e-6,1.5e-6};
    float SPAWN_Y_LIM[] = {-1.5e-6,1.5e-6};
    float SPAWN_Z_LIM[] = {-1.5e-6,1.5e-6};

    int index_nr = 0;
    std::cout << "\nFluidpartikel initialisieren...";
    srand(time(0));

    for (int i = 0; i < n_rows; ++i)
    {
        for (int j = 0; j < n_columns; ++j)
        {
            for (int k = 0; k < n_height; ++k)
            {
                float x = SPAWN_X_LIM[0] + (SPAWN_X_LIM[1] - SPAWN_X_LIM[0]) * (i / static_cast<float>(n_rows - 1));
                float y = SPAWN_Y_LIM[0] + (SPAWN_Y_LIM[1] - SPAWN_Y_LIM[0]) * (j / static_cast<float>(n_columns - 1));
                float z = SPAWN_Z_LIM[0] + (SPAWN_Z_LIM[1] - SPAWN_Z_LIM[0]) * (k / static_cast<float>(n_height - 1));

                // Zufallszahl auf Zeitbasis generieren

                positions[index_nr * 3 + 0] = x + (-0.3e-6 + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.6e-6);
                positions[index_nr * 3 + 1] = y + (-0.3e-6 + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.6e-6);
                positions[index_nr * 3 + 2] = z + offset_z + (-0.3e-6 + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.6e-6);
                
                // Set initial velocities to zero
                velocities[index_nr * 3 + 0] = 0.0f;
                velocities[index_nr * 3 + 1] = 0.0f;
                velocities[index_nr * 3 + 2] = 0.0f;

                ++index_nr;
            }
        }  
    }

    return {positions, velocities};
}

std::pair<std::vector<float>, std::vector<float>> generateSphericalParticles(std::vector<float>& positions, std::vector<float>& velocities, int n_particles)
{
	//Position of generated droplet
    float pos_x = -5.0e-6;
    float pos_y = 1.5e-6;
    float pos_z = 0.0e-6;

    std::cout << "\nFluidpartikel initialisieren...";

    srand(time(0));

    int index_nr = 0;
	
	//Calculate droplet Radius based on number of particles and density
    float radius = pow((n_particles/3.75e19),0.333);

    while (index_nr < n_particles) {
        // Random coordinates within a cube (-1, -1, -1) to (1, 1, 1)
        float x = - radius + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius;
        float y = - radius + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius;
        float z = - radius + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius;
        // Check whether the coordinates lie within the spherical volume
        if (pow((x*x + y*y + z*z),0.5) <= radius)
        {
            positions[index_nr * 3 + 0] = x + pos_x; // X-coordinate
            positions[index_nr * 3 + 1] = y + pos_y; // Y-coordinate
            positions[index_nr * 3 + 2] = z + pos_z; // Z-coordinate

            velocities[index_nr * 3 + 0] = 0.01f; // X-velocity
            velocities[index_nr * 3 + 1] = 0.0f;  // Y-velocity
            velocities[index_nr * 3 + 2] = 0.0f;  // Z-velocity

            ++index_nr;
        }
    }
    
    return {positions, velocities};
}

std::pair<std::vector<float>, std::vector<float>> generateSemiSphericalParticles(std::vector<float>& positions, std::vector<float>& velocities, int n_particles)
{
    float pos_x = 0.0e-6;
    float pos_y = 0.0e-6;
    float pos_z = 0.5e-6;

    std::cout << "\nFluidpartikel initialisieren...";
    srand(time(0));

    int index_nr = 0;

    float radius = 2.5e-6;

    while (index_nr < n_particles) {
        // Zufällige Koordinaten innerhalb des Einheitswürfels (-1, -1, -1) bis (1, 1, 1)
        float x = - radius + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius;
        float y = - radius + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius;
        float z = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius;
        // Überprüfe, ob die Koordinaten innerhalb des kugelförmigen Volumens liegen
        if (pow((x*x + y*y + z*z),0.5) <= radius)
	    {

            positions[index_nr * 3 + 0] = x + pos_x; // X-coordinate
            positions[index_nr * 3 + 1] = y + pos_y; // Y-coordinate
            positions[index_nr * 3 + 2] = z + pos_z; // Z-coordinate

            velocities[index_nr * 3 + 0] = 0.0f; // X-velocity
            velocities[index_nr * 3 + 1] = 0.0f;  // Y-velocity
            velocities[index_nr * 3 + 2] = 0.0f;  // Z-velocity

            ++index_nr;
	    }
    }

    return {positions, velocities};
}

std::pair<std::vector<float>, std::vector<float>> generateVerticalFibreFluid(std::vector<float>& positions, std::vector<float>& velocities)
{
    //-----------Geometrieeinstellungen vertikale Faser (Fluid)----------
    int num_height = 20;
    const int num_radius = 4;
    int num_circumference[num_radius] = {38,42,46,50};
    float inner_radius = 9.0f;
    float outer_radius = 11.0f;
    float height = 12.0f;
    float offset_z = 5;
    float M_PI = 3.14f;
    //-----------Geometrieeinstellungen vertikale Faser (Fluid)----------


    std::cout << "\nFluidpartikel initialisieren...";
    int index_nr = 0;

    for (int i = 0; i < num_height; ++i) {
        for (int k = 0; k < num_radius; ++k) {
            for (int j = 0; j < num_circumference[k]; ++j) {
                float phi = 2* M_PI * (j / static_cast<float>(num_circumference[k]));
                float r = inner_radius + (outer_radius - inner_radius) * (k /static_cast<float>(num_radius - 1));
                float x = r * std::cos(phi);
                float y = r * std::sin(phi);
                float z = height * (i / static_cast<float>(num_height - 1));

                positions[index_nr * 3 + 0] = x; // X-coordinate
                positions[index_nr * 3 + 1] = y; // Y-coordinate
                positions[index_nr * 3 + 2] = z + offset_z; // Z-coordinate

                velocities[index_nr * 3 + 0] = 0.0f; // X-velocity
                velocities[index_nr * 3 + 1] = 0.0f;  // Y-velocity
                velocities[index_nr * 3 + 2] = 0.0f;  // Z-velocity

                ++index_nr;
            }
        }
    }

    return {positions, velocities};
}

std::pair<std::vector<float>, std::vector<float>> generateNewParticlesDroplet(std::vector<float>& positions, std::vector<float>& velocities, int n_particles, int n_new_particles)
{
    std::vector<float> new_positions(n_new_particles * 3);
    
    float pos_x = 0.0e-6;
    float pos_y = 7.0e-6;
    float pos_z = 5.0e-6;

    std::cout << "\nNeu Fluidpartikel initialisieren...";

    srand(time(0));

    int index_nr = 0;
    float radius = pow((n_new_particles/3.75e19),0.333);

    std::cout << "\nRadius: " << radius << std::endl;

    while (index_nr < n_new_particles) {

        // Zufällige Koordinaten innerhalb des Einheitswürfels (-1, -1, -1) bis (1, 1, 1)
        float x = - radius + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius;
        float y = - radius + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius;
        float z = - radius + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius;

        // Überprüfe, ob die Koordinaten innerhalb des kugelförmigen Volumens liegen
        if (pow((x*x + y*y + z*z),0.5) <= radius)
        {
            new_positions[index_nr * 3 + 0] = x + pos_x; // X-coordinate
            new_positions[index_nr * 3 + 1] = y + pos_x; // Y-coordinate
            new_positions[index_nr * 3 + 2] = z + pos_z; // Z-coordinate

            ++index_nr;
        }
    }

    int new_rows = n_particles + n_new_particles;
    std::vector<float> new_pos_vector(new_rows * 3);

    for (int i = 0; i < n_particles; ++i) {
        for (int j = 0; j < 3; ++j) {
            new_pos_vector[i * 3 + j] = positions[i * 3 + j];
        }
    }

    for (int i = 0; i < n_new_particles; ++i) {
        for (int j = 0; j < 3; ++j) {
            new_pos_vector[(n_particles + i) * 3 + j] = new_positions[i * 3 + j];
        }
    }

    positions = new_pos_vector;

    std::vector<float> new_velocities(n_new_particles * 3);

    for (int i = 0; i < n_new_particles; ++i) {
        new_velocities[i * 3 + 0] = 0.01f; // First column constant
        new_velocities[i * 3 + 1] = 0.0f;  // Second column zero
        new_velocities[i * 3 + 2] = 0.0f;  // Third column zero
    }

    std::vector<float> new_vel_vector(new_rows * 3);

    for (int i = 0; i < n_particles; ++i) {
        for (int j = 0; j < 3; ++j) {
            new_vel_vector[i * 3 + j] = velocities[i * 3 + j];
        }
    }

    for (int i = 0; i < n_new_particles; ++i) {
        for (int j = 0; j < 3; ++j) {
            new_vel_vector[(n_particles + i) * 3 + j] = new_velocities[i * 3 + j];
        }
    }

    velocities = new_vel_vector;

    return {positions, velocities};
}

std::pair<std::vector<float>, std::vector<float>> generateWettedFibreHorizontal(std::vector<float>& positions, std::vector<float>& velocities, int n_particles)
{
    float pos_x = -3.0e-6;
    float pos_y = 0.0e-6;
    float pos_z = 0.0e-6;

    std::cout << "\nFluidpartikel initialisieren...";
    srand(time(0));

    int index_nr = 0;

    float radius_Droplet = 2.0e-6;
    float radius_Fibre = 0.3e-6;

    while (index_nr < n_particles) {
        // Zufällige Koordinaten innerhalb des Einheitswürfels (-1, -1, -1) bis (1, 1, 1)
        float x = - radius_Droplet + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius_Droplet;
        float y = - radius_Droplet + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius_Droplet;
        float z = - radius_Droplet + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius_Droplet;
        // Überprüfe, ob die Koordinaten innerhalb des kugelförmigen Volumens liegen
        if ((pow((x*x + y*y + z*z),0.5) <= radius_Droplet) && (pow((y*y + z*z),0.5)>=(radius_Fibre + 0.3e-6)))
        {
            positions[index_nr * 3 + 0] = x + pos_x; // X-coordinate
            positions[index_nr * 3 + 1] = y + pos_y; // Y-coordinate
            positions[index_nr * 3 + 2] = z + pos_z; // Z-coordinate

            velocities[index_nr * 3 + 0] = 0.0f; // X-velocity
            velocities[index_nr * 3 + 1] = 0.0f;  // Y-velocity
            velocities[index_nr * 3 + 2] = 0.0f;  // Z-velocity

            ++index_nr;
        }
    }

    return {positions, velocities};
}
