#include <Eigen/Dense>
#include <iostream>
#include "initializeParticles.h"

Eigen::MatrixXf generateBoxFlow(Eigen::MatrixXf& positions, Eigen::MatrixXf& velocities)
{
    float offset_z = 7.0e-6;
    int n_height = 5; //6
    int n_rows = 5; //10
    int n_columns = 5; //10
    Eigen::RowVector2f SPAWN_X_LIM(-1.5e-6,1.5e-6);
    Eigen::RowVector2f SPAWN_Y_LIM(-1.5e-6,1.5e-6);
    Eigen::RowVector2f SPAWN_Z_LIM(-1.5e-6,1.5e-6);

    int index_nr = 0;
    std::cout << "\nFluidpartikel initialisieren...";
    srand(time(0));

        for (int i = 0; i < n_rows; ++i)
        {
            for (int j = 0; j < n_columns; ++j)
            {
                for (int k = 0; k < n_height; ++k)
                {
                    float x = SPAWN_X_LIM(0) + (SPAWN_X_LIM(1) - SPAWN_X_LIM(0)) * (i / static_cast<float>(n_rows - 1));
                    float y = SPAWN_Y_LIM(0) + (SPAWN_Y_LIM(1) - SPAWN_Y_LIM(0)) * (j / static_cast<float>(n_columns - 1));
                    float z = SPAWN_Z_LIM(0) + (SPAWN_Z_LIM(1) - SPAWN_Z_LIM(0)) * (k / static_cast<float>(n_height - 1));

                    // Zufallszahl auf Zeitbasis generieren
                    float randomValue = -0.3e-6 + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.6e-6;
                    positions(index_nr, 0) = x + randomValue;
                    randomValue = -0.3e-6 + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.6e-6;
                    positions(index_nr, 1) = y + randomValue;
                    randomValue = -0.3e-6 + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.6e-6;
                    positions(index_nr, 2) = z + offset_z + randomValue;
                    velocities.row(index_nr) << 0.0f, 0.0f, 0.0f;

                    ++index_nr;
                }
            }  
        }

    return positions, velocities;
}

Eigen::MatrixXf generateSphericalParticles(Eigen::MatrixXf& positions, Eigen::MatrixXf& velocities)
{
	//Position of generated droplet
    float pos_x = -5.0e-6;
    float pos_y = 1.5e-6;
    float pos_z = 0.0e-6;

    std::cout << "\nFluidpartikel initialisieren...";
    srand(time(0));

    int numParticles = positions.rows();
    int index_nr = 0;
	
	//Calculate droplet Radius based on number of particles and density
    float radius = pow((numParticles/3.75e19),0.333);

    while (index_nr < numParticles) {
        // Random coordinates within a cube (-1, -1, -1) to (1, 1, 1)
        float x = - radius + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius;
        float y = - radius + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius;
        float z = - radius + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius;
        // Check whether the coordinates lie within the spherical volume
        if (pow((x*x + y*y + z*z),0.5) <= radius)
        {
            positions.row(index_nr) << x + pos_x, y + pos_y, z + pos_z;
            velocities.row(index_nr) << 0.01f, 0.0f, 0.0f;
            ++index_nr;
        }
    }

    return positions, velocities;
}

Eigen::MatrixXf generateSemiSphericalParticles(Eigen::MatrixXf& positions, Eigen::MatrixXf& velocities)
{
    float pos_x = 0.0e-6;
    float pos_y = 0.0e-6;
    float pos_z = 0.5e-6;

    std::cout << "\nFluidpartikel initialisieren...";
    srand(time(0));

    int numParticles = positions.rows();
    int index_nr = 0;

    float radius = 2.5e-6;

    while (index_nr < numParticles) {
        // Zufällige Koordinaten innerhalb des Einheitswürfels (-1, -1, -1) bis (1, 1, 1)
        float x = - radius + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius;
        float y = - radius + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius;
        float z = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius;
        // Überprüfe, ob die Koordinaten innerhalb des kugelförmigen Volumens liegen
        if (pow((x*x + y*y + z*z),0.5) <= radius)
	    {
        positions.row(index_nr) << x + pos_x, y + pos_y, z + pos_z;
	    velocities.row(index_nr) << 0.0, 0.0, 0.0;
        ++index_nr;
	    }
    }

    return positions, velocities;
}

Eigen::MatrixXf generateVerticalFibreFluid(Eigen::MatrixXf& positions, Eigen::MatrixXf& velocities)
{
    //-----------Geometrieeinstellungen vertikale Faser (Fluid)----------
    int num_height = 20;
    int num_radius = 4;
    int num_circumference[num_radius] = {38,42,46,50};
    float inner_radius = 9.0f;
    float outer_radius = 11.0f;
    float height = 12.0f;
    float offset_z = 5;
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

                positions(index_nr, 0) = x;
                positions(index_nr, 1) = y;
                positions(index_nr, 2) = z + offset_z;
                velocities.row(index_nr) << 0.0f, 0.0f, 0.0f;

                ++index_nr;
            }
        }
    }

    return positions, velocities;
}

Eigen::MatrixXf generateNewParticlesDroplet(Eigen::MatrixXf& positions, Eigen::MatrixXf& velocities, int n_new_particles)
{
    Eigen::MatrixXf new_positions(n_new_particles, 3);
    
    float pos_x = 0.0e-6;
    float pos_y = 7.0e-6;
    float pos_z = 5.0e-6;

    std::cout << "\nFluidpartikel initialisieren...\n";
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
            new_positions.row(index_nr) << x + pos_x, y + pos_y, z + pos_z;
            ++index_nr;
        }
    }

    int new_rows = positions.rows() + new_positions.rows();
    Eigen::MatrixXf new_pos_matrix(new_rows, positions.cols());
    new_pos_matrix.block(0, 0, positions.rows(), positions.cols()) = positions;
    new_pos_matrix.block(positions.rows(), 0, new_positions.rows(), new_positions.cols()) = new_positions;
    positions = new_pos_matrix;

    Eigen::MatrixXf new_velocities(n_new_particles, 3);
    new_velocities = Eigen::MatrixXf::Constant(n_new_particles, 3, 0.0f);
    new_velocities.col(0).setConstant(0.01f);
    new_velocities.col(1).setZero();
    new_velocities.col(2).setZero();

    Eigen::MatrixXf new_vel_matrix(new_rows, velocities.cols());
    new_vel_matrix.block(0, 0, velocities.rows(), velocities.cols()) = velocities;
    new_vel_matrix.block(velocities.rows(), 0, new_velocities.rows(), new_velocities.cols()) = new_velocities;
    velocities = new_vel_matrix;

    return positions, velocities;
}

Eigen::MatrixXf generateWettedFibreHorizontal(Eigen::MatrixXf& positions, Eigen::MatrixXf& velocities)
{
    float pos_x = -3.0e-6;
    float pos_y = 0.0e-6;
    float pos_z = 0.0e-6;

    std::cout << "\nFluidpartikel initialisieren...";
    srand(time(0));

    int numParticles = positions.rows();
    int index_nr = 0;

    float radius_Droplet = 2.0e-6;
    float radius_Fibre = 0.3e-6;

    while (index_nr < numParticles) {
        // Zufällige Koordinaten innerhalb des Einheitswürfels (-1, -1, -1) bis (1, 1, 1)
        float x = - radius_Droplet + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius_Droplet;
        float y = - radius_Droplet + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius_Droplet;
        float z = - radius_Droplet + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 * radius_Droplet;
        // Überprüfe, ob die Koordinaten innerhalb des kugelförmigen Volumens liegen
        if ((pow((x*x + y*y + z*z),0.5) <= radius_Droplet) && (pow((y*y + z*z),0.5)>=(radius_Fibre + 0.3e-6)))
        {
            positions.row(index_nr) << x + pos_x, y + pos_y, z + pos_z;
            velocities.row(index_nr) << 0.0, 0.0, 0.0;
            ++index_nr;
        }
    }

    return positions, velocities;
}
