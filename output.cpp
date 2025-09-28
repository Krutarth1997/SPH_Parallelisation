#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <cstdlib>
#include "output.h"

void saveParticleAsVTK(const std::string& path, float timestep, int n_particles, const Eigen::MatrixXf& positions, const Eigen::MatrixXf& velocities,
const Eigen::VectorXf& pressure, const Eigen::VectorXf& density, const Eigen::VectorXf& density_corr, const Eigen::MatrixXf& gradColorfield,
const Eigen::MatrixXf& forces, const Eigen::VectorXf& randpart, const Eigen::VectorXi& wandpart, const Eigen::VectorXf& adhesion)
{
    std::string filename = path + "particles_" + std::to_string(timestep) + ".vtk";

    //Öffnen der VTK-Datei zum Schreiben
    std::ofstream outputVTK(filename, std::ios::out);

    //Header-Datei erstellen
    outputVTK << "# vtk DataFile Version 2.0\nParticles\nASCII\nDATASET POLYDATA\nPOINTS ";
    
    //Partikelkoordinaten
    outputVTK << n_particles << " float\n";
    for (int i = 0; i < positions.rows(); ++i) {
        for (int j = 0; j < positions.cols(); ++j) {
            outputVTK << positions(i, j);
            if (j < positions.cols() - 1) {
                outputVTK << " ";
            }
        }
        outputVTK << std::endl; // Neue Zeile für jede Zeile der Matrix
    }

    //Vertices
    outputVTK << "VERTICES " << n_particles << " " << 2*n_particles << std::endl;
    for (int i=0; i<n_particles; i++)
    {
        outputVTK << "1 " << i << std::endl;
    }

    //Attribute
    outputVTK << "POINT_DATA " << n_particles << "\nFIELD attributes 15\n";

    //Testattribut
    outputVTK << "Particle 1 " << n_particles << " float" << std::endl;
    for (int i=0; i<n_particles; i++)
    {
        outputVTK << "1" << std::endl;
    }

    //Geschwindigkeit (x)
    outputVTK << "u1 1 " << n_particles << " float" << std::endl;
    for (int i = 0; i < velocities.rows(); ++i)
    {
        outputVTK << velocities(i, 0) << std::endl;
    }

    //Geschwindigkeit (y)
    outputVTK << "u2 1 " << n_particles << " float" << std::endl;
    for (int i = 0; i < velocities.rows(); ++i)
    {
        outputVTK << velocities(i, 1) << std::endl;
    }

    //Geschwindigkeit (z)
    outputVTK << "u3 1 " << n_particles << " float" << std::endl;
    for (int i = 0; i < velocities.rows(); ++i)
    {
        outputVTK << velocities(i, 2) << std::endl;
    }

    //Kraft (x)
    outputVTK << "F1 1 " << n_particles << " float" << std::endl;
    for (int i = 0; i < forces.rows(); ++i)
    {
        outputVTK << forces(i, 0) << std::endl;
    }

    //Kraft (y)
    outputVTK << "F3 1 " << n_particles << " float" << std::endl;
    for (int i = 0; i < forces.rows(); ++i)
    {
        outputVTK << forces(i, 2) << std::endl;
    }

    //Dichte
    outputVTK << "density 1 " << n_particles << " float" << std::endl;
    for (int i = 0; i < density.rows(); ++i)
    {
        outputVTK << density(i) << std::endl;
    }

    //Dichte (mit Randpartikel)
    outputVTK << "randpart 1 " << n_particles << " float" << std::endl;
    for (int i = 0; i < randpart.rows(); ++i)
    {
        outputVTK << randpart(i) << std::endl;
    }

    //Druck
    outputVTK << "pressure 1 " << n_particles << " float" << std::endl;
    for (int i = 0; i < pressure.rows(); ++i)
    {
        outputVTK << pressure(i) << std::endl;
    }

    //Density_corr
    outputVTK << "densities_corr 1 " << n_particles << " float" << std::endl;
    for (int i = 0; i < density_corr.rows(); ++i)
    {
        outputVTK << density_corr(i) << std::endl;
    }

    //GradColorfield (x)
    outputVTK << "gradColor(x) 1 " << n_particles << " float" << std::endl;
    for (int i = 0; i < gradColorfield.rows(); ++i)
    {
        outputVTK << gradColorfield(i, 0) << std::endl;
    }

    //GradColorfield (y)
    outputVTK << "gradColor(y) 1 " << n_particles << " float" << std::endl;
    for (int i = 0; i < gradColorfield.rows(); ++i)
    {
        outputVTK << gradColorfield(i, 1) << std::endl;
    }

    //GradColorfield (z)
    outputVTK << "gradColor(z) 1 " << n_particles << " float" << std::endl;
    for (int i = 0; i < gradColorfield.rows(); ++i)
    {
        outputVTK << gradColorfield(i, 2) << std::endl;
    }

    //Wandpartikel
    outputVTK << "wandpart 1 " << n_particles << " float" << std::endl;
    for (int i = 0; i < wandpart.rows(); ++i)
    {
	outputVTK << wandpart(i) << std::endl;
    }

    //Adhesion
    outputVTK << "adhesion 1 " << n_particles << " float" << std::endl;
    for (int i = 0; i < adhesion.rows(); ++i)
    {
        outputVTK << adhesion(i) << std::endl;
    }

    outputVTK << std::endl;
    outputVTK.close();
}

void saveFibreAsVTK(const std::string& path, float timestep, int n_particles, const Eigen::MatrixXf& positions)
{
    std::string filename = path + "fibre_" + std::to_string(timestep) + ".vtk";

    //Öffnen der VTK-Datei zum Schreiben
    std::ofstream outputVTK(filename, std::ios::out);

    //Header-Datei erstellen
    outputVTK << "# vtk DataFile Version 2.0\nFibre\nASCII\nDATASET POLYDATA\nPOINTS ";
    
    //Partikelkoordinaten
    outputVTK << n_particles << " float\n";
    for (int i = 0; i < positions.rows(); ++i) {
        for (int j = 0; j < positions.cols(); ++j) {
            outputVTK << positions(i, j);
            if (j < positions.cols() - 1) {
                outputVTK << " ";
            }
        }
        outputVTK << std::endl; // Neue Zeile für jede Zeile der Matrix
    }

    //Vertices
    outputVTK << "VERTICES " << n_particles << " " << 2*n_particles << std::endl;
    for (int i=0; i<n_particles; i++)
    {
        outputVTK << "1 " << i << std::endl;
    }

    //Attribute
    outputVTK << "POINT_DATA " << n_particles << "\nFIELD attributes 1\n";

    //Testattribut
    outputVTK << "Fibre 1 " << n_particles << " float" << std::endl;
    for (int i=0; i<n_particles; i++)
    {
        outputVTK << "1" << std::endl;
    }

    outputVTK << std::endl;
    outputVTK.close();
}

void progressBar(float iter, float timestep)
{
    float progress = float(iter) / float(timestep);
    int barWidth = 40;

    std::cout << "[";
    float pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << progress * 100.0 << " %\r";
    std::cout.flush();

    std::cout << std::endl;
}
