//---------------Include Bibliotheken---------------
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>
#include <random>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <chrono>
#include <tuple>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
//---------------Include Bibliotheken---------------

//-------------Include external Header-------------
#include "output.h"
#include "initializeParticles.h"
#include "initializeFibre.h"
#include "matrixOperation.h"
#include "calculatePhysics.h"
#include "kernel.h"
#include "DynamicVariables.h"
//-------------Include externe Header--------------

//------------Physical Properties Fluid-----------

float PARTICLE_MASS = 1.8e-16;
float ISOTROPIC_EXPONENT = 1e-21;
float BASE_DENSITY = 1e3;
float SMOOTHING_LENGTH = 2.8e-6f;
float viscosity_artificial = 5e6; //artificial values for modelling of special forces
float artificial_surface_tension_curvature = 1;
float artificial_surface_tension_cohesion = 2;
float artificial_surface_tension_adhesion = 5e15;
float SURFACE_TENSION = 1.0;
float DYNAMIC_VISCOSITY = 1.5;
//------------Physical Properties Fluid-----------

//------------Physical Properties Fibre-----------
const float PARTICLE_MASS_FIBRE = 0.1e-16;
const float SMOOTHING_LENGTH_FIBRE = 1.5e-6f;
const float DYNAMIC_VISCOSITY_FIBRE = 1.0;
float LJP_DISTANCE = 1e-6; //LJP: Lennard-Jones-Potential (Force to prevent from particle diffusion)
const int LJP_P1 = 10;
const int LJP_P2 = 4;
float LJP_COEF = 1e-7;
const float contactAngleForce = 1.0f;
//------------Physical Properties Fibre-----------

//------------Timestep properties-----------------
const float SIM_TIME = 0.003f; //0.000025f; //Full simulation time
float TIME_STEP_LENGTH = 0.000005f; //time step length is dynamic (Courant)
const int PLOT_EVERY = 10; //Reduce the amount of saved Timesteps to reduce Memory usage
float CURR_TIME = 0.0f; //start time (normaly = 0.0s)
const float COURANT = 0.5f; //maximum Courant-Nr for adaptiv timestep length
//------------Timestep properties-----------------

//------------Additional Particles----------------
const int ADD_PARTICLES = 600; //Adds new particles every "ADD_PARTICLES" timesteps
int N_NEW_PARTICLES = 150; //amount of added Particles
bool boolAddParticles = false; //boolean to check if you want to add additional particles while simulation is running
//------------Additional Particles----------------


//------------Kernel normalization factors----------- (not all used)
const float NORMALIZATION_DENSITY = (315.0f * PARTICLE_MASS) / (64.0f * 3.14159265f * std::pow(SMOOTHING_LENGTH, 9));
const float NORMALIZATION_PRESSURE_FORCE = -(45.0f * PARTICLE_MASS) / (3.14159265f * std::pow(SMOOTHING_LENGTH, 6));
const float NORMALIZATION_VISCOUS_FORCE = (45.0f * DYNAMIC_VISCOSITY * PARTICLE_MASS) / (3.14159265f * std::pow(SMOOTHING_LENGTH, 6));
const float NORMALIZATION_POLY_KERNEL = 4 / (3.14159265f * std::pow(SMOOTHING_LENGTH, 8));
const float NORMALIZATION_SPIKY_KERNEL_DER3 = 4 / (3.14159265f * std::pow(SMOOTHING_LENGTH, 8));
const float NORM_KUB = 3 / (2 * 3.14159265f * pow(SMOOTHING_LENGTH,3));
const float NORM_KUB_DIF = -3 / (4 * 3.14159265f * pow(SMOOTHING_LENGTH,3));
const float NORM_POLY = 315.0f / (64.0f * 3.14159265f * std::pow(SMOOTHING_LENGTH, 9));
const float NORM_POLY_DIF2 = 945.0f / (8.0f * 3.14159265f * std::pow(SMOOTHING_LENGTH, 9));
const float NORM_SPIKY = 15 / (3.14159265f * std::pow(SMOOTHING_LENGTH, 6));
const float NORM_SPIKY_DIF = 45 / (3.14159265f * std::pow(SMOOTHING_LENGTH, 6));
const float NORM_SPIKY_DIF2 = 90 / (3.14159265f * std::pow(SMOOTHING_LENGTH, 6));
const float NORM_QUARTIC = 315 / (208 * 3.14159265f * std::pow(SMOOTHING_LENGTH, 3));
const float NORM_QUARTIC_DIF = 315 / (208 * 3.14159265f * std::pow(SMOOTHING_LENGTH, 4));
const float NORM_QUARTIC_DIF2 = 315 / (208 * 3.14159265f * std::pow(SMOOTHING_LENGTH, 5));
const float NORM_CUBIC_SPLINE = 8 / (3.14159265f * std::pow(SMOOTHING_LENGTH, 3));
//------------Kernel normalization factors-----------

//------------Boundaries of calculation domain------ (Particles outside the boundaries get deleted)
float DOMAIN_X_LIM[] = {-10.0e-6, 10.0e-6};
float DOMAIN_Y_LIM[] = {-10.0e-6, 10.0e-6};
float DOMAIN_Z_LIM[] = {-10.0e-6, 10.0e-6};
//------------Boundaries of calculation domain------

int main(int argc, char *argv[])
{
    // Check if a folder name was passed as a command-line argument
    std::string foldername = "Test_cuda_3"; // Default name

    // Base folder where data is stored
    std::string mainfolder = "D:/BUW/cuda/execution/";

    // Create the complete directory for particles and fibres
    std::string folderending_part = "/particles/";
    std::string folderending_fibre = "/fibre/";

    // Create the full path for particles and fibre folders
    std::string particleFolder = mainfolder + foldername + folderending_part;
    std::string fibreFolder = mainfolder + foldername + folderending_fibre;

    // Start timekeeping for results
    auto starttime = std::chrono::high_resolution_clock::now();

    //----------------Properties of surrounding gas flow----------------
    const float cw = 0.5; //Drag-Coefficient
    const float density_gas = 1.225;
    std::vector<float> velocity_gas = {0.1f, 0.0f, 0.0f}; // Velocity of surrounding gas (uniform velocity profile)
    //----------------Properties of surrounding gas flow----------------

    //Definition of a constant gravitational force
    std::vector<float> CONSTANT_FORCE = {0.0f, -1.09e-15f, 0.0f};//-1.79e-15 it's so small because it gets divided by mass

    //----------------Specify the flow type-----------------
    bool boolVerticalFibreCoated = false;
    bool boolBoxFlow = false;
    bool boolSphericalParticles = true; //-> Spherical particle droplet next to the fibre gets initialized
    bool boolSemiSphericalParticles = false;
    bool boolWettedHorizontalFibre = false;
    //----------------Specify the flow type-----------------

    //----------------Specify the boundary type-------------
    bool boolPlane = false;
    bool boolVerticalFibre = false;
    bool boolHorizontalFibre = false;
    bool boolImportSTL = true; //-> Import the Fibre as a STL data (Fibre.stl)
    bool boolmovementFibre = false; //If fibre have movements
    //----------------Specify the boundary type-------------

    //----------------Number of particles-------------------
    int n_particlesFibre = 5000; //choose a high number, because it gets reduced if the stl contains less Fibre-Particles
    int n_particles = 200; //With spherical particles: It creates a spherical droplet with n particles inside

    //----------------Matrix definitions--------------------
    // Host-side arrays to set up initial values for Particle position, velocity
    std::vector<float> positions(n_particles * 3);//Particle Positions (X,Y,Z)
    std::vector<float> velocities(n_particles * 3);//Particle Velocities (u,w,v)

    // Host-side arrays to set up initial values for Fibre position, velocity
    std::vector<float> positionsFibre(n_particlesFibre * 3);//Fibre Positions (X,Y,Z)
    std::vector<float> velocitiesFibre(n_particlesFibre * 3);//Fibre Velocities (u,w,v)
    //----------------Matrix definitions--------------------

    //Start iteration process from t=0....sim_time
    int iter = 0;

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    //-------------Initialize Fluid Particles in first iteration-----------
    if (iter == 0)
    {
        if (boolBoxFlow){std::tie(positions, velocities) = generateBoxFlow(positions, velocities);} //from your SPH-Neighbour-Search
        else if (boolVerticalFibreCoated){std::tie(positions, velocities) = generateVerticalFibreFluid(positions, velocities);}
        else if (boolSphericalParticles){std::tie(positions, velocities) = generateSphericalParticles(positions, velocities, n_particles);}//Is used here!
        else if (boolSemiSphericalParticles){std::tie(positions, velocities) = generateSemiSphericalParticles(positions, velocities, n_particles);}
        else if (boolWettedHorizontalFibre){std::tie(positions, velocities) = generateWettedFibreHorizontal(positions, velocities, n_particles);}
    }
    
    //-------------Initialize Geometry Particles-------------------
    if (iter == 0)
    {
        if (boolPlane) {std::tie(positionsFibre, velocitiesFibre) = generateFlatSurface(positionsFibre, velocitiesFibre);}
        else if (boolVerticalFibre) {std::tie(positionsFibre, velocitiesFibre) = generateVerticalFibre(positionsFibre, velocitiesFibre);}
        else if (boolHorizontalFibre) {std::tie(positionsFibre, velocitiesFibre) = generateHorizontalFibre(positionsFibre, velocitiesFibre);}
        else if (boolImportSTL) {std::tie(positionsFibre, velocitiesFibre,n_particlesFibre) = generateSTLFibre(positionsFibre, velocitiesFibre, n_particlesFibre);} //Is used here
        std::cout << "\nGeometrie erstellt.\n";
    }
    //-------------Initialize Geometry Particles-------------------

    //Initialize 
    int* d_n_particlesFibre; //should be pointer if it is updated on cpu again
    float* d_positions;
    float* d_velocities;
    float* d_positionsFibre;
    float* d_velocitiesFibre;

    // Allocate memory on the GPU
    cudaMalloc(&d_n_particlesFibre, sizeof(int));
    cudaMalloc(&d_positions, n_particles * 3 * sizeof(float));
    cudaMalloc(&d_velocities, n_particles * 3 * sizeof(float));
    cudaMalloc(&d_positionsFibre, n_particlesFibre * 3 * sizeof(float));
    cudaMalloc(&d_velocitiesFibre, n_particlesFibre * 3 * sizeof(float));

    // Copy the updated value of nparticlesFibre to the device
    cudaMemcpy(d_n_particlesFibre, &n_particlesFibre, sizeof(int), cudaMemcpyHostToDevice);

    // Copy positions and velocity to the device
    cudaMemcpy(d_positions, positions.data(), n_particles * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, velocities.data(), n_particles * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positionsFibre, positionsFibre.data(), n_particlesFibre * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocitiesFibre, velocitiesFibre.data(), n_particlesFibre * 3 * sizeof(float), cudaMemcpyHostToDevice);

    int count = 1;

    while (CURR_TIME <= SIM_TIME)
    {  
        //-------------Reinitialize Particles if boolAddparticles==true---------
        if ((iter != 0) && (iter % ADD_PARTICLES == 0) && (boolAddParticles))
        {  
            std::tie(positions, velocities) = generateNewParticlesDroplet(positions, velocities, n_particles, N_NEW_PARTICLES);
            n_particles += N_NEW_PARTICLES;
        }
    //-------------Reinitialize Particles if boolAddparticles==true---------

    //-----------------Displaying Initial information in the terminal---------
        if (iter == 0)
        {   
            std::cout << "\nFluidpartikel: " << n_particles;
            std::cout << "\nFaserpartikel: " << n_particlesFibre;
            std::cout << "\nInitiierter Timestep: " << TIME_STEP_LENGTH << " (Co <= " << COURANT << ")";
            std::cout << "\nSimulationszeit: " << SIM_TIME << "\n\n";
        }
    //-----------------Displaying Initial information in the terminal---------

    //-------------Initialize Block and Grid size--------
        int total_particles = n_particles + n_particlesFibre;
        int blockSize = 1024; // 256/32(32 threads per warp) = 8 warps
        int numBlocks = (total_particles + blockSize - 1) / blockSize;
        //size_t sharedMemSize = blockSize * (3 * sizeof(float) + sizeof(float)); // Positions + Densities

    //------------Calculate physical Properties of geometry----------------
        //Vorbereitung Nachbar-Suche
        //Nachbarschaftspartikel + Abstandsvektoren erstellen
        std::vector<int> neighbor_idsFibre(n_particlesFibre * n_particlesFibre);
        std::vector<float> distancesFibre(n_particlesFibre * n_particlesFibre);
        std::vector<int> wandpart2(n_particlesFibre); //eigentlich erst für Wandpartikel (Fluid-Faser-Interaktion)

        int* d_neighbor_idsFibre;
        float* d_distancesFibre;
        int* d_total_neighborsFibre;
        cudaMalloc(&d_total_neighborsFibre, n_particlesFibre * sizeof(int));
        cudaMalloc(&d_neighbor_idsFibre, n_particlesFibre * n_particlesFibre * sizeof(int));
        cudaMalloc(&d_distancesFibre, n_particlesFibre * n_particlesFibre * sizeof(float));
        
    //----Dichte eines Partikels anhand der Nachbarn berechnen------
        std::vector<float> densitiesFibre(n_particlesFibre,0.0f);
        float* d_densitiesFibre;
        cudaMalloc(&d_densitiesFibre, n_particlesFibre * sizeof(float));

        if (iter == 0 || boolmovementFibre)
        {
        //---------Nachbarsuche Faser-----------
            calculateNeighborhoodKernelfifi<<<numBlocks, blockSize>>>(d_positionsFibre, d_positionsFibre, d_neighbor_idsFibre, d_distancesFibre, d_n_particlesFibre, d_n_particlesFibre, SMOOTHING_LENGTH, d_total_neighborsFibre);
            
        //---------Density Faser-----------
            calculateDensities<<<numBlocks, blockSize>>>(d_neighbor_idsFibre, d_densitiesFibre, SMOOTHING_LENGTH_FIBRE,
                        d_distancesFibre, PARTICLE_MASS_FIBRE , d_n_particlesFibre, d_total_neighborsFibre);
    //------------Calculate physical Properties of geometry----------------
        }

    //------------Neighbour-Search for fluid particles---------------------
        std::vector<int> neighbor_ids(n_particles * n_particles);//store neighbour ids
        std::vector<float> distances(n_particles * n_particles);//store distance of neighbours
        std::vector<int> wandpart(n_particles); //checking for particles directly next to a wall

        int* d_neighbor_ids;
        float* d_distances;
        int* d_total_neighbors;
        cudaMalloc(&d_total_neighbors, n_particles * sizeof(int));
        cudaMalloc(&d_neighbor_ids, n_particles * n_particles * sizeof(int));
        cudaMalloc(&d_distances, n_particles * n_particles * sizeof(float));
        
        calculateNeighborhoodKernelff<<<numBlocks, blockSize>>>(d_positions, d_positions, d_neighbor_ids, d_distances, n_particles, n_particles, SMOOTHING_LENGTH, d_total_neighbors);
    //------------Neighbour-Search for fluid particles---------------------

    //------------Neighbour-Search for geometry particles---------------------
        std::vector<int> neighbor_idsFluidFibre(n_particles * n_particlesFibre);
        std::vector<float> distancesFluidFibre(n_particles * n_particlesFibre);

        int* d_neighbor_idsFluidFibre;
        float* d_distancesFluidFibre;
        int* d_total_neighborsFluidFibre;
        cudaMalloc(&d_total_neighborsFluidFibre, n_particles * sizeof(int));
        cudaMalloc(&d_neighbor_idsFluidFibre, n_particles * n_particlesFibre * sizeof(int));
        cudaMalloc(&d_distancesFluidFibre, n_particles * n_particlesFibre * sizeof(float));

        calculateNeighborhoodKernelffi<<<numBlocks, blockSize>>>(d_positions, d_positionsFibre, d_neighbor_idsFluidFibre, d_distancesFluidFibre, n_particles, d_n_particlesFibre, SMOOTHING_LENGTH_FIBRE, d_total_neighborsFluidFibre);
    //------------Neighbour-Search for geometry particles---------------------

    //------------Calculate Particle density-------------------------
        std::vector<float> densities(n_particles, 0.0f);
        float* d_densities;
        cudaMalloc(&d_densities, n_particles * sizeof(float));

        DensityKernelParticle<<<numBlocks, blockSize>>>(d_neighbor_ids, d_densities, SMOOTHING_LENGTH, d_distances, PARTICLE_MASS, n_particles, d_total_neighbors);
    //------------Calculate Particle density-------------------------

    //------------Calculate corrected Particle density---------------
        std::vector<float> densities_corr(n_particles, 0.0f);
        float* d_densities_corr;
        cudaMalloc(&d_densities_corr, n_particles * sizeof(float));

        DensityCorrectedParticle<<<numBlocks, blockSize>>>(d_neighbor_ids, d_densities,d_densities_corr, SMOOTHING_LENGTH, d_distances, PARTICLE_MASS, n_particles, d_total_neighbors);
    //------------Calculate corrected Particle density---------------

    //------------Check for boundary particles---------------
        std::vector<float> randpart(n_particles, 0.0f);
        float* d_randpart;
        cudaMalloc(&d_randpart, n_particles * sizeof(float));

        CheckBoundaryParticle<<<numBlocks, blockSize>>>(d_neighbor_idsFluidFibre, d_randpart, d_densities, SMOOTHING_LENGTH_FIBRE, d_distancesFluidFibre, PARTICLE_MASS_FIBRE, n_particles, d_n_particlesFibre, d_total_neighborsFluidFibre);
    //------------Check for boundary particles---------------

    //------------Calculaing Pressure---------------
        std::vector<float> pressures(n_particles, 0.0f);
        float* d_pressures;
        cudaMalloc(&d_pressures, n_particles * sizeof(float));
        
        ComputePressures<<<numBlocks, blockSize>>>(d_pressures, d_densities, ISOTROPIC_EXPONENT, BASE_DENSITY, n_particles);
        
    //------------Calculaing Pressure---------------

        cudaDeviceSynchronize();

    //------------initialize Gradient of Colorfield (for surface tension)---------
        std::vector<float> gradColorField(n_particles * 3, 0.0f);
        float* d_gradColorField;
        cudaMalloc(&d_gradColorField, n_particles * 3 * sizeof(float));

        ColorFieldKernel<<<numBlocks, blockSize>>>(d_neighbor_ids, d_gradColorField, d_positions, d_densities, SMOOTHING_LENGTH, d_distances, PARTICLE_MASS, n_particles, d_total_neighbors);
    //------------initialize Gradient of Colorfield (for surface tension)---------

    //----------------------Calculate Forces between fluid particles---------------------------------------------------
        std::vector<float> forces(n_particles * 3, 0.0f);
        float* d_forces;
        cudaMalloc(&d_forces, n_particles * 3 * sizeof(float));

        std::vector<float> adhesion(n_particles, 0.0f);
        float* d_adhesion;
        cudaMalloc(&d_adhesion, n_particles * sizeof(float));

        // Allocate and copy velocity_gas to GPU
        float* d_velocity_gas;
        cudaMalloc(&d_velocity_gas, 3 * sizeof(float));
        cudaMemcpy(d_velocity_gas, velocity_gas.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

        // Allocate and copy CONSTANT_FORCE to GPU
        float* d_CONSTANT_FORCE;
        cudaMalloc(&d_CONSTANT_FORCE, 3 * sizeof(float));
        cudaMemcpy(d_CONSTANT_FORCE, CONSTANT_FORCE.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();

        ForcesFluid<<<numBlocks, blockSize>>>( d_neighbor_ids, d_distances, d_positions, d_pressures, d_densities, d_velocities, d_forces, d_gradColorField, 
                                    SMOOTHING_LENGTH, PARTICLE_MASS, viscosity_artificial, DYNAMIC_VISCOSITY, BASE_DENSITY, artificial_surface_tension_cohesion,
                                    artificial_surface_tension_curvature, SURFACE_TENSION, density_gas, cw, d_CONSTANT_FORCE, d_velocity_gas, n_particles, d_total_neighbors,count);
    //----------------------Calculate Forces between fluid particles---------------------------------------------------

        cudaDeviceSynchronize();

    //Calculate Forces between fluid and geometry-----------------------------------------------    
        ForcesGeometry<<<numBlocks, blockSize>>>( d_neighbor_idsFluidFibre, d_distancesFluidFibre, d_positions, d_positionsFibre, 
                                        d_velocities, d_velocitiesFibre, d_densities, d_forces, d_adhesion, SMOOTHING_LENGTH, 
                                        SMOOTHING_LENGTH_FIBRE, PARTICLE_MASS, PARTICLE_MASS_FIBRE, viscosity_artificial,
                                        DYNAMIC_VISCOSITY_FIBRE, BASE_DENSITY, artificial_surface_tension_adhesion, LJP_DISTANCE, 
                                        LJP_P1, LJP_P2, LJP_COEF, d_total_neighborsFluidFibre, n_particles, d_n_particlesFibre, count);
    //Calculate Forces between fluid and geometry-----------------------------------------------

        cudaDeviceSynchronize();

        //update particle velocities based on acting forces
        updateVelocities<<<numBlocks, blockSize>>>(d_velocities, d_forces, TIME_STEP_LENGTH, n_particles);

        cudaDeviceSynchronize();

        //temporary: delete very high and very low movements-----------------
        deleteunneccessaryparticles<<<numBlocks, blockSize>>>( d_velocities, d_forces, n_particles);

        cudaDeviceSynchronize();   

        //update particle postions based on calculated velocities
        updatePositions<<<numBlocks, blockSize>>>(d_positions, d_velocities, TIME_STEP_LENGTH, n_particles);

        cudaDeviceSynchronize();   

        // Copy data back to host
        cudaMemcpy(densities.data(), d_densities, n_particles * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(densities_corr.data(), d_densities_corr, n_particles * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(randpart.data(), d_randpart, n_particles * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gradColorField.data(), d_gradColorField, n_particles * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(pressures.data(), d_pressures, n_particles * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(adhesion.data(), d_adhesion, n_particles * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(forces.data(), d_forces, n_particles * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(positions.data(), d_positions, n_particles * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(velocities.data(), d_velocities, n_particles * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        
        if ((count == 1) || (count == 300) || (count == 600)){
            std::cout << "Run " << count << " successful" << "\n";
            for (int i = 0; i < n_particles; ++i) {
                if (i == 0 || i == n_particles - 1) {
                    std::cout << "Position: " 
                            << positions[i * 3 + 0] << " "
                            << positions[i * 3 + 1] << " "
                            << positions[i * 3 + 2] << "\n";

                    std::cout << "Velocity: " 
                            << velocities[i * 3 + 0] << " "
                            << velocities[i * 3 + 1] << " "
                            << velocities[i * 3 + 2] << "\n";

                    std::cout << "Density: " << densities[i] << "\n";
                    std::cout << "RandPart: " << randpart[i] << "\n";
                    std::cout << "Pressure: " << pressures[i] << "\n";
                    std::cout << "Density Corr: " << densities_corr[i] << "\n";

                    std::cout << "Grad Color Field: " 
                            << gradColorField[i * 3 + 0] << " "
                            << gradColorField[i * 3 + 1] << " "
                            << gradColorField[i * 3 + 2] << "\n";

                    std::cout << "Forces: " 
                            << forces[i * 3 + 0] << " "
                            << forces[i * 3 + 1] << " "
                            << forces[i * 3 + 2] << "\n";

                    std::cout << "adhesion: " << adhesion[i] << "\n";
                }
            }
        }

        //----------------------Save results for postprocessing-------------------------
        if (iter % PLOT_EVERY == 0)
        {
            // Set vtk file paths and save VTK files
            std::cout << "Saving VTK data... ";
            std::cout << "particles, ";
            std::string vtkFilename = particleFolder; // Use the created particle folder
            saveParticleAsVTK(vtkFilename, CURR_TIME, n_particles, positions.data(), velocities.data(), densities.data(), randpart.data(), pressures.data(), densities_corr.data(), gradColorField.data(), forces.data(), wandpart.data(), adhesion.data());
            
            std::cout << "fibre \n";
            vtkFilename = fibreFolder; // Use the created fibre folder
            saveFibreAsVTK(vtkFilename, CURR_TIME, n_particlesFibre, positionsFibre.data());
            
            //Progressbar in terminal for longer calculations
            progressBar(CURR_TIME, SIM_TIME);
            std::cout << "Time: " << CURR_TIME << " delta T: " << TIME_STEP_LENGTH << std::endl;
        }
        //----------------------Save results for postprocessing-------------------------

        //----------------------Timestep adjustments-----------------------
        //TIME_STEP_LENGTH = calculateTimestepLength(velocities, forces, n_particles, SMOOTHING_LENGTH, COURANT); //calculate timestep length based on Courantnumber
        iter += 1;
        CURR_TIME += TIME_STEP_LENGTH;
        count += 1;
        //----------------------Timestep adjustments-----------------------

        cudaFree(d_total_neighborsFibre);
        cudaFree(d_distancesFibre);
        cudaFree(d_neighbor_idsFibre);
        cudaFree(d_densitiesFibre);
        cudaFree(d_total_neighbors);
        cudaFree(d_neighbor_ids);
        cudaFree(d_distances);
        cudaFree(d_total_neighborsFluidFibre);
        cudaFree(d_neighbor_idsFluidFibre);
        cudaFree(d_distancesFluidFibre);
        cudaFree(d_densities);
        cudaFree(d_densities_corr);
        cudaFree(d_randpart);
        cudaFree(d_gradColorField);
        cudaFree(d_pressures);
        cudaFree(d_forces);
        cudaFree(d_adhesion);
        cudaFree(d_velocity_gas);
        cudaFree(d_CONSTANT_FORCE);
    }
    
    //-----------------------------End of timekeeping and simulation----------------------
    auto finishtime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finishtime - starttime);
    std::cout<< "Time used for simulation: " << float(duration.count()) / 1000.0 << " seconds\n" << std::endl;
    std::cout << "ENDE mit final count: " << count << std::endl;

    //-----------------------------End of timekeeping and simulation----------------------

    // Free device memory
    cudaFree(d_n_particlesFibre);
    cudaFree(d_positions);
    cudaFree(d_velocities);
    cudaFree(d_positionsFibre);
    cudaFree(d_velocitiesFibre);

    // Destroy cuBLAS handle
    cublasDestroy(handle);

    std::cout << "\nEND\n\n";
    
    // Pause the program to keep the window open
    printf("Press Enter to exit...\n");
    getchar();
    return 0;
}
