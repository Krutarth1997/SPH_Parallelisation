//---------------Include Bibliotheken---------------
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <thread>
#include <chrono>
#include <mpi.h>
#include <filesystem>
#include <sstream>
#include <algorithm>
//---------------Include Bibliotheken---------------

//-------------Include external Header-------------
#include "output.h"
#include "initializeParticles.h"
#include "initializeFibre.h"
#include "matrixOperation.h"
#include "calculatePhysics.h"
#include "kernel.h"
#include "DynamicVariables.h"
#include "gather.h"
#include "updateTime.h"
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
const float SIM_TIME = 0.003f; //Full simulation time
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
Eigen::RowVector2f DOMAIN_X_LIM(-10.0e-4, 10.0e-4);
Eigen::RowVector2f DOMAIN_Y_LIM(-10.0e-4, 10.0e-4);
Eigen::RowVector2f DOMAIN_Z_LIM(-10.0e-4, 10.0e-4);
//------------Boundaries of calculation domain------


int main(int argc, char *argv[])
{
    //MPI Initialization
    MPI_Init(&argc, &argv);

    //current processor, total parallel processors, MPI communicator
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try{
        std::cout << "Process started with " << rank << " processor out of " << size << " total processesor" << std::endl;

        // //Set the name of the data folder
        // std::string foldername = "Test_MPI_1";

        // Check if a folder name was passed as a command-line argument
        std::string foldername = "Test_MPI"; // Default name
        if (argc > 1) {
            foldername = argv[1]; // Use the argument for folder name if provided
        }

        // Base folder where data is stored
        std::string mainfolder = "/home/m2130800/SPH-Parallelisation/SPH_3D_lin_MPI/execution/";

        // Create the complete directory for particles and fibres
        std::string folderending_part = "/particles/";
        std::string folderending_fibre = "/fibre/";

        // Create the full path for particles and fibre folders
        std::string particleFolder = mainfolder + foldername + folderending_part;
        std::string fibreFolder = mainfolder + foldername + folderending_fibre;

        // Create directories for particles and fibres if they don't exist
        if (rank == 0) {
            std::filesystem::create_directories(particleFolder); // Recursively create particle directory
            std::filesystem::create_directories(fibreFolder);    // Recursively create fibre directory
        }

        float starttime, finishtime; // Variables to track the time taken by the program

        // Closing the barrier for the timing
        MPI_Barrier(MPI_COMM_WORLD);
        starttime = MPI_Wtime();

        //----------------Properties of surrounding gas flow----------------
        const float cw = 0.5; //Drag-Coefficient
        const float density_gas = 1.225;
        Eigen::Matrix<float, 1, 3> velocity_gas; //Velocity of sorrounding gas (uniform velocity profile)
        velocity_gas << 0.1f, 0.0f, 0.0f;
        //----------------Properties of surrounding gas flow----------------

        //Definition of a constant gravitational force
        Eigen::Matrix<float, 1, 3> CONSTANT_FORCE;
        CONSTANT_FORCE << 0.0f, -1.09e-15, 0.0f; //-1.79e-15 it's so small because it gets divided by mass

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
        Eigen::MatrixXf positions(n_particles, 3); //Particle Positions (X,Y,Z)
        Eigen::MatrixXf velocities(n_particles, 3); //Particle Velocities (u,w,v)

        Eigen::MatrixXf positionsFibre(n_particlesFibre, 3); //Fibre Positions (x,y,z)
        Eigen::MatrixXf velocitiesFibre(n_particlesFibre, 3); //Fibre Velocities (u,v,w: normaly = 0)
        //----------------Matrix definitions--------------------

        //Start iteration process from t=0....sim_time
        int iter = 0;
        //Root Processor is 0
        //-------------Initialize Fluid Particles in first iteration at rank 0-----------
        if (rank == 0)
        {
            if (boolBoxFlow){ std::tie(positions, velocities) = generateBoxFlow(positions, velocities); } //from your SPH-Neighbour-Search
            else if (boolVerticalFibreCoated) { std::tie(positions, velocities) = generateVerticalFibreFluid(positions, velocities); }
            else if (boolSphericalParticles){std::tie(positions, velocities) = generateSphericalParticles(positions, velocities);}//Is used here!
            else if (boolSemiSphericalParticles){std::tie(positions, velocities) = generateSemiSphericalParticles(positions, velocities);}
            else if (boolWettedHorizontalFibre){std::tie(positions, velocities) = generateWettedFibreHorizontal(positions, velocities);}
        }
        //-------------Initialize Fluid Particles in first iteration at rank 0-----------
        
        //-------------Initialize Geometry Particles at rank 0-------------------
        if (rank == 0)
        {
            if (boolPlane){std::tie(positionsFibre, velocitiesFibre) = generateFlatSurface(positionsFibre, velocitiesFibre);} 
            else if (boolVerticalFibre){std::tie(positionsFibre, velocitiesFibre) = generateVerticalFibre(positionsFibre, velocitiesFibre);} 
            else if (boolHorizontalFibre){std::tie(positionsFibre, velocitiesFibre) = generateHorizontalFibre(positionsFibre, velocitiesFibre);}
            else if (boolImportSTL){std::tie(positionsFibre, velocitiesFibre) = generateSTLFibre(positionsFibre, velocitiesFibre, &n_particlesFibre);}//Is used here
            std::cout << "\nGeometrie erstellt.\n";
        }
        //-------------Initialize Geometry Particles at rank 0-------------------

        // Broadcast n_particles Fibre to all processes
        MPI_Bcast(&n_particlesFibre, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Resize matrices based on the new n_particlesFibre
        positionsFibre.conservativeResize(n_particlesFibre, 3);
        velocitiesFibre.conservativeResize(n_particlesFibre, 3);

        // Closing the barrier after Bcast n_particlesFibre and resize variables
        MPI_Barrier(MPI_COMM_WORLD);

        // Broadcast base variables of Fibre and fluid to all processes
        MPI_Bcast(positions.data(), n_particles * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(velocities.data(), n_particles * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(positionsFibre.data(), n_particlesFibre * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(velocitiesFibre.data(), n_particlesFibre * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

        std::cout<<"Bcast successful for rank "<<rank<<std::endl;

        int count = 1;

        while (CURR_TIME <= SIM_TIME)
        {  
            //-------------Reinitialize Particles if boolAddparticles==true---------
            if ((iter != 0) && (iter % ADD_PARTICLES == 0) && (boolAddParticles))
            {  
                std::tie(positions, velocities) = generateNewParticlesDroplet(positions, velocities, N_NEW_PARTICLES);
                n_particles += N_NEW_PARTICLES;
            }
            //-------------Reinitialize Particles if boolAddparticles==true---------

            //-------------Initialize start and end for each processor---------
            //For Fluid: start and end for each rank
            int blockSize = n_particles / size;
            int start = rank * blockSize;
            int end = (rank == size - 1) ? n_particles - 1 : start + blockSize - 1; //if last processor then total particles - 1

            //For Fibre: start and end for each rank
            int blockSizeFibre = n_particlesFibre / size;
            int startFibre = rank * blockSizeFibre;
            int endFibre = (rank == size - 1) ? n_particlesFibre - 1 : startFibre + blockSizeFibre - 1;
            //-------------Initialize start and end for each processor---------
            
            //-----------------Displaying Initial information in the terminal---------
            if ((iter == 0) && (rank == 0))
            {   
                std::cout << "\nFluidpartikel: " << n_particles;
                std::cout << "\nFaserpartikel: " << n_particlesFibre;
                std::cout << "\nInitiierter Timestep: " << TIME_STEP_LENGTH << " (Co <= " << COURANT << ")";
                std::cout << "\nSimulationszeit: " << SIM_TIME << "\n\n";
            }
            //-----------------Displaying Initial information in the terminal---------

            //------------Calculate physical Properties of geometry----------------
            //Vorbereitung Nachbar-Suche
            //Nachbarschaftspartikel + Abstandsvektoren erstellen
            std::vector<std::vector<int>> neighbor_idsFibre(n_particlesFibre);
            std::vector<std::vector<float>> distancesFibre(n_particlesFibre);
            Eigen::VectorXi local_wandpart2(endFibre - startFibre + 1); //eigentlich erst für Wandpartikel (Fluid-Faser-Interaktion)

            //----Dichte eines Partikels anhand der Nachbarn berechnen------
            Eigen::VectorXf local_densitiesFibre(endFibre - startFibre + 1);
            local_densitiesFibre.setZero();

            if (iter == 0 || boolmovementFibre)
            {
                //---------Nachbarsuche Faser-----------
                neighborhoodMPI(positionsFibre, positionsFibre, SMOOTHING_LENGTH, neighbor_idsFibre, distancesFibre, startFibre, endFibre);

                //---------Density Faser-----------
                calculateDensities(neighbor_idsFibre, local_densitiesFibre, SMOOTHING_LENGTH_FIBRE, distancesFibre, PARTICLE_MASS_FIBRE, startFibre, endFibre);
            }               
            //------------Calculate physical Properties of geometry----------------

            //------------Neighbour-Search for fluid particles---------------------
            std::vector<std::vector<int>> neighbor_ids(n_particles); //store neighbour ids
            std::vector<std::vector<float>> distances(n_particles); //store distance of neighbours
            Eigen::VectorXi local_wandpart(end - start + 1); //checking for particles directly next to a wall
            
            neighborhoodMPI(positions, positions, SMOOTHING_LENGTH, neighbor_ids, distances, start, end);
            //------------Neighbour-Search for fluid particles---------------------

            //------------Neighbour-Search for geometry particles---------------------
            std::vector<std::vector<int>> neighbor_idsFluidFibre(n_particles);
            std::vector<std::vector<float>> distancesFluidFibre(n_particles);

            neighborhoodMPI(positions, positionsFibre, SMOOTHING_LENGTH_FIBRE, neighbor_idsFluidFibre, distancesFluidFibre, start, end);
            //------------Neighbour-Search for geometry particles---------------------

            //------------Calculate Particle density-------------------------
            Eigen::VectorXf local_densities(end - start + 1);
            local_densities.setZero();

            DensityKernelParticle(neighbor_ids, local_densities, distances, SMOOTHING_LENGTH, PARTICLE_MASS, start, end);
            float meandensity = local_densities.mean();
            //------------Calculate Particle density-------------------------

            //------------Calculate corrected Particle density---------------
            Eigen::VectorXf local_densities_corr(end - start + 1);
            local_densities_corr.setZero();

            DensityCorrectedParticle(neighbor_ids, local_densities_corr, distances, SMOOTHING_LENGTH, PARTICLE_MASS, local_densities, start, end);
            //------------Calculate corrected Particle density---------------

            //------------Check for boundary particles---------------
            Eigen::VectorXf local_randpart(end - start + 1);
            local_randpart.setZero();

            CheckBoundaryParticle(neighbor_idsFluidFibre, local_randpart, local_densities, distancesFluidFibre, SMOOTHING_LENGTH_FIBRE, PARTICLE_MASS_FIBRE, start, end);
            //------------Check for boundary particles---------------

            //------------Initialize Additional matrix---------------
            Eigen::VectorXf local_pressures = ISOTROPIC_EXPONENT * (local_densities.array() - BASE_DENSITY); //calculation of fluid pressure
            
            Eigen::MatrixXf local_forces = Eigen::MatrixXf::Zero(end - start + 1, 3); //Initialise forces
            
            removeOwnElement(neighbor_ids, distances, start, end);
            removeOwnElement(neighbor_idsFluidFibre, distancesFluidFibre,  start, end);

            //------------Initialize Additional matrix---------------

            // Gather densities
            Eigen::VectorXf densities = gather_data_vector(local_densities, MPI_FLOAT, rank, size);

            // Gather pressures
            Eigen::VectorXf pressures = gather_data_vector(local_pressures, MPI_FLOAT, rank, size);

            //------------initialize Gradient of Colorfield (for surface tension)---------
            Eigen::MatrixXf local_gradColorField(end - start + 1, 3);
            local_gradColorField.setZero(); 

            ColorFieldKernel(neighbor_ids, local_gradColorField, positions, distances, SMOOTHING_LENGTH, PARTICLE_MASS, densities, start, end);
            //------------initialize Gradient of Colorfield (for surface tension)---------

            // Gather densities_corr
            Eigen::VectorXf densities_corr = gather_data_vector(local_densities_corr, MPI_FLOAT, rank, size);

            // Gather gradColorField
            Eigen::MatrixXf gradColorField = gather_data_matrix(local_gradColorField, MPI_FLOAT, rank, size);

            //----------------------Calculate Forces between fluid particles---------------------------------------------------
            ForcesFluid(neighbor_ids,distances,positions,pressures,densities,velocities,local_forces,SMOOTHING_LENGTH,PARTICLE_MASS,
                        viscosity_artificial,DYNAMIC_VISCOSITY,BASE_DENSITY,artificial_surface_tension_cohesion,artificial_surface_tension_curvature,
                        SURFACE_TENSION,density_gas,cw, CONSTANT_FORCE, velocity_gas, gradColorField, start, end);
            //----------------------Calculate Forces between fluid particles---------------------------------------------------
            
            //Temporary controll for adhesion
            Eigen::VectorXf local_adhesion(end - start + 1);
            local_adhesion.setZero();
            
            //Calculate Forces between fluid and geometry-----------------------------------------------    
            ForcesFluidGeometry(neighbor_idsFluidFibre, distancesFluidFibre, positions, positionsFibre, velocities, velocitiesFibre, densities, 
                                local_forces, local_adhesion, SMOOTHING_LENGTH, SMOOTHING_LENGTH_FIBRE, PARTICLE_MASS, PARTICLE_MASS_FIBRE, viscosity_artificial,
                                DYNAMIC_VISCOSITY_FIBRE, BASE_DENSITY, artificial_surface_tension_adhesion, LJP_DISTANCE, LJP_P1, LJP_P2, LJP_COEF, start, end);

            // Gather forces
            Eigen::MatrixXf forces = gather_data_matrix(local_forces, MPI_FLOAT, rank, size);

            // Gather randpart
            Eigen::VectorXf randpart = gather_data_vector(local_randpart, MPI_FLOAT, rank, size);

            // Gather wandpart
            Eigen::VectorXi wandpart = gather_data_vectorInt(local_wandpart, MPI_INT, rank, size);

            // Gather adhesion
            Eigen::VectorXf adhesion = gather_data_vector(local_adhesion, MPI_FLOAT, rank, size);

            // Local velocities for particles assigned to this process
            Eigen::MatrixXf local_positions(end - start + 1, 3);
            Eigen::MatrixXf local_velocities(end - start + 1, 3);

            //update particle velocities based on acting forces
            local_velocities = updateVelocities(velocities, forces, TIME_STEP_LENGTH, start, end);

            //temporary: delete very high and very low movements-----------------
            deleteunneccessaryparticles(local_velocities, forces, start, end);

            // Gather velocities
            velocities = gather_data_matrix(local_velocities, MPI_FLOAT, rank, size);

            //update particle postions based on calculated velocities
            local_positions = updatePositions(positions, velocities, TIME_STEP_LENGTH, start, end);

            // Gather positions
            positions = gather_data_matrix(local_positions, MPI_FLOAT, rank, size);

            if ((count == 1) || (count == 300) || (count == 600)){
                std::cout << "Run " << count << " successful for rank " << rank << "\n";
            }

            //----------------------Save results for postprocessing-------------------------
            if (rank == 0)
            {
                if (iter % PLOT_EVERY == 0)
                {
                    // Set vtk file paths and save VTK files
                    std::cout << "Saving VTK data... ";
                    std::cout << "particles, ";
                    std::string vtkFilename = particleFolder; // Use the created particle folder
                    saveParticleAsVTK(vtkFilename, CURR_TIME, n_particles, positions, velocities, pressures, densities, densities_corr, gradColorField, forces, randpart, wandpart, adhesion);
                    
                    std::cout << "fibre \n";
                    vtkFilename = fibreFolder; // Use the created fibre folder
                    saveFibreAsVTK(vtkFilename, CURR_TIME, n_particlesFibre, positionsFibre);
                    // std::cout << "vtk, ";
                    // //Save in .vtk style for postprocessing with ParaView -> change path
                    // std::string mainfolder = "D:/BUW/Thesis/Program/03_main/SPH-Parallelisation/data/";
                    // std::string folderending_part = "/particles/";
                    // std::string folderending_fibre = "/fibre/";
                    // std::cout << " particles, ";
                    // std::string vtkFilename = mainfolder + foldername + folderending_part;
                    // saveParticleAsVTK(vtkFilename, CURR_TIME, n_particles, positions, velocities, pressures, densities, densities_corr, gradColorField, forces, randpart, wandpart, adhesion);
                    // std::cout << " fibre \n";
                    // vtkFilename = mainfolder + foldername + folderending_fibre;
                    // saveFibreAsVTK(vtkFilename, CURR_TIME, n_particlesFibre, positionsFibre);
                    
                    //Progressbar in terminal for longer calculations
                    progressBar(CURR_TIME, SIM_TIME);
                    std::cout << "Time: " << CURR_TIME << " delta T: " << TIME_STEP_LENGTH << std::endl;
                }
            }
            //----------------------Save results for postprocessing-------------------------

            //----------------------Timestep adjustments-----------------------
            //TIME_STEP_LENGTH = calculateTimestepLength(velocities, forces, n_particles, SMOOTHING_LENGTH, COURANT); //calculate timestep length based on Courantnumber
            iter += 1;
            CURR_TIME += TIME_STEP_LENGTH;
            count += 1;
            //----------------------Timestep adjustments-----------------------

            n_particles = removeParticle(velocities, positions, DOMAIN_X_LIM(0), DOMAIN_X_LIM(1), DOMAIN_Y_LIM(0), DOMAIN_Y_LIM(1), DOMAIN_Z_LIM(0), DOMAIN_Z_LIM(1)); //remove Particles outside the boundaries
        }
        
        //-----------------------------End of timekeeping and simulation----------------------
        // Closing the barrier for the timing
        MPI_Barrier(MPI_COMM_WORLD);
        finishtime = MPI_Wtime();

        if (rank == 0) {
            double elapsed_time = finishtime - starttime;
	    printf("Time used by the CPU = %e seconds \n\n", elapsed_time);

        // Update the CSV file with nprocs and time
            updateTimeCSV(size, elapsed_time);
            std::cout << "ENDE mit final count: " << count << std::endl;
        }

        //-----------------------------End of timekeeping and simulation----------------------
        MPI_Finalize();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception on rank " << rank << ": " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception on rank " << rank << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
