//---------------Include Bibliotheken---------------
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
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
Eigen::RowVector2f DOMAIN_X_LIM(-10.0e-6, 10.0e-6);
Eigen::RowVector2f DOMAIN_Y_LIM(-10.0e-6, 10.0e-6);
Eigen::RowVector2f DOMAIN_Z_LIM(-10.0e-6, 10.0e-6);
//------------Boundaries of calculation domain------


int main(int argc, char *argv[])
{
	//Set the name of the data folder
    std::string foldername = "Test_01";

    //Start of timekeeping
    auto start = std::chrono::high_resolution_clock::now();

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
	//----------------Specify the boundary type-------------

    //----------------Number of particles-------------------
    int n_particlesFibre = 5000; //choose a high number, because it gets reduced if the stl contains less Fibre-Particles
    int n_particles = 200; //With spherical particles: It creates a spherical droplet with n particles inside

    //----------------Matrix definitions--------------------
    Eigen::VectorXf densitiesFibre(n_particlesFibre);  
    densitiesFibre.setZero();

    Eigen::MatrixXf positions(n_particles, 3); //Particle Positions (X,Y,Z)
    Eigen::MatrixXf velocities(n_particles, 3); //Particle Velocities (u,w,v)

    Eigen::MatrixXf positionsFibre(n_particlesFibre, 3); //Fibre Positions (x,y,z)
    Eigen::MatrixXf velocitiesFibre(n_particlesFibre, 3); //Fibre Velocities (u,v,w: normaly = 0)
	//----------------Matrix definitions--------------------

	//Start iteration process from t=0....sim_time
	int iter = 0;
    int count = 1;

    while (CURR_TIME <= SIM_TIME)
    {
        //-------------Initialize Fluid Particles in first iteration-----------
        if (iter == 0)
        {
            if (boolBoxFlow) positions, velocities = generateBoxFlow(positions, velocities); //from your SPH-Neighbour-Search
            else if (boolVerticalFibreCoated) positions, velocities = generateVerticalFibreFluid(positions, velocities);
	        else if (boolSphericalParticles) positions, velocities = generateSphericalParticles(positions, velocities); //Is used here!
            else if (boolSemiSphericalParticles) positions, velocities = generateSemiSphericalParticles(positions, velocities);
            else if (boolWettedHorizontalFibre) positions, velocities = generateWettedFibreHorizontal(positions, velocities);
        }
		//-------------Initialize Fluid Particles in first iteration-----------
		
        //-------------Reinitialize Particles if boolAddparticles==true---------
        if ((iter != 0) && (iter % ADD_PARTICLES == 0) && (boolAddParticles))
        {  
            positions, velocities = generateNewParticlesDroplet(positions, velocities, N_NEW_PARTICLES);
            n_particles += N_NEW_PARTICLES;
        }
		//-------------Reinitialize Particles if boolAddparticles==true---------

        //-------------Initialize Geometry Particles---------------------------
        if (iter == 0)
        {
            if (boolPlane) positionsFibre, velocitiesFibre = generateFlatSurface(positionsFibre, velocitiesFibre);
            else if (boolVerticalFibre) positionsFibre, velocitiesFibre = generateVerticalFibre(positionsFibre, velocitiesFibre);
            else if (boolHorizontalFibre) positionsFibre, velocitiesFibre = generateHorizontalFibre(positionsFibre, velocitiesFibre);
            else if (boolImportSTL) positionsFibre, velocitiesFibre = generateSTLFibre(positionsFibre, velocitiesFibre, &n_particlesFibre); //Is used here
            std::cout << "\nGeometrie erstellt.\n";
        }
		//-------------Initialize Geometry Particles---------------------------

	//-----------------Just some displayed information in the terminal---------
	if (iter == 0)
	{   
	    std::cout << "\nFluidpartikel: " << n_particles;
	    std::cout << "\nFaserpartikel: " << n_particlesFibre;
	    std::cout << "\nInitiierter Timestep: " << TIME_STEP_LENGTH << " (Co <= " << COURANT << ")";
	    std::cout << "\nSimulationszeit: " << SIM_TIME << "\n\n";
	}
	//-----------------Just some displayed information in the terminal---------

        //------------Calculate physical Properties of geometry----------------
        if (iter == 0)
        {
            std::cout << "\nNachbarschaftspartikel + Abstandsvektoren erstellen \n";

            //Vorbereitung Nachbar-Suche
            //Nachbarschaftspartikel + Abstandsvektoren erstellen
            std::vector<std::vector<int>> neighbor_idsFibre(n_particlesFibre);
            std::vector<std::vector<float>> distancesFibre(n_particlesFibre);
	        Eigen::VectorXi wandpart2(n_particlesFibre); //eigentlich erst für Wandpartikel (Fluid-Faser-Interaktion)

            //Nachbarsuche Faser
            calculateNeighborhood(positionsFibre, positionsFibre, SMOOTHING_LENGTH, neighbor_idsFibre, distancesFibre);

            //Dichte eines Partikels anhand der Nachbarn berechnen
            calculateDensities(n_particlesFibre, neighbor_idsFibre, densitiesFibre, SMOOTHING_LENGTH_FIBRE, distancesFibre, PARTICLE_MASS_FIBRE, 0, n_particlesFibre);
        }
		//------------Calculate physical Properties of geometry----------------
		
		//------------Neighbour-Search for fluid particles---------------------
        std::vector<std::vector<int>> neighbor_ids(n_particles); //store neighbour ids
        std::vector<std::vector<float>> distances(n_particles); //store distance of neighbours
	    Eigen::VectorXi wandpart(n_particles); //checking for particles directly next to a wall
        
        calculateNeighborhood(positions, positions, SMOOTHING_LENGTH, neighbor_ids, distances); //neighboursearch with thread-parallelisation
		//------------Neighbour-Search for fluid particles---------------------

        //------------Neighbour-Search for geometry particles---------------------
        std::vector<std::vector<int>> neighbor_idsFluidFibre(n_particles);
        std::vector<std::vector<float>> distancesFluidFibre(n_particles);
        
        calculateNeighborhood(positions, positionsFibre, SMOOTHING_LENGTH_FIBRE, neighbor_idsFluidFibre, distancesFluidFibre);
		//------------Neighbour-Search for geometry particles---------------------

        //------------Calculate Particle density-------------------------
        Eigen::VectorXf densities = DensityKernelParticle(n_particles, neighbor_ids, distances, SMOOTHING_LENGTH, PARTICLE_MASS);
        
        float meandensity = densities.mean();
		//------------Calculate Particle density-------------------------

        //------------Calculate corrected Particle density---------------
        Eigen::VectorXf densities_corr = DensityCorrectedParticle(n_particles, neighbor_ids, distances, SMOOTHING_LENGTH, PARTICLE_MASS, densities);
        //------------Calculate corrected Particle density---------------

        //------------Check for boundary particles---------------
        Eigen::VectorXf randpart = CheckBoundaryParticle(n_particles, neighbor_idsFluidFibre,distancesFluidFibre, SMOOTHING_LENGTH_FIBRE, PARTICLE_MASS_FIBRE);
		//------------Check for boundary particles---------------

		//------------Initialize Additional matrix---------------
        Eigen::VectorXf pressures = ISOTROPIC_EXPONENT * (densities.array() - BASE_DENSITY); //calculation of fluid pressure
        Eigen::MatrixXf forces = Eigen::MatrixXf::Zero(n_particles, 3); //Initialise forces
        removeOwnElement(neighbor_ids, distances,0, n_particles); //remove own element from the neighbour list
        removeOwnElement(neighbor_idsFluidFibre, distancesFluidFibre,0, n_particles);
		//------------Initialize Additional matrix---------------

        //------------initialize Gradient of Colorfield (for surface tension)---------
        Eigen::MatrixXf gradColorField = ColorFieldKernel(n_particles, neighbor_ids, positions, distances, SMOOTHING_LENGTH, PARTICLE_MASS, densities);
		//------------initialize Gradient of Colorfield (for surface tension)---------

        //----------------------Calculate Forces between fluid particles---------------------------------------------------
        ForcesFluid(n_particles,neighbor_ids,distances,positions,pressures,densities,velocities,forces,SMOOTHING_LENGTH,PARTICLE_MASS,
                    viscosity_artificial,DYNAMIC_VISCOSITY,BASE_DENSITY,artificial_surface_tension_cohesion,artificial_surface_tension_curvature,
                    SURFACE_TENSION,density_gas,cw, CONSTANT_FORCE, velocity_gas, gradColorField, count);
		//----------------------Calculate Forces between fluid particles---------------------------------------------------
        
        //Temporary controll for adhesion
        Eigen::VectorXf adhesion(n_particles);
        adhesion.setZero();
		
		//Calculate Forces between fluid and geometry-----------------------------------------------    
        ForcesFluidGeometry(n_particles, neighbor_idsFluidFibre, distancesFluidFibre, positions, positionsFibre, velocities, velocitiesFibre, densities, 
                            forces, adhesion, SMOOTHING_LENGTH, SMOOTHING_LENGTH_FIBRE, PARTICLE_MASS, PARTICLE_MASS_FIBRE, viscosity_artificial,
                            DYNAMIC_VISCOSITY_FIBRE, BASE_DENSITY, artificial_surface_tension_adhesion, LJP_DISTANCE, LJP_P1, LJP_P2, LJP_COEF, count);

        //update particle velocities based on acting forces
        updateVelocities(velocities, forces, n_particles, TIME_STEP_LENGTH, 0, n_particles);

        //temporary: delete very high and very low movements-----------------
        deleteunneccessaryparticles(n_particles, velocities, forces);

        //update particle postions based on calculated velocities
        updatePositions(positions, velocities, n_particles, TIME_STEP_LENGTH, 0, n_particles); 

		//----------------------Save results for postprocessing-------------------------
        if (iter % PLOT_EVERY == 0)
        {
            std::cout << "vtk \n";
            //Save in .vtk style for postprocessing with ParaView -> change path
            std::string mainfolder = "D:/BUW/Thesis/Program/03_main/SPH-Parallelisation/data/";
            std::string folderending_part = "/particles/";
            std::string folderending_fibre = "/fibre/";
            std::cout << "particles \n";
            std::string vtkFilename = mainfolder + foldername + folderending_part;
            saveParticleAsVTK(vtkFilename, CURR_TIME, n_particles, positions, velocities, pressures, densities, densities_corr, gradColorField, forces, randpart, wandpart, adhesion);
            std::cout << "fibre \n";
            vtkFilename = mainfolder + foldername + folderending_fibre;
            saveFibreAsVTK(vtkFilename, CURR_TIME, n_particlesFibre, positionsFibre);
			
			//Progressbar in terminal for longer calculations
			progressBar(CURR_TIME, SIM_TIME);
			std::cout << "Time: " << CURR_TIME << " delta T: " << TIME_STEP_LENGTH << std::endl;
        }
		//----------------------Save results for postprocessing-------------------------
		
		//----------------------Timestep adjustments-----------------------
		iter += 1;
		CURR_TIME += TIME_STEP_LENGTH;
        count += 1;
		//----------------------Timestep adjustments-----------------------

        n_particles = removeParticle(velocities, positions, DOMAIN_X_LIM(0), DOMAIN_X_LIM(1), DOMAIN_Y_LIM(0), DOMAIN_Y_LIM(1), DOMAIN_Z_LIM(0), DOMAIN_Z_LIM(1)); //remove Particles outside the boundaries
    }
	
	//-----------------------------End of timekeeping and simulation----------------------
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\nTime taken by process: " << float(duration.count()) / 1000.0 << " seconds\n" << std::endl;
    std::cout << "\nfinal count" << count << std::endl;    
    std::cout << "\nENDE\n\n";
	//-----------------------------End of timekeeping and simulation----------------------
	
    return 0;
}
