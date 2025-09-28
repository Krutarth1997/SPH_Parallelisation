#!/bin/bash

# Build (g++) and run SPH-Code
g++ -std=c++17 -fdiagnostics-color=always -pthread -g /scratchX/Hartwig/01_HyFilDrain/SPH-Code/SPH_3D_lin_MPI/*.cpp /scratchX/Hartwig/01_HyFilDrain/SPH-Code/SPH_3D_lin_MPI/*.h -o /scratchX/Hartwig/01_HyFilDrain/SPH-Code/SPH_3D_lin_MPI/runSPH -I/home/varchasvi/toolbox/eigen-3.4.0
./runSPH
