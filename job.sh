#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=5
#SBATCH --partition=compute2011
#SBATCH --exclusive
#SBATCH --output=./output.out
#SBATCH --error=./error.err

module load mpi/openmpi/4.1.0
mpirun ./runSPH
