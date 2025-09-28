#!/bin/bash

# File with configurations
CONFIG_FILE="configurations.csv"

# Load necessary modules
module load mpi/openmpi/4.1.0

# Read the CSV file and loop through each configuration
while IFS=',' read -r nodes ntasks_per_node
do
    # Skip the header line
    if [[ "$nodes" == "nodes" ]]; then
        continue
    fi

    # Replace the values in the SBATCH options with the current values
    sbatch --nodes="$nodes" --ntasks-per-node="$ntasks_per_node" <<EOL
#!/bin/bash
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=$ntasks_per_node
#SBATCH --partition=compute2011
#SBATCH --exclusive
#SBATCH --output=./output_${nodes}_${ntasks_per_node}.out
#SBATCH --error=./error_${nodes}_${ntasks_per_node}.err

module load mpi/openmpi/4.1.0
mpirun ./runSPH
EOL

done < "$CONFIG_FILE"

