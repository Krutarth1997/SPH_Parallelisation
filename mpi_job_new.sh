#!/bin/bash

# File with configurations
CONFIG_FILE="configurations.csv"
OUTPUT_DIR="./execution"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Load necessary modules
# module load 2021a GCC/10.3.0 OpenMPI/4.1.1
module load mpi/openmpi/4.1.0

# Read the CSV file and loop through each configuration
while IFS=',' read -r nodes ntasks_per_node
do
    # Skip the header line
    if [[ "$nodes" == "nodes" ]]; then
        continue
    fi

    # Create a unique folder name for this configuration
    unique_folder="Test_MPI_${nodes}_${ntasks_per_node}"

    # Replace the values in the SBATCH options with the current values
    sbatch --nodes="$nodes" --ntasks-per-node="$ntasks_per_node" <<EOL
#!/bin/bash
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=$ntasks_per_node
#SBATCH --partition=compute2011
#SBATCH --exclusive
#SBATCH --output=$OUTPUT_DIR/output_${nodes}_${ntasks_per_node}.out
#SBATCH --error=$OUTPUT_DIR/error_${nodes}_${ntasks_per_node}.err

module load mpi/openmpi/4.1.0
mpirun ./runSPH $unique_folder
EOL

done < "$CONFIG_FILE"

