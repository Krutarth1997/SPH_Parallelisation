#include "gather.h"

Eigen::VectorXf gather_data_vector(const Eigen::VectorXf& local_data, MPI_Datatype mpi_type, int rank, int size) {
    std::vector<int> particle_sizes(size);
    std::vector<int> particle_displs(size);

    int local_size = local_data.size();

    // Gather sizes of local data from all processes
    MPI_Allgather(&local_size, 1, MPI_INT, particle_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate displacements based on sizes
    particle_displs[0] = 0;
    for (int i = 1; i < size; i++) {
        particle_displs[i] = particle_displs[i - 1] + particle_sizes[i - 1];
    }

    int total_size = particle_displs[size - 1] + particle_sizes[size - 1];

    // Create a global data vector with the correct total size
    Eigen::VectorXf global_data(total_size);

    // Gather all local data into global data
    MPI_Allgatherv(local_data.data(), local_size, mpi_type,
                   global_data.data(), particle_sizes.data(), particle_displs.data(), mpi_type, MPI_COMM_WORLD);

    // Return the gathered global data
    return global_data;
}

Eigen::VectorXi gather_data_vectorInt(const Eigen::VectorXi& local_data, MPI_Datatype mpi_type, int rank, int size) {
    std::vector<int> particle_sizes(size);
    std::vector<int> particle_displs(size);

    int local_size = local_data.size();

    // Gather sizes of local data from all processes
    MPI_Allgather(&local_size, 1, MPI_INT, particle_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate displacements based on sizes
    particle_displs[0] = 0;
    for (int i = 1; i < size; i++) {
        particle_displs[i] = particle_displs[i - 1] + particle_sizes[i - 1];
    }

    int total_size = particle_displs[size - 1] + particle_sizes[size - 1];

    // Create a global data vector with the correct total size
    Eigen::VectorXi global_data(total_size);

    // Gather all local data into global data
    MPI_Allgatherv(local_data.data(), local_size, mpi_type,
                   global_data.data(), particle_sizes.data(), particle_displs.data(), mpi_type, MPI_COMM_WORLD);

    // Return the gathered global data
    return global_data;
}

Eigen::MatrixXf gather_data_matrix(const Eigen::MatrixXf& local_data, MPI_Datatype mpi_type, int rank, int size) {
    // Transpose the local row-major matrix to a column-major format
    Eigen::MatrixXf transposed_local_data = local_data.transpose();

    std::vector<int> particle_sizes(size);
    std::vector<int> particle_displs(size);

    int local_rows = transposed_local_data.cols(); // Transposed, so cols represent original rows
    int cols = transposed_local_data.rows();

    // Gather sizes of local data from all processes
    MPI_Allgather(&local_rows, 1, MPI_INT, particle_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate displacements based on sizes
    particle_displs[0] = 0;
    for (int i = 1; i < size; i++) {
        particle_displs[i] = particle_displs[i - 1] + particle_sizes[i - 1];
    }

    // Multiply displacements by the number of rows to account for the transposed matrix
    for (int i = 0; i < size; ++i) {
        particle_displs[i] *= cols;
        particle_sizes[i] *= cols;
    }

    int total_rows = particle_displs[size - 1] + particle_sizes[size - 1];

    // Resize global_transposed_data to total_rows / cols by cols (back to row-major)
    Eigen::MatrixXf global_transposed_data(cols, total_rows / cols);

    // Gather all local data into global data (now in row-major format after transposing back)
    int ierr = MPI_Allgatherv(transposed_local_data.data(), local_rows * cols, mpi_type,
                              global_transposed_data.data(), particle_sizes.data(), particle_displs.data(), mpi_type, MPI_COMM_WORLD);

    if (ierr != MPI_SUCCESS) {
        std::cerr << "MPI_Allgatherv failed with error code " << ierr << std::endl;
    }

    // Transpose back to the original row-major format and return
    return global_transposed_data.transpose();
}
