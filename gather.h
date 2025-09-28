#ifndef GATHER_H
#define GATHER_H

#include <mpi.h>
#include <Eigen/Dense>
#include <vector>
#include <iostream>

Eigen::VectorXf gather_data_vector(const Eigen::VectorXf& local_data, MPI_Datatype mpi_type, int rank, int size);
Eigen::VectorXi gather_data_vectorInt(const Eigen::VectorXi& local_data, MPI_Datatype mpi_type, int rank, int size);
Eigen::MatrixXf gather_data_matrix(const Eigen::MatrixXf& local_data, MPI_Datatype mpi_type, int rank, int size);

#endif // GATHER_H
