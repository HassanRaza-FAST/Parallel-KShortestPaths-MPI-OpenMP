# Parallel-KShortestPaths

This project implements parallel k-shortest path algorithms using MPI and OpenMP. The aim is to enhance the computational efficiency of finding the k-shortest paths in a given graph.

## Overview
The project includes three implementations of the k-shortest paths algorithm:
1. Serial version
2. Parallel version using an adjacency list
3. Parallel version using an adjacency matrix

The graph data can be read from either a CSV file or a TXT file, and the implementation supports both directed and undirected graphs.

## Features

- **Parallel Computing**: Utilizes MPI and OpenMP to parallelize the k-shortest paths algorithm.
- **Graph Representation**: Supports various graph representations and input formats.
- **Scalability**: Designed to handle large graphs efficiently.
- **Performance Metrics**: Includes tools to measure and compare the performance of the parallel algorithms.

## Requirements

- C++ compiler (e.g., g++)
- MPI library (e.g., OpenMPI)
- OpenMP library

## Compilation

```sh
mpic++ -fopenmp -o k_shortest_paths k_shortest_paths.cpp
```

## Execution

```
mpirun -np <number_of_processes> ./k_shortest_paths
```
