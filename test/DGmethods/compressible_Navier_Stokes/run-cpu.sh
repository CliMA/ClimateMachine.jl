#!/bin/bash

#SBATCH --job-name=Squall_2D
#SBATCH --output=out-squall.txt
#SBATCH --nodes=2                 # Number of nodes
#SBATCH --ntasks=64                # Number of MPI process
#SBATCH --time=0-03:00:00          # time (DD-HH:MM)
mpiexec /home/asridhar/julia-178d70318b/bin/julia --project=@. squall_line.jl
