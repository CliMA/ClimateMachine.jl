using CLIMAAtmosDycore
using MPI, Test

MPI.Init()

include("topology.jl")

MPI.Finalize()
