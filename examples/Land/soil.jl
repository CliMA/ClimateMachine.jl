# import necessary 

using MPI
using CLIMA
using Logging
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using LinearAlgebra
using CLIMA.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps
using CLIMA.ODESolvers

include("soilmodel.jl")
MPI.Initialized() || MPI.Init()
mpicomm = MPI.COMM_WORLD

# README: how to get it running
# simple tests


# create function 
#  - topology/grid
#  - conductivity
#  - boundary values
#  

# set up domain
topl = StackedBrickTopology(mpicomm, (0:1,0:-1:-10); periodicity = (true,false),boundary=((0,0),(1,2)))
grid = DiscontinuousSpectralElementGrid(topl, FloatType = Float64, DeviceArray = Array, polynomialorder = 5)

m = SoilModel(1.0,1.0,1.0,20.0,10.0)

# Set up DG scheme
dg = DGModel( # 
  m, # "PDE part"
  grid,
  CentralNumericalFluxNonDiffusive(), # penalty terms for discretizations
  CentralNumericalFluxDiffusive(),
  CentralGradPenalty())


Δ = min_node_distance(grid)
CFL_bound = (Δ^2 / (2m.λ/(m.ρ*m.c)))
dt = CFL_bound/2 # TODO: provide a "default" timestep based on  Δx,Δy,Δz

# state variable
Q = init_ode_state(dg, Float64(0))

# initialize ODE solver
lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

# run model
solve!(Q, lsrk; timeend=dt)

# TODO: extract state for plot
