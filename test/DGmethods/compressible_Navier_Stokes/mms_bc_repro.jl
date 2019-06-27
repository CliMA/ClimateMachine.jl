using MPI
using CLIMA
using CLIMA.Topologies
using CLIMA.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using LinearAlgebra
using StaticArrays

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayType = CuArray
else
  const ArrayType = Array
end

include("mms_solution_generated.jl")
include("mms_model.jl")

warpfun = (x, y, z) -> begin
          (x + (x-1/2)*cos(2*π*y*z)/4,
           y + exp(sin(2π*(x*y+z)))/20,
          z + x/4 + y^2/2 + sin(x*y*z))
end



l = 1
DFloat = Float64
Ne = (4, 4)

brickrange = (range(0.0; length=Ne[1]+1, stop=1),
              range(0.0; length=Ne[2]+1, stop=1),
              range(0.0; length=Ne[2]+1, stop=1))


MPI.Initialized() || MPI.Init()
mpicomm = MPI.COMM_WORLD
topl = BrickTopology(mpicomm, brickrange,  periodicity = (false, false, false))

grid = DiscontinuousSpectralElementGrid(topl,
  FloatType = Float64,
  DeviceArray = ArrayType,
  polynomialorder = 4,
  meshwarp = warpfun,
)

dg = DGModel(MMSModel{3}(),
  grid,
  Rusanov(),
  DefaultGradNumericalFlux())


param = init_ode_param(dg)
Q = init_ode_state(dg, param, 0.0)

dt = 5e-3 / Ne[1]




lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)
#dg(lsrk.dQ, Q, param, dt, increment = false)
#dg(lsrk.dQ, Q, param, dt, increment = true)

solve!(Q, lsrk, param; timeend=1)
 
