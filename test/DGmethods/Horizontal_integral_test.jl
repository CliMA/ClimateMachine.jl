using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.Atmos
using CLIMA.VariableTemplates
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK
using CLIMA.Atmos: vars_state, vars_aux
using DelimitedFiles
using GPUifyLoops
using Random
using CLIMA.IOstrings
@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray,) 
else
  const ArrayTypes = (Array,)
end
FT=Float64
MPI.Initialized() || MPI.Init()
  mpicomm = MPI.COMM_WORLD
ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))
@static if haspkg("CUDAnative")
      device!(MPI.Comm_rank(mpicomm) % length(devices()))
  end
for ArrayType in ArrayTypes
warpfun = (ξ1, ξ2, ξ3) -> begin
  x1 = ξ1 + (ξ1 - 1/2) * cos(2 * π * ξ2 * ξ3) / 4
  x2 = ξ2 + exp(sin(2π * (ξ1 * ξ2 + ξ3)))/20
  x3 = ξ3 + ξ1 / 4 + ξ2^2 / 2 + sin(ξ1 * ξ2 * ξ3)
  return (x1, x2, x3)
end
for N in 3:6
for Ne in 1:5
println(N," ",Ne)
brickrange = (range(FT(0); length=Ne+1, stop=1),
              range(FT(0); length=Ne+1, stop=1),
              range(FT(0); length=2, stop=1))
topl = StackedBrickTopology(mpicomm, brickrange,
                            periodicity = (false, false, false))
grid = DiscontinuousSpectralElementGrid(topl,
                                        FloatType = FT,
                                        DeviceArray = ArrayType,
                                        polynomialorder = N,
                                        meshwarp = warpfun,
                                       )
N = polynomialorder(grid)
  vgeo=grid.vgeo
  Nq = N + 1
  Nqk = dimensionality(grid) == 2 ? 1 : Nq
  nrealelem = length(topl.realelems)
  host_array = Array ∈ typeof(vgeo).parameters
  localvgeo = host_array ? vgeo : Array(vgeo)
  S = zeros(Nqk)
  S1 = zeros(Nqk)
  for e in 1:nrealelem
  for k in 1:Nqk
  for j in 1:Nq
    for i in 1:Nq
      ijk = i + Nq * ((j-1)+ Nq * (k-1))
      S[k] += localvgeo[ijk,grid.x1id,e] * localvgeo[ijk,grid.MHid,e]
      S1[k] += localvgeo[ijk,grid.MHid,e]
    end
  end
  end
  end
  for k in 1:Nqk
  println(0.5-S[k]/S1[k])
  end
  end
  end
end
