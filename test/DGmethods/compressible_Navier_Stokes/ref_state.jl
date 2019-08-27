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

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray, )
else
  const ArrayTypes = (Array, )
end

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

using CLIMA.Atmos
using CLIMA.Atmos: internal_energy, get_phase_partition, thermo_state
import CLIMA.Atmos: MoistureModel, temperature, pressure, soundspeed, update_aux!

init_state!(state, aux, coords, t) = nothing
source!(source, state, aux, t) = nothing

# initial condition
using CLIMA.Atmos: vars_aux

function run(mpicomm, ArrayType, dim, topl, warpfun, N, timeend, DFloat, dt)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N
                                         )

  T_s = 320.0
  RH = 0.01
  if dim == 2
    model = AtmosModel(FlatOrientation(),
                       IsothermalHydrostaticState(T_s, RH),
                       ConstantViscosityWithDivergence(DFloat(1)),
                       EquilMoist(),
                       NoRadiation(),
                       source!,
                       NoFluxBC(),
                       init_state!)
  else
    model = AtmosModel(FlatOrientation(),
                       IsothermalHydrostaticState(T_s, RH),
                       ConstantViscosityWithDivergence(DFloat(1)),
                       EquilMoist(),
                       NoRadiation(),
                       source!,
                       NoFluxBC(),
                       init_state!)
  end
  dg = DGModel(model,
               grid,
               Rusanov(),
               DefaultGradNumericalFlux())

  param = init_ode_param(dg)

  Q = init_ode_state(dg, param, DFloat(0))

  mkpath("vtk")
  outprefix = @sprintf("vtk/refstate")
  writevtk(outprefix, param[1], dg, flattenednames(vars_aux(model, DFloat)))
  return DFloat(0)
end

using Test
let
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

  polynomialorder = 4
  base_num_elem = 4

  expected_result = [0.0 0.0 0.0; 0.0 0.0 0.0]
  lvls = integration_testing ? size(expected_result, 2) : 1

@testset "$(@__FILE__)" for ArrayType in ArrayTypes
for DFloat in (Float64,) #Float32)
  result = zeros(DFloat, lvls)
  for dim = 2:3
    for l = 1:lvls
      if dim == 2
        Ne = (2^(l-1) * base_num_elem, 2^(l-1) * base_num_elem)
        brickrange = (range(DFloat(0); length=Ne[1]+1, stop=1),
                      range(DFloat(0); length=Ne[2]+1, stop=1))
        topl = BrickTopology(mpicomm, brickrange,
                             periodicity = (false, false))
        dt = 1e-2 / Ne[1]
        warpfun = (x1, x2, _) -> begin
          (x1 + sin(x1*x2), x2 + sin(2*x1*x2), 0)
        end

      elseif dim == 3
        Ne = (2^(l-1) * base_num_elem, 2^(l-1) * base_num_elem)
        brickrange = (range(DFloat(0); length=Ne[1]+1, stop=1),
                      range(DFloat(0); length=Ne[2]+1, stop=1),
        range(DFloat(0); length=Ne[2]+1, stop=1))
        topl = BrickTopology(mpicomm, brickrange,
                             periodicity = (false, false, false))
        dt = 5e-3 / Ne[1]
        warpfun = (x1, x2, x3) -> begin
          (x1 + (x1-1/2)*cos(2*π*x2*x3)/4,
           x2 + exp(sin(2π*(x1*x2+x3)))/20,
          x3 + x1/4 + x2^2/2 + sin(x1*x2*x3))
        end
      end
      timeend = 2*dt
      nsteps = ceil(Int64, timeend / dt)
      dt = timeend / nsteps

      @info (ArrayType, DFloat, dim)
      result[l] = run(mpicomm, ArrayType, dim, topl, warpfun,
                      polynomialorder, timeend, DFloat, dt)
      @test result[l] ≈ DFloat(expected_result[dim-1, l])
    end
    if integration_testing
      @info begin
        msg = ""
        for l = 1:lvls-1
          rate = log2(result[l]) - log2(result[l+1])
          msg *= @sprintf("\n  rate for level %d = %e\n", l, rate)
        end
        msg
      end
    end
  end
end
end
end


#nothing
