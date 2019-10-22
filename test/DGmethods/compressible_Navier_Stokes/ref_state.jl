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
using CLIMA.Atmos: internal_energy, thermo_state
import CLIMA.Atmos: MoistureModel, temperature, pressure, soundspeed

init_state!(state, aux, coords, t) = nothing

# initial condition
using CLIMA.Atmos: vars_aux

function run1(mpicomm, ArrayType, dim, topl, N, timeend, DFloat, dt)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N
                                         )

  T_s = 320.0
  RH = 0.01
  model = AtmosModel(FlatOrientation(),
                     HydrostaticState(IsothermalProfile(T_s), RH),
                     ConstantViscosityWithDivergence(DFloat(1)),
                     EquilMoist(),
                     NoRadiation(),
                     nothing,
                     NoFluxBC(),
                     init_state!)

  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())

  Q = init_ode_state(dg, DFloat(0))

  mkpath("vtk")
  outprefix = @sprintf("vtk/refstate")
  writevtk(outprefix, dg.auxstate, dg, flattenednames(vars_aux(model, DFloat)))
  return DFloat(0)
end

function run2(mpicomm, ArrayType, dim, topl, N, timeend, DFloat, dt)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N
                                         )

  T_min, T_s, Γ = DFloat(290), DFloat(320), DFloat(6.5*10^-3)
  RH = 0.01
  model = AtmosModel(FlatOrientation(),
                     HydrostaticState(LinearTemperatureProfile(T_min, T_s, Γ), RH),
                     ConstantViscosityWithDivergence(DFloat(1)),
                     EquilMoist(),
                     NoRadiation(),
                     nothing,
                     NoFluxBC(),
                     init_state!)

  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())

  Q = init_ode_state(dg, DFloat(0))

  mkpath("vtk")
  outprefix = @sprintf("vtk/refstate")
  writevtk(outprefix, dg.auxstate, dg, flattenednames(vars_aux(model, DFloat)))
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
  x_max = DFloat(25*10^3)
  y_max = DFloat(25*10^3)
  z_max = DFloat(25*10^3)
  dim = 3
  for l = 1:lvls
    Ne = (2^(l-1) * base_num_elem, 2^(l-1) * base_num_elem)
    brickrange = (range(DFloat(0); length=Ne[1]+1, stop=x_max),
                  range(DFloat(0); length=Ne[2]+1, stop=y_max),
                  range(DFloat(0); length=Ne[2]+1, stop=z_max))
    topl = BrickTopology(mpicomm, brickrange,
                         periodicity = (false, false, false))
    dt = 5e-3 / Ne[1]

    timeend = 2*dt
    nsteps = ceil(Int64, timeend / dt)
    dt = timeend / nsteps

    @info (ArrayType, DFloat, dim)
    result[l] = run1(mpicomm, ArrayType, dim, topl,
                    polynomialorder, timeend, DFloat, dt)
    result[l] = run2(mpicomm, ArrayType, dim, topl,
                    polynomialorder, timeend, DFloat, dt)
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


#nothing
