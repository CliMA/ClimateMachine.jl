using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
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

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

using CLIMA.Atmos
using CLIMA.Atmos: internal_energy, thermo_state
import CLIMA.Atmos: MoistureModel, temperature, pressure, soundspeed

init_state!(bl, state, aux, coords, t) = nothing

# initial condition
using CLIMA.Atmos: vars_aux

function run1(mpicomm, ArrayType, dim, topl, N, timeend, FT, dt)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N
                                         )

  T_s = 320.0
  RH = 0.01
  model = AtmosModel{FT}(AtmosLESConfiguration;
                         ref_state=HydrostaticState(IsothermalProfile(T_s), RH),
                        init_state=init_state!)

  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralNumericalFluxGradient())

  Q = init_ode_state(dg, FT(0))

  mkpath("vtk")
  outprefix = @sprintf("vtk/refstate")
  writevtk(outprefix, dg.auxstate, dg, flattenednames(vars_aux(model, FT)))
  return FT(0)
end

function run2(mpicomm, ArrayType, dim, topl, N, timeend, FT, dt)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N
                                         )

  T_min, T_s, Γ = FT(290), FT(320), FT(6.5*10^-3)
  RH = 0.01
  model = AtmosModel{FT}(AtmosLESConfiguration;
                         ref_state=HydrostaticState(LinearTemperatureProfile(T_min, T_s, Γ), RH),
                         init_state=init_state!)

  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralNumericalFluxGradient())

  Q = init_ode_state(dg, FT(0))

  mkpath("vtk")
  outprefix = @sprintf("vtk/refstate")
  writevtk(outprefix, dg.auxstate, dg, flattenednames(vars_aux(model, FT)))
  return FT(0)
end

using Test
let
  CLIMA.init()
  ArrayType = CLIMA.array_type()

  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  polynomialorder = 4
  base_num_elem = 4

  expected_result = [0.0 0.0 0.0; 0.0 0.0 0.0]
  lvls = integration_testing ? size(expected_result, 2) : 1

for FT in (Float64,) #Float32)
  result = zeros(FT, lvls)
  x_max = FT(25*10^3)
  y_max = FT(25*10^3)
  z_max = FT(25*10^3)
  dim = 3
  for l = 1:lvls
    Ne = (2^(l-1) * base_num_elem, 2^(l-1) * base_num_elem)
    brickrange = (range(FT(0); length=Ne[1]+1, stop=x_max),
                  range(FT(0); length=Ne[2]+1, stop=y_max),
                  range(FT(0); length=Ne[2]+1, stop=z_max))
    topl = BrickTopology(mpicomm, brickrange,
                         periodicity = (false, false, false))
    dt = 5e-3 / Ne[1]

    timeend = 2*dt
    nsteps = ceil(Int64, timeend / dt)
    dt = timeend / nsteps

    @info (ArrayType, FT, dim)
    result[l] = run1(mpicomm, ArrayType, dim, topl,
                    polynomialorder, timeend, FT, dt)
    result[l] = run2(mpicomm, ArrayType, dim, topl,
                    polynomialorder, timeend, FT, dt)
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


#nothing
