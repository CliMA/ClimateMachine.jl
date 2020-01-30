# Load Packages
using MPI
using CLIMA
using CLIMA.Mesh.Topologies: StackedBrickTopology
using CLIMA.Mesh.Grids: DiscontinuousSpectralElementGrid
using CLIMA.Mesh.Geometry
using CLIMA.Mesh.Filters
using CLIMA.DGmethods: DGModel, init_ode_state
using CLIMA.DGmethods.NumericalFluxes: Rusanov, CentralNumericalFluxGradient,
                                       CentralNumericalFluxDiffusive,
                                       CentralNumericalFluxGradient
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.SubgridScaleParameters
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
using Random
using CLIMA.Atmos: vars_state, vars_aux

const ArrayType = CLIMA.array_type()

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

# -------------- Problem constants ------------------- #
const L               = 1000
const (xmin,xmax)     = (0,L*π)
const (ymin,ymax)     = (0,L*π)
const (zmin,zmax)     = (0,L*π)
const Ne              = (10,10,10)
const polynomialorder = 4
const dim             = 3
const dt              = 0.01

Base.@kwdef struct TaylorGreenVortexSetup{FT}
  M₀::FT    = 0.1
  ρ_ref::FT = 1.20
  p_ref::FT = MSLP
  γ::FT     = 1.4
  T₀::FT    = p_ref / ((γ - 1) * cv_d * ρ_ref)
  c::FT     = soundspeed_air(T₀)
  V₀::FT    = M₀ * c
end

# ------------- Initial condition function ----------- #
function (setup::TaylorGreenVortexSetup)(state::Vars, aux::Vars, (x1,x2,x3), t)
  FT = eltype(state)

  ρ_ref = setup.ρ_ref
  p_ref = setup.p_ref
  V₀    = setup.V₀

  p  = p_ref + ρ_ref * V₀^2 / 16 * (cos(2 * x1 / L) + cos(2 * x2 / L)) * (cos(2 * x3 / L) + 2)

  u1 = V₀ * sin(x1 / L) * cos(x2 / L) * cos(x3 / L)
  u2 = -V₀ * cos(x1 / L) * sin(x2 / L) * sin(x3 / L)
  u = SVector(u1, u2, 0)

  state.ρ  = ρ_ref
  state.ρu = state.ρ * u

  kinetic_energy = u' * u / 2

  T₁ = p / (R_d * state.ρ)

  state.ρe = state.ρ * total_energy(kinetic_energy, FT(0), T₁)
end

# --------------- Driver definition ------------------ #
function run(mpicomm, setup,
             topl, dim, Ne, polynomialorder,
             timeend, FT, dt)

  # -------------- Define grid ----------------------------------- #
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = polynomialorder)
  # -------------- Define model ---------------------------------- #
  model = AtmosModel(NoOrientation(),
                     NoReferenceState(),
                     Vreman{FT}(C_smag),
                     DryModel(),
                     NoPrecipitation(),
                     NoRadiation(),
                     NoSubsidence{FT}(),
                     nothing,
                     PeriodicBC(),
                     setup)

  # -------------- Define dgbalancelaw --------------------------- #
  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralNumericalFluxGradient())

  Q = init_ode_state(dg, FT(0))

  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e
  ArrayType = %s
  FloatType = %s""" eng0 ArrayType FT

  # Set up the information callback (output field dump is via vtk callback: see cbinfo)
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(10, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      ρu₁ = Array(Q[:, 2, :])
      ρu₂ = Array(Q[:, 3, :])
      ρu₃ = Array(Q[:, 4, :])
      ρᵣ  = Array(Q[:, 1, :])
      ke  = @views sum((ρu₁ .^ 2 + ρu₂ .^ 2 + ρu₃ .^ 2) ./ (2 * ρᵣ))
      @info @sprintf("""Update
                     simtime        = %.16e
                     runtime        = %s
                     norm(Q)        = %.16e
                     kinetic energy = %.16e""", ODESolvers.gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy, ke)
    end
  end

  vtkdir = "vtk_tgv"
  mkpath(vtkdir)
  vtkstep = 0
  do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)

  # setup the output callback
  outputtime = timeend
  stepsize = 10
  # stepsize = floor(outputtime / dt)
  cbvtk = GenericCallbacks.EveryXSimulationSteps(stepsize) do
    vtkstep += 1
    do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo,cbvtk))
  # End of the simulation information
  engfe = norm(Qe)
  errf = euclidean_distance(Q, Qe)
  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  norm(Q - Qe)            = %.16e
  norm(Q - Qe) / norm(Qe) = %.16e
  """ engf engf/eng0 engf-eng0 errf errf / engfe
engf/eng0
end

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, model, testname = "taylor-green-vortex")
  ## name of the file that this MPI rank will write
  filename = @sprintf("%s/%s_mpirank%04d_step%04d",
                      vtkdir, testname, MPI.Comm_rank(mpicomm), vtkstep)

  statenames = flattenednames(vars_state(model, eltype(Q)))

  writevtk(filename, Q, dg, statenames)

  ## Generate the pvtu file for these vtk files
  if MPI.Comm_rank(mpicomm) == 0
    ## name of the pvtu file
    pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

    ## name of each of the ranks vtk files
    prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
      @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
    end

    writepvtu(pvtuprefix, prefixes, (statenames...,))

    @info "Done writing VTK: $pvtuprefix"
  end
end

# --------------- Test block / Loggers ------------------ #
using Test
let
  CLIMA.init()
  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))
  for FT in (Float64,)
    setup = TaylorGreenVortexSetup{FT}()
    V₀    = setup.V₀
    timeend = 9 * L / V₀
    @show timeend
    brickrange = (range(FT(xmin); length=Ne[1]+1, stop=xmax),
                  range(FT(ymin); length=Ne[2]+1, stop=ymax),
                  range(FT(zmin); length=Ne[3]+1, stop=zmax))
    topl = StackedBrickTopology(mpicomm, brickrange, periodicity = (true, true, true))
    engf_eng0 = run(mpicomm, setup,
                    topl, dim, Ne, polynomialorder,
                    timeend, FT, dt)
  end
end
