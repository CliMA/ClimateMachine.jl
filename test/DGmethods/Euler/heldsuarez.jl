using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Filters
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.Vtk
using CLIMA.PlanetParameters: planet_radius, R_d, cv_d, cp_d, MSLP
using CLIMA.MoistThermodynamics: air_pressure, air_temperature, internal_energy, air_density,
                                 gas_constant_air, soundspeed_air

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
end

include("standalone_euler_model.jl")

struct HeldSuarez{DFloat} <: EulerProblem
  domain_height::DFloat
end

function parametrized_source!(::ReferenceStateEulerModel, ::HeldSuarez,
                              source::Vars, state::Vars, aux::Vars)

  δρ, ρu⃗, δρe = state.δρ, state.ρu⃗, state.δρe
  ρ_ref, ρe_ref = aux.ρ_ref, aux.ρe_ref
  ρ, ρe = ρ_ref + δρ, ρe_ref + δρe
  x⃗, ϕ = aux.coord, aux.gravity.ϕ

  e = ρe / ρ
  u⃗ = ρu⃗ / ρ
  e_int = e - u⃗' * u⃗ / 2 - ϕ
  T = air_temperature(e_int)
  p = air_pressure(T, ρ)

  DFloat = eltype(ρu⃗)
  # Held-Suarez constants
  p0 :: DFloat = MSLP
  ka :: DFloat = 1 / 40 / 86400
  kf :: DFloat = 1 / 86400
  ks :: DFloat = 1 / 4 / 86400
  ΔT_y :: DFloat = 60 
  ΔT_z :: DFloat = 10   
  temp0 :: DFloat = 315
  temp_min :: DFloat = 200
  hd :: DFloat = 7000 #from Smolarkiewicz JAS 2001 paper    
  σ_b :: DFloat = 7 / 10

  r = norm(x⃗, 2)
  λ = atan(x⃗[2], x⃗[1])
  φ = asin(x⃗[3] / r)
  h = r - DFloat(planet_radius)

  σ = exp(-h/hd)
#  σ = P/p0
  Δσ = (σ - σ_b)/(1 - σ_b)
  π = σ^(R_d/cp_d)
  c = max(0, Δσ)
  T_eq = ( temp0 - ΔT_y*sin(φ)^2 - ΔT_z*log(σ)*cos(φ)^2 )*π
  T_eq = max(temp_min, T_eq)
  kt = ka + (ks-ka)*c*cos(φ)^4
  kv = kf*c
  kt = kt*(1 - T_eq/T)

  source.ρu⃗ += -kv * ρu⃗
  source.δρe += -(kt * ρ * cv_d * T +  kv * ρu⃗' * ρu⃗ / ρ)
end

function initial_condition!(::ReferenceStateEulerModel, ::HeldSuarez, state::Vars, _...)
  state.δρ = 0
  state.ρu⃗ = SVector(0, 0, 0)
  state.δρe = 0
end

function reference_state!(::ReferenceStateEulerModel, ::HeldSuarez, aux::Vars, x⃗)
  DFloat = eltype(x⃗)
  
  r = norm(x⃗, 2)
  h = r - DFloat(planet_radius)
  ϕ = aux.gravity.ϕ

  T_ref = DFloat(255)
  scale_height = R_d * T_ref / grav

  P_ref = MSLP * exp(-h / scale_height)
  ρ_ref = air_density(T_ref, P_ref)
  ρe_ref = ρ_ref * (internal_energy(T_ref) + ϕ)

  aux.ρ_ref = ρ_ref
  aux.ρe_ref = ρe_ref
end

gravitymodel(::HeldSuarez) = SphereGravity(planet_radius)

function run(mpicomm, ArrayType, problem, topl, N, outputtime, timeend, DFloat, dt)
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                          meshwarp = Topologies.cubedshellwarp
                                         )

  dg = DGModel(ReferenceStateEulerModel(true, problem, Coriolis()),
               grid,
               Rusanov(),
               DefaultGradNumericalFlux())

  param = init_ode_param(dg)

  Q = init_ode_state(dg, param, DFloat(0))

  lsrk = LSRK144NiegemannDiehlBusch(dg, Q; dt = dt, t0 = 0)

  filter = ExponentialFilter(grid, 0, 14)
  cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
    Filters.apply!(Q, 1:size(Q, 2), grid, filter)
    nothing
  end

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", ODESolvers.gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
    end
  end

  ## Define a convenience function for VTK output
  VTKDIR = "hs_vtk"
  mkpath(VTKDIR)
  function do_output(vtk_step)
    ## name of the file that this MPI rank will write
    filename = @sprintf("%s/held_suarez_mpirank%04d_step%04d",
                        VTKDIR, MPI.Comm_rank(mpicomm), vtk_step)

    ## write the vtk file for this MPI rank
    statenames = ("δρ", "ρu", "ρv", "ρw", "δρe")
    auxnames = ("x", "y", "z" ,"ρ_ref", "ρe_ref", "ϕ", "∇ϕ_x", "∇ϕ_y", "∇ϕ_z")
    writevtk(filename, Q, dg, statenames, param.aux, auxnames)

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
      ## name of the pvtu file
      pvtuprefix = @sprintf("%s/held_suarez_step%04d", VTKDIR, vtk_step)

      ## name of each of the ranks vtk files
      prefixes = ntuple(i->
                        @sprintf("held_suarez_mpirank%04d_step%04d",
                                 i-1, vtk_step),
                        MPI.Comm_size(mpicomm))

      ## Write out the pvtu file
      writepvtu(pvtuprefix, prefixes, (statenames..., auxnames...))

      ## write that we have written the file
      @info @sprintf("Done writing VTK: %s", pvtuprefix)
    end

  end

  ## Setup callback for writing VTK every hour of simulation time and dump
  #initial file
  vtk_step = 0
  do_output(vtk_step)
  cbvtk = GenericCallbacks.EveryXSimulationSteps(floor(outputtime / dt)) do
    vtk_step += 1
    do_output(vtk_step)
    nothing
  end

  solve!(Q, lsrk, param; timeend=timeend, callbacks=(cbinfo, cbfilter, cbvtk))

  engf = norm(Q)

  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  """ engf engf-eng0
  engf
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

  DFloat = Float64
  polynomialorder = 5
  num_elem_horz = 6
  num_elem_vert = 8

  @static if haspkg("CuArrays")
    ArrayType = CuArray
  else
    ArrayType = Array
  end

  domain_height = DFloat(30e3)
  hs = HeldSuarez(domain_height)

  Rrange = range(DFloat(planet_radius), length = num_elem_vert + 1,
                 stop = planet_radius + hs.domain_height)

  topl = StackedCubedSphereTopology(mpicomm, num_elem_horz, Rrange)

  element_size = (hs.domain_height / num_elem_vert)
  acoustic_speed = soundspeed_air(DFloat(315))
  lucas_magic_factor = 14
  dt = lucas_magic_factor * element_size / acoustic_speed / polynomialorder ^ 2
  
  seconds = 1
  minutes = 60
  hours = 3600
  days = 86400
  outputtime = minutes / 2
  timeend = 1minutes

  expected_error = 2.4944957714032431e+09
  error = run(mpicomm, ArrayType, hs, topl, polynomialorder, outputtime, timeend, DFloat, dt)
  @test error ≈ expected_error
end
nothing
