using CLIMA: haspkg
using CLIMA.Mesh.Topologies: StackedCubedSphereTopology, cubedshellwarp, grid1d
using CLIMA.Mesh.Grids: DiscontinuousSpectralElementGrid
using CLIMA.DGmethods: DGModel, init_ode_state
using CLIMA.DGmethods.NumericalFluxes: Rusanov, CentralGradPenalty,
                                       CentralNumericalFluxDiffusive
using CLIMA.ODESolvers: solve!, gettime
using CLIMA.LowStorageRungeKuttaMethod: LSRK54CarpenterKennedy
using CLIMA.VTK: writevtk, writepvtu
using CLIMA.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps
using CLIMA.PlanetParameters: planet_radius, day
using CLIMA.MoistThermodynamics: air_density, soundspeed_air, internal_energy
using CLIMA.Atmos: AtmosModel, SphericalOrientation,
                   DryModel, NoRadiation, NoFluxBC,
                   ConstantViscosityWithDivergence,
                   vars_state, vars_aux,
                   Gravity, HydrostaticState, IsothermalProfile
using CLIMA.VariableTemplates: flattenednames

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test
@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayType = CuArray
else
  const ArrayType = Array
end

const output_vtk = true

function main()
  MPI.Initialized() || MPI.Init()
  mpicomm = MPI.COMM_WORLD

  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = Dict("DEBUG" => Logging.Debug,
                  "WARN"  => Logging.Warn,
                  "ERROR" => Logging.Error,
                  "INFO"  => Logging.Info)[ll]

  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  polynomialorder = 5
  numelem_horz = 4

  numelem_vert = 4
  #numelem_vert = 30 # Resolution required for stable long time result

  timeend = 60
  #timeend = 33 * 60 * 60 # Full simulation

  outputtime = 60 * 60
  
  for FT in (Float64,)

    run(mpicomm, polynomialorder, numelem_horz, numelem_vert,
        timeend, outputtime, ArrayType, FT)
  end
end

function run(mpicomm, polynomialorder, numelem_horz, numelem_vert,
             timeend, outputtime, ArrayType, FT)

  setup = AcousticWaveSetup{FT}()

  vert_range = grid1d(FT(planet_radius), FT(planet_radius + setup.domain_height), nelem = numelem_vert)
  topology = StackedCubedSphereTopology(mpicomm, numelem_horz, vert_range)

  grid = DiscontinuousSpectralElementGrid(topology,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = polynomialorder,
                                          meshwarp = cubedshellwarp)

  model = AtmosModel(SphericalOrientation(),
                     HydrostaticState(IsothermalProfile(setup.T_ref), FT(0), true),
                     ConstantViscosityWithDivergence(FT(0)),
                     DryModel(),
                     NoRadiation(),
                     Gravity(), 
                     NoFluxBC(),
                     setup)

  dg = DGModel(model, grid, Rusanov(),
               CentralNumericalFluxDiffusive(), CentralGradPenalty())

  # determine the time step
  element_size = (setup.domain_height / numelem_vert)
  acoustic_speed = soundspeed_air(FT(setup.T_ref))
  dt = element_size / acoustic_speed / polynomialorder ^ 2
  # Adjust the time step so we exactly hit 1 hour for VTK output
  dt = 60 * 60 / ceil(60 * 60 / dt)

  Q = init_ode_state(dg, FT(0))
  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
                    ArrayType       = %s
                    FT              = %s
                    polynomialorder = %d
                    numelem_horz    = %d
                    numelem_vert    = %d
                    dt              = %.16e
                    norm(Q₀)        = %.16e
                    """ "$ArrayType" "$FT" polynomialorder numelem_horz numelem_vert dt eng0

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      runtime = Dates.format(convert(DateTime, now() - starttime[]), dateformat"HH:MM:SS")
      @info @sprintf """Update
                        simtime = %.16e
                        runtime = %s
                        norm(Q) = %.16e
                        """ gettime(lsrk) runtime energy
    end
  end
  callbacks = (cbinfo,)

  if output_vtk
    # create vtk dir
    vtkdir = "vtk_acousticwave" *
      "_poly$(polynomialorder)_horz$(numelem_horz)_vert$(numelem_vert)" *
      "_$(ArrayType)_$(FT)"
    mkpath(vtkdir)

    vtkstep = 0
    # output initial step
    do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)

    # setup the output callback
    cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
      vtkstep += 1
      Qe = init_ode_state(dg, gettime(lsrk))
      do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)
    end
    callbacks = (callbacks..., cbvtk)
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=callbacks)

  # final statistics
  engf = norm(Q)
  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  """ engf engf/eng0 engf-eng0
end

Base.@kwdef struct AcousticWaveSetup{FT}
  domain_height::FT = 10e3
  T_ref::FT = 300
  α::FT = 3
  γ::FT = 100
  nv::Int = 1
end

function (setup::AcousticWaveSetup)(state, aux, coords, t) 
  # callable to set initial conditions
  FT = eltype(state)

  r = norm(coords, 2)
  @inbounds λ = atan(coords[2], coords[1])
  @inbounds φ = asin(coords[3] / r)
  h = r - FT(planet_radius)

  β = min(FT(1), setup.α * acos(cos(φ) * cos(λ)))
  f = (1 + cos(π * β)) / 2
  g = sin(setup.nv * π * h / setup.domain_height)
  Δp = setup.γ * f * g
  p = aux.ref_state.p + Δp

  state.ρ = air_density(setup.T_ref, p)
  state.ρu = SVector{3, FT}(0, 0, 0)
  state.ρe = state.ρ * (internal_energy(setup.T_ref) + aux.orientation.Φ)
  nothing
end

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, model, testname = "acousticwave")
  ## name of the file that this MPI rank will write
  filename = @sprintf("%s/%s_mpirank%04d_step%04d",
                      vtkdir, testname, MPI.Comm_rank(mpicomm), vtkstep)

  statenames = flattenednames(vars_state(model, eltype(Q)))
  auxnames = flattenednames(vars_aux(model, eltype(Q))) 
  writevtk(filename, Q, dg, statenames, dg.auxstate, auxnames)

  ## Generate the pvtu file for these vtk files
  if MPI.Comm_rank(mpicomm) == 0
    ## name of the pvtu file
    pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

    ## name of each of the ranks vtk files
    prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
      @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
    end

    writepvtu(pvtuprefix, prefixes, (statenames..., auxnames...))

    @info "Done writing VTK: $pvtuprefix"
  end
end

main()
