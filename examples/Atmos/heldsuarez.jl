using CLIMA: haspkg
using CLIMA.Mesh.Topologies: StackedCubedSphereTopology, cubedshellwarp, grid1d
using CLIMA.Mesh.Grids: DiscontinuousSpectralElementGrid
using CLIMA.Mesh.Filters
using CLIMA.DGmethods: DGModel, init_ode_state
using CLIMA.DGmethods.NumericalFluxes: Rusanov, CentralGradPenalty,
                                       CentralNumericalFluxDiffusive
using CLIMA.ODESolvers: solve!, gettime
using CLIMA.LowStorageRungeKuttaMethod: LSRK144NiegemannDiehlBusch
using CLIMA.VTK: writevtk, writepvtu
using CLIMA.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps
using CLIMA.MPIStateArrays: euclidean_distance
using CLIMA.PlanetParameters: R_d, grav, MSLP, planet_radius, cp_d, cv_d, day
using CLIMA.MoistThermodynamics: air_density, total_energy, soundspeed_air, internal_energy, air_temperature
using CLIMA.Atmos: AtmosModel, SphericalOrientation, NoReferenceState,
                   DryModel, NoRadiation, NoFluxBC,
                   ConstantViscosityWithDivergence,
                   vars_state, vars_aux,
                   Gravity, Coriolis,
                   HydrostaticState, IsothermalProfile
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
  numelem_horz = 6
  numelem_vert = 8
  timeend = 60 # 400day
  outputtime = 2day
  
  for FT in (Float64,)

    run(mpicomm, polynomialorder, numelem_horz, numelem_vert,
        timeend, outputtime, ArrayType, FT)
  end
end

function run(mpicomm, polynomialorder, numelem_horz, numelem_vert,
             timeend, outputtime, ArrayType, FT)

  setup = HeldSuarezSetup{FT}()

  vert_range = grid1d(FT(planet_radius), FT(planet_radius + setup.domain_height), nelem = numelem_vert)
  topology = StackedCubedSphereTopology(mpicomm, numelem_horz, vert_range)

  grid = DiscontinuousSpectralElementGrid(topology,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = polynomialorder,
                                          meshwarp = cubedshellwarp)

  model = AtmosModel(SphericalOrientation(),
                     HydrostaticState(IsothermalProfile(setup.T_initial), FT(0)),
                     ConstantViscosityWithDivergence(FT(0)),
                     DryModel(),
                     NoRadiation(),
                     (Gravity(), Coriolis(), held_suarez_forcing!), 
                     NoFluxBC(),
                     setup)

  dg = DGModel(model, grid, Rusanov(),
               CentralNumericalFluxDiffusive(), CentralGradPenalty())

  # determine the time step
  element_size = (setup.domain_height / numelem_vert)
  acoustic_speed = soundspeed_air(FT(315))
  lucas_magic_factor = 14
  dt = lucas_magic_factor * element_size / acoustic_speed / polynomialorder ^ 2

  Q = init_ode_state(dg, FT(0))
  lsrk = LSRK144NiegemannDiehlBusch(dg, Q; dt = dt, t0 = 0)

  filterorder = 14
  filter = ExponentialFilter(grid, 0, filterorder)
  cbfilter = EveryXSimulationSteps(1) do
    Filters.apply!(Q, 1:size(Q, 2), grid, filter)
    nothing
  end

  eng0 = norm(Q)
  @info @sprintf """Starting
                    ArrayType       = %s
                    FT              = %s
                    polynomialorder = %d
                    numelem_horz    = %d
                    numelem_vert    = %d
                    filterorder     = %d
                    dt              = %.16e
                    norm(Q₀)        = %.16e
                    """ "$ArrayType" "$FT" polynomialorder numelem_horz numelem_vert filterorder dt eng0

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
  callbacks = (cbinfo, cbfilter)

  if output_vtk
    # create vtk dir
    vtkdir = "vtk_heldsuarez" *
      "_poly$(polynomialorder)_horz$(numelem_horz)_vert$(numelem_vert)" *
      "_filter$(filterorder)_$(ArrayType)_$(FT)"
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

Base.@kwdef struct HeldSuarezSetup{FT}
  p_ground::FT = MSLP
  T_initial::FT = 255
  domain_height::FT = 30e3
end

function (setup::HeldSuarezSetup)(state, aux, coords, t) 
  # callable to set initial conditions
  FT = eltype(state)

  r = norm(coords, 2)
  h = r - FT(planet_radius)

  scale_height = R_d * setup.T_initial / grav
  p = setup.p_ground * exp(-h / scale_height)

  state.ρ = air_density(setup.T_initial, p)
  state.ρu = SVector{3, FT}(0, 0, 0)
  state.ρe = state.ρ * (internal_energy(setup.T_initial) + aux.orientation.Φ)
  nothing
end

function held_suarez_forcing!(source, state, aux, t::Real)
  FT = eltype(state)

  ρ = state.ρ
  ρu = state.ρu
  ρe = state.ρe
  coord = aux.coord
  Φ = aux.orientation.Φ
  e = ρe / ρ
  u = ρu / ρ
  e_int = e - u' * u / 2 - Φ
  T = air_temperature(e_int)
  # Held-Suarez constants
  k_a = FT(1 / (40 * day))
  k_f = FT(1 / day)
  k_s = FT(1 / (4 * day))
  ΔT_y = FT(60)
  Δθ_z = FT(10)
  T_equator = FT(315)
  T_min = FT(200)
  σ_b = FT(7 / 10)
  r = norm(coord, 2)
  @inbounds λ = atan(coord[2], coord[1])
  @inbounds φ = asin(coord[3] / r)
  h = r - FT(planet_radius)
  scale_height = FT(7000) #from Smolarkiewicz JAS 2001 paper
  σ = exp(-h / scale_height)
  # TODO: use
  #  p = air_pressure(T, ρ)
  #  σ = p/p0
  exner_p = σ ^ (R_d / cp_d)
  Δσ = (σ - σ_b) / (1 - σ_b)
  height_factor = max(0, Δσ)
  T_equil = (T_equator - ΔT_y * sin(φ) ^ 2 - Δθ_z * log(σ) * cos(φ) ^ 2 ) * exner_p
  T_equil = max(T_min, T_equil)
  k_T = k_a + (k_s - k_a) * height_factor * cos(φ) ^ 4
  k_v = k_f * height_factor
  source.ρu += -k_v * ρu
  source.ρe += -k_T * ρ * cv_d * (T - T_equil) - k_v * ρu' * ρu / ρ
end

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, model, testname = "heldsuarez")
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
