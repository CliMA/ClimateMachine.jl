using CLIMA
using CLIMA.Mesh.Topologies: StackedCubedSphereTopology, cubedshellwarp, grid1d
using CLIMA.Mesh.Grids: DiscontinuousSpectralElementGrid, VerticalDirection
using CLIMA.Mesh.Filters
using CLIMA.DGmethods: DGModel, init_ode_state
using CLIMA.DGmethods.NumericalFluxes: Rusanov, CentralNumericalFluxGradient,
                                       CentralNumericalFluxDiffusive
using CLIMA.ODESolvers
using CLIMA.GeneralizedMinimalResidualSolver
using CLIMA.ColumnwiseLUSolver: ManyColumnLU
using CLIMA.VTK: writevtk, writepvtu
using CLIMA.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps
using CLIMA.PlanetParameters: planet_radius, day
using CLIMA.MoistThermodynamics: air_density, soundspeed_air, internal_energy, PhaseDry_given_pT, PhasePartition
using CLIMA.Atmos: AtmosModel, SphericalOrientation,
                   DryModel, NoPrecipitation, NoRadiation, NoFluxBC,
                   ConstantViscosityWithDivergence,
                   vars_state, vars_aux,
                   Gravity, HydrostaticState, IsothermalProfile,
                   AtmosAcousticGravityLinearModel, AtmosLESConfiguration,
                   altitude, latitude, longitude, gravitational_potential
using CLIMA.VariableTemplates: flattenednames

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test

const output_vtk = false

function main()
  CLIMA.init()
  ArrayType = CLIMA.array_type()

  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = Dict("DEBUG" => Logging.Debug,
                  "WARN"  => Logging.Warn,
                  "ERROR" => Logging.Error,
                  "INFO"  => Logging.Info)[ll]

  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  polynomialorder = 5
  numelem_horz = 10
  numelem_vert = 5

  timeend = 60 * 60
  # timeend = 33 * 60 * 60 # Full simulation
  outputtime = 60 * 60

  expected_result = Dict()
  expected_result[Float32] = 9.5064378310656000e+13
  expected_result[Float64] = 9.5073452847081828e+13

  for FT in (Float32, Float64)
    result = run(mpicomm, polynomialorder, numelem_horz, numelem_vert,
                 timeend, outputtime, ArrayType, FT)
    @test result ≈ expected_result[FT]
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

  model = AtmosModel{FT}(AtmosLESConfiguration;
                         orientation=SphericalOrientation(),
                           ref_state=HydrostaticState(IsothermalProfile(setup.T_ref), FT(0)),
                          turbulence=ConstantViscosityWithDivergence(FT(0)),
                            moisture=DryModel(),
                              source=Gravity(),
                          init_state=setup)
  linearmodel = AtmosAcousticGravityLinearModel(model)

  dg = DGModel(model, grid, Rusanov(),
               CentralNumericalFluxDiffusive(), CentralNumericalFluxGradient())

  lineardg = DGModel(linearmodel, grid, Rusanov(),
                     CentralNumericalFluxDiffusive(),
                     CentralNumericalFluxGradient();
                     direction=VerticalDirection(),
                     auxstate=dg.auxstate)

  # determine the time step
  element_size = (setup.domain_height / numelem_vert)
  acoustic_speed = soundspeed_air(FT(setup.T_ref))
  dt_factor = 445
  dt = dt_factor * element_size / acoustic_speed / polynomialorder ^ 2
  # Adjust the time step so we exactly hit 1 hour for VTK output
  dt = 60 * 60 / ceil(60 * 60 / dt)
  nsteps = ceil(Int, timeend / dt)

  Q = init_ode_state(dg, FT(0))

  linearsolver = ManyColumnLU()

  odesolver = ARK2GiraldoKellyConstantinescu(dg, lineardg, linearsolver, Q;
                                             dt = dt, t0 = 0, split_nonlinear_linear=false)

  filterorder = 18
  filter = ExponentialFilter(grid, 0, filterorder)
  cbfilter = EveryXSimulationSteps(1) do
    Filters.apply!(Q, 1:size(Q, 2), grid, filter, VerticalDirection())
    nothing
  end

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
                        """ gettime(odesolver) runtime energy
    end
  end
  callbacks = (cbinfo, cbfilter)

  if output_vtk
    # create vtk dir
    vtkdir = "vtk_acousticwave" *
      "_poly$(polynomialorder)_horz$(numelem_horz)_vert$(numelem_vert)" *
      "_dt$(dt_factor)x_$(ArrayType)_$(FT)"
    mkpath(vtkdir)

    vtkstep = 0
    # output initial step
    do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)

    # setup the output callback
    cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
      vtkstep += 1
      Qe = init_ode_state(dg, gettime(odesolver))
      do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)
    end
    callbacks = (callbacks..., cbvtk)
  end

  solve!(Q, odesolver; numberofsteps=nsteps, adjustfinalstep=false, callbacks=callbacks)

  # final statistics
  engf = norm(Q)
  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  """ engf engf/eng0 engf-eng0
  engf
end

Base.@kwdef struct AcousticWaveSetup{FT}
  domain_height::FT = 10e3
  T_ref::FT = 300
  α::FT = 3
  γ::FT = 100
  nv::Int = 1
end

function (setup::AcousticWaveSetup)(bl, state, aux, coords, t)
  # callable to set initial conditions
  FT = eltype(state)

  λ = longitude(bl.orientation, aux)
  φ = latitude(bl.orientation, aux)
  z = altitude(bl.orientation, aux)

  β = min(FT(1), setup.α * acos(cos(φ) * cos(λ)))
  f = (1 + cos(FT(π) * β)) / 2
  g = sin(setup.nv * FT(π) * z / setup.domain_height)
  Δp = setup.γ * f * g
  p = aux.ref_state.p + Δp

  ts       = PhaseDry_given_pT(p, setup.T_ref)
  q_pt     = PhasePartition(ts)
  e_pot    = gravitational_potential(bl.orientation, aux)
  e_int    = internal_energy(ts)

  state.ρ  = air_density(ts)
  state.ρu = SVector{3, FT}(0, 0, 0)
  state.ρe = state.ρ * (e_int + e_pot)
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
