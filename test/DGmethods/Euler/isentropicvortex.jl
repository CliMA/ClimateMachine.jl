using CLIMA: haspkg
using CLIMA.Mesh.Topologies: BrickTopology
using CLIMA.Mesh.Grids: DiscontinuousSpectralElementGrid
using CLIMA.DGmethods: DGModel, init_ode_param, init_ode_state
using CLIMA.DGmethods.NumericalFluxes: Rusanov, DefaultGradNumericalFlux
using CLIMA.ODESolvers: solve!, gettime
using CLIMA.LowStorageRungeKuttaMethod: LSRK54CarpenterKennedy
using CLIMA.VTK: writevtk, writepvtu
using CLIMA.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps
using CLIMA.MPIStateArrays: euclidean_distance
using CLIMA.PlanetParameters: kappa_d
using CLIMA.MoistThermodynamics: air_density, total_energy, soundspeed_air
using CLIMA.Atmos: AtmosModel, FlatOrientation, NoReferenceState,
                   NoViscosity, DryModel, NoRadiation, PeriodicBC

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test
@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray,)
else
  const ArrayTypes = (Array,)
end

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

const output_vtk = false

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
  polynomialorder = 4
  numlevels = integration_testing ? 4 : 1

  expected_error = Dict()

  expected_error[Float64, 2, 1] = 1.1990999506538110e+01
  expected_error[Float64, 2, 2] = 2.0813000228865612e+00
  expected_error[Float64, 2, 3] = 6.3752572004789149e-02
  expected_error[Float64, 2, 4] = 2.0984975076420455e-03

  expected_error[Float64, 3, 1] = 3.7918897283243393e+00
  expected_error[Float64, 3, 2] = 6.5818065500423617e-01
  expected_error[Float64, 3, 3] = 2.0669666996100750e-02
  expected_error[Float64, 3, 4] = 4.6083032825833658e-03

  expected_error[Float32, 2, 1] = 1.1990854263305664e+01
  expected_error[Float32, 2, 2] = 2.0812149047851563e+00
  expected_error[Float32, 2, 3] = 6.7652329802513123e-02
  expected_error[Float32, 2, 4] = 3.6849677562713623e-02

  expected_error[Float32, 3, 1] = 3.7918038368225098e+00
  expected_error[Float32, 3, 2] = 6.5812408924102783e-01
  expected_error[Float32, 3, 3] = 2.1983036771416664e-02
  expected_error[Float32, 3, 4] = 1.1587927117943764e-02

  @testset "$(@__FILE__)" begin
    for ArrayType in ArrayTypes, DFloat in (Float64, Float32), dims in (2, 3)
      @info @sprintf """Configuration
                        ArrayType = %s
                        DFloat    = %s
                        dims      = %d
                        """ "$ArrayType" "$DFloat" dims

      setup = IsentropicVortexSetup{DFloat}()
      errors = Vector{DFloat}(undef, numlevels)

      for level in 1:numlevels
        numelems = ntuple(dim -> dim == 3 ? 1 : 2 ^ (level - 1) * 5, dims)
        errors[level] =
          run(mpicomm, polynomialorder, numelems, setup, ArrayType, DFloat, dims, level)

        rtol = sqrt(eps(DFloat))
        # increase rtol for comparing with GPU results using Float32
        if DFloat === Float32 && !(ArrayType === Array)
          rtol *= 10 # why does this factor have to be so big :(
        end
        @test isapprox(errors[level], expected_error[DFloat, dims, level]; rtol = rtol)
      end

      rates = @. log2(first(errors[1:numlevels-1]) / first(errors[2:numlevels]))
      @info "Convergence rates\n" *
        join(["rate for levels $l → $(l + 1) = $(rates[l])" for l in 1:numlevels-1], "\n")
    end
  end
end

function run(mpicomm, polynomialorder, numelems, setup, ArrayType, DFloat, dims, level)
  brickrange = ntuple(dims) do dim
    range(-setup.domain_halflength; length=numelems[dim] + 1, stop=setup.domain_halflength)
  end

  topology = BrickTopology(mpicomm,
                           brickrange;
                           periodicity=ntuple(_ -> true, dims))

  grid = DiscontinuousSpectralElementGrid(topology,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = polynomialorder)

  initialcondition! = function(args...)
    isentropicvortex_initialcondition!(setup, args...)
  end
  model = AtmosModel(FlatOrientation(),
                     NoReferenceState(),
                     NoViscosity(),
                     DryModel(),
                     NoRadiation(),
                     nothing,
                     PeriodicBC(),
                     initialcondition!)

  dg = DGModel(model, grid, Rusanov(), DefaultGradNumericalFlux())
  param = init_ode_param(dg)

  timeend = DFloat(2 * setup.domain_halflength / 10 / setup.translation_speed)

  # determine the time step
  elementsize = minimum(step.(brickrange))
  dt = elementsize / soundspeed_air(setup.T∞) / polynomialorder ^ 2
  nsteps = ceil(Int, timeend / dt)
  dt = timeend / nsteps

  Q = init_ode_state(dg, param, DFloat(0))
  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  dims == 2 && (numelems = (numelems..., 0))
  @info @sprintf """Starting refinement level %d
                    numelems  = (%d, %d, %d)
                    dt        = %.16e
                    norm(Q₀)  = %.16e
                    """ level numelems... dt eng0

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
    vtkdir = "vtk_isentropicvortex" *
      "_poly$(polynomialorder)_dims$(dims)_$(ArrayType)_$(DFloat)_level$(level)"
    mkpath(vtkdir)

    # output initial step
    do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q)

    # setup the output callback
    outputtime = timeend
    vtkstep = 0
    cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
      vtkstep += 1
      Qe = init_ode_state(dg, param, gettime(lsrk))
      do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe)
    end
    callbacks = (callbacks..., cbvtk)
  end

  solve!(Q, lsrk, param; timeend=timeend, callbacks=callbacks)

  # final statistics
  Qe = init_ode_state(dg, param, timeend)
  engf = norm(Q)
  engfe = norm(Qe)
  errf = euclidean_distance(Q, Qe)
  @info @sprintf """Finished refinement level %d
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  norm(Q - Qe)            = %.16e
  norm(Q - Qe) / norm(Qe) = %.16e
  """ level engf engf/eng0 engf-eng0 errf errf/engfe
  errf
end

Base.@kwdef struct IsentropicVortexSetup{DFloat}
  p∞::DFloat = 10 ^ 5
  T∞::DFloat = 300
  ρ∞::DFloat = air_density(DFloat(T∞), DFloat(p∞))
  translation_speed::DFloat = 150
  translation_angle::DFloat = pi / 4
  vortex_speed::DFloat = 50
  vortex_radius::DFloat = 1 // 200
  domain_halflength::DFloat = 1 // 20
end

function isentropicvortex_initialcondition!(setup, state, aux, coords, t)
  DFloat = eltype(state)
  x = MVector(coords)

  ρ∞ = setup.ρ∞
  p∞ = setup.p∞
  T∞ = setup.T∞
  translation_speed = setup.translation_speed
  α = setup.translation_angle
  vortex_speed = setup.vortex_speed
  R = setup.vortex_radius
  L = setup.domain_halflength

  u∞ = SVector(translation_speed * cos(α), translation_speed * sin(α), 0)

  x .-= u∞ * t
  # make the function periodic
  x .-= floor.((x + L) / 2L) * 2L

  @inbounds begin
    r = sqrt(x[1] ^ 2 + x[2] ^ 2)
    δu_x = -vortex_speed * x[2] / R * exp(-(r / R) ^ 2 / 2)
    δu_y =  vortex_speed * x[1] / R * exp(-(r / R) ^ 2 / 2)
  end
  u = u∞ .+ SVector(δu_x, δu_y, 0)

  T = T∞ * (1 - kappa_d * vortex_speed ^ 2 / 2 * ρ∞ / p∞ * exp(-(r / R) ^ 2))
  # adiabatic/isentropic relation
  p = p∞ * (T / T∞) ^ (DFloat(1) / kappa_d)
  ρ = air_density(T, p)

  state.ρ = ρ
  state.ρu = ρ * u
  e_kin = u' * u / 2
  state.ρe = ρ * total_energy(e_kin, DFloat(0), T)
end

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, testname = "isentropicvortex")
  ## name of the file that this MPI rank will write
  filename = @sprintf("%s/%s_mpirank%04d_step%04d",
                      vtkdir, testname, MPI.Comm_rank(mpicomm), vtkstep)

  statenames = ("ρ", "ρu", "ρv", "ρw", "ρe")
  exactnames = statenames .* "_exact"

  writevtk(filename, Q, dg, statenames, Qe, exactnames)

  ## Generate the pvtu file for these vtk files
  if MPI.Comm_rank(mpicomm) == 0
    ## name of the pvtu file
    pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

    ## name of each of the ranks vtk files
    prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
      @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
    end

    writepvtu(pvtuprefix, prefixes, (statenames..., exactnames...))

    @info "Done writing VTK: $pvtuprefix"
  end
end

main()
