using CLIMA
using CLIMA.Mesh.Topologies: BrickTopology
using CLIMA.Mesh.Grids: DiscontinuousSpectralElementGrid
using CLIMA.DGmethods: DGModel, init_ode_state
using CLIMA.DGmethods.NumericalFluxes: Rusanov, CentralNumericalFluxGradient,
                                       CentralNumericalFluxDiffusive,
                                       CentralNumericalFluxNonDiffusive
using CLIMA.ODESolvers
using CLIMA.VTK: writevtk, writepvtu
using CLIMA.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps
using CLIMA.MPIStateArrays: euclidean_distance
using CLIMA.PlanetParameters: kappa_d
using CLIMA.MoistThermodynamics: air_density, total_energy, soundspeed_air, PhaseDry_given_pT
using CLIMA.Atmos: AtmosModel, NoOrientation, NoReferenceState,
                   DryModel, NoPrecipitation, NoRadiation, PeriodicBC,
                   ConstantViscosityWithDivergence, vars_state,
                   AtmosLESConfiguration
using CLIMA.VariableTemplates: flattenednames

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

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
  polynomialorder = 4
  numlevels = integration_testing ? 4 : 1

  expected_error = Dict()

  # just to make it shorter and aligning
  Central = CentralNumericalFluxNonDiffusive

  expected_error[Float64, 2, Rusanov, 1] = 1.1990999506538110e+01
  expected_error[Float64, 2, Rusanov, 2] = 2.0813000228865612e+00
  expected_error[Float64, 2, Rusanov, 3] = 6.3752572004789149e-02
  expected_error[Float64, 2, Rusanov, 4] = 2.0984975076420455e-03

  expected_error[Float64, 2, Central, 1] = 2.0840574601661153e+01
  expected_error[Float64, 2, Central, 2] = 2.9255455365299827e+00
  expected_error[Float64, 2, Central, 3] = 3.6935849488949657e-01
  expected_error[Float64, 2, Central, 4] = 8.3528804679907434e-03

  expected_error[Float64, 3, Rusanov, 1] = 3.7918869862613858e+00
  expected_error[Float64, 3, Rusanov, 2] = 6.5816485664822677e-01
  expected_error[Float64, 3, Rusanov, 3] = 2.0160333422867591e-02
  expected_error[Float64, 3, Rusanov, 4] = 6.6360317881818034e-04

  expected_error[Float64, 3, Central, 1] = 6.5903683487905749e+00
  expected_error[Float64, 3, Central, 2] = 9.2513872939749997e-01
  expected_error[Float64, 3, Central, 3] = 1.1680141169828175e-01
  expected_error[Float64, 3, Central, 4] = 2.6414127301659534e-03

  expected_error[Float32, 2, Rusanov, 1] = 1.1990854263305664e+01
  expected_error[Float32, 2, Rusanov, 2] = 2.0812149047851563e+00
  expected_error[Float32, 2, Rusanov, 3] = 6.6969044506549835e-02
  expected_error[Float32, 2, Rusanov, 4] = 5.2888177335262299e-02

  expected_error[Float32, 2, Central, 1] = 2.0840496063232422e+01
  expected_error[Float32, 2, Central, 2] = 2.9250388145446777e+00
  expected_error[Float32, 2, Central, 3] = 3.7026408314704895e-01
  expected_error[Float32, 2, Central, 4] = 1.1543836444616318e-01

  expected_error[Float32, 3, Rusanov, 1] = 3.7918324470520020e+00
  expected_error[Float32, 3, Rusanov, 2] = 6.5811443328857422e-01
  expected_error[Float32, 3, Rusanov, 3] = 2.0889002829790115e-02
  expected_error[Float32, 3, Rusanov, 4] = 1.1552370153367519e-02

  expected_error[Float32, 3, Central, 1] = 6.5902600288391113e+00
  expected_error[Float32, 3, Central, 2] = 9.2505264282226563e-01
  expected_error[Float32, 3, Central, 3] = 1.1701638251543045e-01
  expected_error[Float32, 3, Central, 4] = 2.1023442968726158e-02

  @testset "$(@__FILE__)" begin
    for FT in (Float64, Float32), dims in (2, 3)
      for NumericalFlux in (Rusanov, Central)
        @info @sprintf """Configuration
                          ArrayType     = %s
                          FT        = %s
                          NumericalFlux = %s
                          dims          = %d
                          """ ArrayType "$FT" "$NumericalFlux" dims

        setup = IsentropicVortexSetup{FT}()
        errors = Vector{FT}(undef, numlevels)

        for level in 1:numlevels
          numelems = ntuple(dim -> dim == 3 ? 1 : 2 ^ (level - 1) * 5, dims)
          errors[level] =
            run(mpicomm, ArrayType, polynomialorder, numelems,
                NumericalFlux, setup, FT, dims, level)

          rtol = sqrt(eps(FT))
          # increase rtol for comparing with GPU results using Float32
          if FT === Float32 && ArrayType !== Array
            rtol *= 10 # why does this factor have to be so big :(
          end
          @test isapprox(errors[level],
                         expected_error[FT, dims, NumericalFlux, level]; rtol = rtol)
        end

        rates = @. log2(first(errors[1:numlevels-1]) / first(errors[2:numlevels]))
        numlevels > 1 && @info "Convergence rates\n" *
          join(["rate for levels $l → $(l + 1) = $(rates[l])" for l in 1:numlevels-1], "\n")
      end
    end
  end
end

function run(mpicomm, ArrayType, polynomialorder, numelems,
             NumericalFlux, setup, FT, dims, level)
  brickrange = ntuple(dims) do dim
    range(-setup.domain_halflength; length=numelems[dim] + 1, stop=setup.domain_halflength)
  end

  topology = BrickTopology(mpicomm,
                           brickrange;
                           periodicity=ntuple(_ -> true, dims))

  grid = DiscontinuousSpectralElementGrid(topology,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = polynomialorder)

  model = AtmosModel{FT}(AtmosLESConfiguration;
                         orientation=NoOrientation(),
                           ref_state=NoReferenceState(),
                          turbulence=ConstantViscosityWithDivergence(0.0),
                            moisture=DryModel(),
                              source=nothing,
                   boundarycondition=PeriodicBC(),
                          init_state=isentropicvortex_initialcondition!)

  dg = DGModel(model, grid, NumericalFlux(),
               CentralNumericalFluxDiffusive(), CentralNumericalFluxGradient())

  timeend = FT(2 * setup.domain_halflength / 10 / setup.translation_speed)

  # determine the time step
  elementsize = minimum(step.(brickrange))
  dt = elementsize / soundspeed_air(setup.T∞) / polynomialorder ^ 2
  nsteps = ceil(Int, timeend / dt)
  dt = timeend / nsteps

  Q = init_ode_state(dg, FT(0), setup)
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
      "_poly$(polynomialorder)_dims$(dims)_$(ArrayType)_$(FT)_level$(level)"
    mkpath(vtkdir)

    vtkstep = 0
    # output initial step
    do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model)

    # setup the output callback
    outputtime = timeend
    cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
      vtkstep += 1
      Qe = init_ode_state(dg, gettime(lsrk), setup)
      do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model)
    end
    callbacks = (callbacks..., cbvtk)
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=callbacks)

  # final statistics
  Qe = init_ode_state(dg, timeend, setup)
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

Base.@kwdef struct IsentropicVortexSetup{FT}
  p∞::FT = 10 ^ 5
  T∞::FT = 300
  ρ∞::FT = air_density(FT(T∞), FT(p∞))
  translation_speed::FT = 150
  translation_angle::FT = pi / 4
  vortex_speed::FT = 50
  vortex_radius::FT = 1 // 200
  domain_halflength::FT = 1 // 20
end

function isentropicvortex_initialcondition!(bl, state, aux, coords, t, args...)
  setup = first(args)
  FT = eltype(state)
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
  x .-= floor.((x .+ L) / 2L) * 2L

  @inbounds begin
    r = sqrt(x[1] ^ 2 + x[2] ^ 2)
    δu_x = -vortex_speed * x[2] / R * exp(-(r / R) ^ 2 / 2)
    δu_y =  vortex_speed * x[1] / R * exp(-(r / R) ^ 2 / 2)
  end
  u = u∞ .+ SVector(δu_x, δu_y, 0)

  T = T∞ * (1 - kappa_d * vortex_speed ^ 2 / 2 * ρ∞ / p∞ * exp(-(r / R) ^ 2))
  # adiabatic/isentropic relation
  p = p∞ * (T / T∞) ^ (FT(1) / kappa_d)
  ts = PhaseDry_given_pT(p, T)
  ρ = air_density(ts)

  e_pot = FT(0)
  state.ρ = ρ
  state.ρu = ρ * u
  e_kin = u' * u / 2
  state.ρe = ρ * total_energy(e_kin, e_pot, ts)
end

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model, testname = "isentropicvortex")
  ## name of the file that this MPI rank will write
  filename = @sprintf("%s/%s_mpirank%04d_step%04d",
                      vtkdir, testname, MPI.Comm_rank(mpicomm), vtkstep)

  statenames = flattenednames(vars_state(model, eltype(Q)))
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
