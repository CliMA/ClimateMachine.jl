using CLIMA
using CLIMA.Mesh.Topologies: BrickTopology
using CLIMA.Mesh.Grids: DiscontinuousSpectralElementGrid
using CLIMA.DGmethods: DGModel, init_ode_state, LocalGeometry
using CLIMA.DGmethods.NumericalFluxes: Rusanov, CentralNumericalFluxGradient,
                                       CentralNumericalFluxDiffusive
using CLIMA.ODESolvers
using CLIMA.GeneralizedMinimalResidualSolver: GeneralizedMinimalResidual
using CLIMA.VTK: writevtk, writepvtu
using CLIMA.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps
using CLIMA.MPIStateArrays: euclidean_distance
using CLIMA.PlanetParameters: kappa_d
using CLIMA.MoistThermodynamics: air_density, total_energy, internal_energy,
                                 soundspeed_air
using CLIMA.Atmos: AtmosModel,
                   AtmosAcousticLinearModel, RemainderModel,
                   NoOrientation,
                   NoReferenceState, ReferenceState,
                   DryModel, NoPrecipitation, NoRadiation, PeriodicBC,
                   ConstantViscosityWithDivergence, vars_state,
                   AtmosLESConfiguration
using CLIMA.VariableTemplates: @vars, Vars, flattenednames
import CLIMA.Atmos: atmos_init_aux!, vars_aux

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
  expected_error[Float64, SSPRK33ShuOsher, 1] = 2.3222373077778794e+01
  expected_error[Float64, SSPRK33ShuOsher, 2] = 5.2782503174265516e+00
  expected_error[Float64, SSPRK33ShuOsher, 3] = 1.2281763287878383e-01
  expected_error[Float64, SSPRK33ShuOsher, 4] = 2.3761870907666096e-03

  expected_error[Float64, ARK2GiraldoKellyConstantinescu, 1] = 2.3245216640111998e+01
  expected_error[Float64, ARK2GiraldoKellyConstantinescu, 2] = 5.2626584944153949e+00
  expected_error[Float64, ARK2GiraldoKellyConstantinescu, 3] = 1.2324230746483673e-01
  expected_error[Float64, ARK2GiraldoKellyConstantinescu, 4] = 3.8777995619211627e-03

  @testset "$(@__FILE__)" begin
    for FT in (Float64,), dims in 2
      for FastMethod in (SSPRK33ShuOsher, ARK2GiraldoKellyConstantinescu)
        @info @sprintf """Configuration
                          ArrayType  = %s
                          FastMethod = %s
                          FT     = %s
                          dims       = %d
                          """ ArrayType "$FastMethod" "$FT" dims

        setup = IsentropicVortexSetup{FT}()
        errors = Vector{FT}(undef, numlevels)

        for level in 1:numlevels
          numelems = ntuple(dim -> dim == 3 ? 1 : 2 ^ (level - 1) * 5, dims)
          errors[level] =
            run(mpicomm, ArrayType, polynomialorder, numelems, setup,
                FT, FastMethod, dims, level)

          @test errors[level] ≈ expected_error[FT, FastMethod, level]
        end

        rates = @. log2(first(errors[1:numlevels-1]) / first(errors[2:numlevels]))
        numlevels > 1 && @info "Convergence rates\n" *
          join(["rate for levels $l → $(l + 1) = $(rates[l])" for l in 1:numlevels-1], "\n")
      end
    end
  end
end

function run(mpicomm, ArrayType, polynomialorder, numelems, setup,
             FT, FastMethod, dims, level)
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
                           ref_state=IsentropicVortexReferenceState{FT}(setup),
                          turbulence=ConstantViscosityWithDivergence(FT(0)),
                            moisture=DryModel(),
                              source=nothing,
                   boundarycondition=PeriodicBC(),
                          init_state=isentropicvortex_initialcondition!)
  # The linear model has the fast time scales
  fast_model = AtmosAcousticLinearModel(model)
  # The nonlinear model has the slow time scales
  slow_model = RemainderModel(model, (fast_model,))

  dg = DGModel(model, grid, Rusanov(), CentralNumericalFluxDiffusive(),
               CentralNumericalFluxGradient())
  fast_dg = DGModel(fast_model,
                    grid, Rusanov(), CentralNumericalFluxDiffusive(),
                    CentralNumericalFluxGradient();
                    auxstate=dg.auxstate)
  slow_dg = DGModel(slow_model,
                    grid, Rusanov(), CentralNumericalFluxDiffusive(),
                    CentralNumericalFluxGradient();
                    auxstate=dg.auxstate)

  timeend = FT(2 * setup.domain_halflength / setup.translation_speed)
  # determine the slow time step
  elementsize = minimum(step.(brickrange))
  slow_dt = 8 * elementsize / soundspeed_air(setup.T∞) / polynomialorder ^ 2
  nsteps = ceil(Int, timeend / slow_dt)
  slow_dt = timeend / nsteps

  # arbitrary and not needed for stability, just for testing
  fast_dt = slow_dt / 3

  Q = init_ode_state(dg, FT(0), setup)

  slow_ode_solver = LSRK144NiegemannDiehlBusch(slow_dg, Q; dt = slow_dt)

  # check if FastMethod is ARK, is there a better way ?
  if FastMethod == ARK2GiraldoKellyConstantinescu

    linearsolver = GeneralizedMinimalResidual(Q; M=10, rtol=1e-10)
    # splitting the fast part into full and linear but the fast part
    # is already linear so full_dg == linear_dg == fast_dg
    fast_ode_solver = FastMethod(fast_dg, fast_dg, linearsolver, Q; dt = fast_dt, paperversion = true)
  else
    fast_ode_solver = FastMethod(fast_dg, Q; dt = fast_dt)
  end

  ode_solver = MultirateRungeKutta((slow_ode_solver, fast_ode_solver))

  eng0 = norm(Q)
  dims == 2 && (numelems = (numelems..., 0))
  @info @sprintf """Starting refinement level %d
                    numelems  = (%d, %d, %d)
                    slow_dt   = %.16e
                    fast_dt   = %.16e
                    norm(Q₀)  = %.16e
                    """ level numelems... slow_dt fast_dt eng0

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
                        """ gettime(ode_solver) runtime energy
    end
  end
  callbacks = (cbinfo,)

  if output_vtk
    # create vtk dir
    vtkdir = "vtk_isentropicvortex_multirate" *
      "_poly$(polynomialorder)_dims$(dims)_$(ArrayType)_$(FT)" *
      "_$(FastMethod)_level$(level)"
    mkpath(vtkdir)

    vtkstep = 0
    # output initial step
    do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model)

    # setup the output callback
    outputtime = timeend
    cbvtk = EveryXSimulationSteps(floor(outputtime / slow_dt)) do
      vtkstep += 1
      Qe = init_ode_state(dg, gettime(ode_solver), setup)
      do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model)
    end
    callbacks = (callbacks..., cbvtk)
  end

  solve!(Q, ode_solver; timeend=timeend, callbacks=callbacks)

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

struct IsentropicVortexReferenceState{FT} <: ReferenceState
  setup::IsentropicVortexSetup{FT}
end
vars_aux(::IsentropicVortexReferenceState, FT) = @vars(ρ::FT, ρe::FT, p::FT, T::FT)
function atmos_init_aux!(m::IsentropicVortexReferenceState, atmos::AtmosModel, aux::Vars, geom::LocalGeometry)
  setup = m.setup
  ρ∞ = setup.ρ∞
  p∞ = setup.p∞
  T∞ = setup.T∞

  aux.ref_state.ρ = ρ∞
  aux.ref_state.p = p∞
  aux.ref_state.T = T∞
  aux.ref_state.ρe = ρ∞ * internal_energy(T∞)
end

function isentropicvortex_initialcondition!(bl, state, aux, coords, t, setup)
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
  ρ = air_density(T, p)

  state.ρ = ρ
  state.ρu = ρ * u
  e_kin = u' * u / 2
  state.ρe = ρ * total_energy(e_kin, FT(0), T)
end

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model,
                   testname = "isentropicvortex_mutirate")
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
