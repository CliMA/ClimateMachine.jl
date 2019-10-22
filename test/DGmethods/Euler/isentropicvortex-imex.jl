using CLIMA: haspkg
using CLIMA.Mesh.Topologies: BrickTopology
using CLIMA.Mesh.Grids: DiscontinuousSpectralElementGrid
using CLIMA.DGmethods: DGModel, init_ode_state, LocalGeometry
using CLIMA.DGmethods.NumericalFluxes: Rusanov, CentralGradPenalty,
                                       CentralNumericalFluxDiffusive
using CLIMA.ODESolvers: solve!, gettime
using CLIMA.AdditiveRungeKuttaMethod
using CLIMA.GeneralizedMinimalResidualSolver: GeneralizedMinimalResidual
using CLIMA.VTK: writevtk, writepvtu
using CLIMA.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps
using CLIMA.MPIStateArrays: euclidean_distance
using CLIMA.PlanetParameters: kappa_d
using CLIMA.MoistThermodynamics: air_density, total_energy, internal_energy,
                                 soundspeed_air
using CLIMA.Atmos: AtmosModel,
                   AtmosAcousticLinearModel, AtmosAcousticNonlinearModel,
                   NoOrientation,
                   NoReferenceState, ReferenceState,
                   DryModel, NoRadiation, PeriodicBC,
                   ConstantViscosityWithDivergence, vars_state
using CLIMA.VariableTemplates: @vars, Vars, flattenednames
import CLIMA.Atmos: atmos_init_aux!, vars_aux

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
  
  expected_error[Float64, false, 1] = 2.3225467541870387e+01
  expected_error[Float64, false, 2] = 5.2663709730295070e+00
  expected_error[Float64, false, 3] = 1.2183770894070467e-01
  expected_error[Float64, false, 4] = 2.8660813871243937e-03
  
  expected_error[Float64, true, 1] = 2.3225467618783981e+01
  expected_error[Float64, true, 2] = 5.2663710765946341e+00
  expected_error[Float64, true, 3] = 1.2183771242881866e-01
  expected_error[Float64, true, 4] = 2.8660023410820249e-03

  @testset "$(@__FILE__)" begin
    for ArrayType in ArrayTypes, DFloat in (Float64,), dims in 2
      for split_nonlinear_linear in (false, true)
        let
          split = split_nonlinear_linear ? "(Nonlinear, Linear)" :
                                           "(Full, Linear)"
          @info @sprintf """Configuration
                            ArrayType = %s
                            DFloat    = %s
                            dims      = %d
                            splitting = %s
                            """ "$ArrayType" "$DFloat" dims split
        end

        setup = IsentropicVortexSetup{DFloat}()
        errors = Vector{DFloat}(undef, numlevels)

        for level in 1:numlevels
          numelems = ntuple(dim -> dim == 3 ? 1 : 2 ^ (level - 1) * 5, dims)
          errors[level] =
            run(mpicomm, polynomialorder, numelems, setup, split_nonlinear_linear,
                ArrayType, DFloat, dims, level)

          @test errors[level] ≈ expected_error[DFloat, split_nonlinear_linear, level]
        end

        rates = @. log2(first(errors[1:numlevels-1]) / first(errors[2:numlevels]))
        numlevels > 1 && @info "Convergence rates\n" *
          join(["rate for levels $l → $(l + 1) = $(rates[l])" for l in 1:numlevels-1], "\n")
      end
    end
  end
end

function run(mpicomm, polynomialorder, numelems, setup,
             split_nonlinear_linear, ArrayType, DFloat, dims, level)
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
  

  model = AtmosModel(NoOrientation(),
                     IsentropicVortexReferenceState{DFloat}(setup),
                     ConstantViscosityWithDivergence(DFloat(0)),
                     DryModel(),
                     NoRadiation(),
                     nothing,
                     PeriodicBC(),
                     initialcondition!)

  linear_model = AtmosAcousticLinearModel(model)
  nonlinear_model = AtmosAcousticNonlinearModel(model)

  dg = DGModel(model, grid, Rusanov(), CentralNumericalFluxDiffusive(), CentralGradPenalty())

  dg_linear = DGModel(linear_model,
                      grid, Rusanov(), CentralNumericalFluxDiffusive(), CentralGradPenalty();
                      auxstate=dg.auxstate)

  if split_nonlinear_linear
    dg_nonlinear = DGModel(nonlinear_model,
                           grid, Rusanov(), CentralNumericalFluxDiffusive(), CentralGradPenalty();
                           auxstate=dg.auxstate)
  end

  timeend = DFloat(2 * setup.domain_halflength / setup.translation_speed)

  # determine the time step
  elementsize = minimum(step.(brickrange))
  dt = elementsize / soundspeed_air(setup.T∞) / polynomialorder ^ 2
  nsteps = ceil(Int, timeend / dt)
  dt = timeend / nsteps

  Q = init_ode_state(dg, DFloat(0))
  
  linearsolver = GeneralizedMinimalResidual(10, Q, 1e-10)
  ode_solver = ARK2GiraldoKellyConstantinescu(split_nonlinear_linear ? dg_nonlinear : dg,
                                              dg_linear,
                                              linearsolver,
                                              Q; dt = dt, t0 = 0,
                                              split_nonlinear_linear = split_nonlinear_linear)

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
                        """ gettime(ode_solver) runtime energy
    end
  end
  callbacks = (cbinfo,)

  if output_vtk
    # create vtk dir
    vtkdir = "vtk_isentropicvortex_imex" *
      "_poly$(polynomialorder)_dims$(dims)_$(ArrayType)_$(DFloat)_level$(level)" *
      "_$(split_nonlinear_linear)"
    mkpath(vtkdir)
    
    vtkstep = 0
    # output initial step
    do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model)

    # setup the output callback
    outputtime = timeend
    cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
      vtkstep += 1
      Qe = init_ode_state(dg, gettime(ode_solver))
      do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model)
    end
    callbacks = (callbacks..., cbvtk)
  end

  solve!(Q, ode_solver; timeend=timeend, callbacks=callbacks)

  # final statistics
  Qe = init_ode_state(dg, timeend)
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

struct IsentropicVortexReferenceState{DFloat} <: ReferenceState
  setup::IsentropicVortexSetup{DFloat}
end
vars_aux(::IsentropicVortexReferenceState, DT) = @vars(ρ::DT, ρe::DT, p::DT, T::DT)
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

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model, testname = "isentropicvortex_imex")
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
