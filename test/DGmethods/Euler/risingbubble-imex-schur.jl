using CLIMA
using CLIMA.Mesh.Topologies: StackedBrickTopology
using CLIMA.Mesh.Geometry: LocalGeometry
using CLIMA.Mesh.Grids: DiscontinuousSpectralElementGrid
using CLIMA.Mesh.Filters
using CLIMA.DGmethods: DGModel, init_ode_state
using CLIMA.DGmethods.NumericalFluxes: Rusanov,
                                       CentralNumericalFluxNonDiffusive,
                                       CentralGradPenalty,
                                       CentralNumericalFluxDiffusive
using CLIMA.ODESolvers: solve!, gettime
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.AdditiveRungeKuttaMethod
using CLIMA.AdditiveRungeKuttaMethod: NaiveVariant, LowStorageVariant
using CLIMA.GeneralizedMinimalResidualSolver: GeneralizedMinimalResidual
using CLIMA.VTK: writevtk, writepvtu
using CLIMA.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps
using CLIMA.PlanetParameters: R_d, grav, MSLP, planet_radius, cp_d, cv_d, day
using CLIMA.MoistThermodynamics: air_density, total_energy, soundspeed_air, internal_energy, air_temperature
using CLIMA.Atmos
using CLIMA.Atmos: ReferenceState, vars_state
using CLIMA.VariableTemplates: Vars, flattenednames, @vars
import CLIMA.Atmos: atmos_init_aux!
import CLIMA.DGmethods: vars_aux

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test

const output_vtk = true
const ArrayType = CLIMA.array_type()

function main()
  CLIMA.init()
  mpicomm = MPI.COMM_WORLD

  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = Dict("DEBUG" => Logging.Debug,
                  "WARN"  => Logging.Warn,
                  "ERROR" => Logging.Error,
                  "INFO"  => Logging.Info)[ll]

  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  polynomialorder = 4
  dim = 2
  numelem = (10, dim == 2 ? 1 : 10, 10)
  timeend = 500
  outputtime = 25
  
  for FT in (Float64,)
    run(mpicomm, dim, polynomialorder, numelem, timeend, outputtime, FT)
  end
end

function run(mpicomm, dim, polynomialorder, numelem, timeend, outputtime, FT)

  setup = RisingBubbleSetup{FT}(dim=dim)

  brickrange = ntuple(d -> range(FT(0); stop=setup.domain_size[d], length=numelem[d]+1), 3)

  topology = StackedBrickTopology(mpicomm, brickrange; periodicity=(true, true, false))

  grid = DiscontinuousSpectralElementGrid(topology;
                                          FloatType=FT,
                                          DeviceArray=ArrayType,
                                          polynomialorder=polynomialorder)

  model = AtmosModel(FlatOrientation(),
                     RisingBubbleRefState(setup),
                     ConstantViscosityWithDivergence(FT(0)),
                     DryModel(),
                     NoRadiation(),
                     Gravity(),
                     NoFluxBC(),
                     setup)
  
  linearmodel = AtmosAcousticLinearModel(model)
  remaindermodel = RemainderModel(model, (linearmodel,))

  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())

  lineardg = DGModel(linearmodel,
                     grid,
                     #Rusanov(),
                     CentralNumericalFluxNonDiffusive(),
                     CentralNumericalFluxDiffusive(),
                     CentralGradPenalty();
                     auxstate = dg.auxstate)

  remainderdg = DGModel(remaindermodel,
                        grid,
                        Rusanov(),
                        CentralNumericalFluxDiffusive(),
                        CentralGradPenalty();
                        auxstate = dg.auxstate)

  # determine the time step
  element_size = minimum(step.(brickrange))
  acoustic_speed = soundspeed_air(FT(setup.θ_ref))
  #lucas_magic_factor = 14
  lucas_magic_factor = 20
  dt = lucas_magic_factor * element_size / acoustic_speed / polynomialorder ^ 2

  Q = init_ode_state(dg, FT(0))
  #odesolver = LSRK144NiegemannDiehlBusch(dg, Q; dt = dt, t0 = 0)

  schur_complement = AtmosSchurComplement(lineardg, Q)
  linearsolver = GeneralizedMinimalResidual(30, schur_complement.P, 1e-6)

  #schur_complement = NoSchur()
  #linearsolver = GeneralizedMinimalResidual(30, Q, 1e-6)

  odesolver = ARK2GiraldoKellyConstantinescu(remainderdg, lineardg, linearsolver, Q;
                                             dt=dt, t0=0,
                                             split_nonlinear_linear=true,
                                             variant=NaiveVariant(),
                                             schur_complement=schur_complement)

  #filterorder = 14
  #filter = ExponentialFilter(grid, 0, filterorder)
  #cbfilter = EveryXSimulationSteps(1) do
  #  Filters.apply!(Q, 1:size(Q, 2), grid, filter)
  #  nothing
  #end

  eng0 = norm(Q)
  @info @sprintf """Starting
                    ArrayType       = %s
                    FT              = %s
                    polynomialorder = %d
                    numelem         = (%d, %d, %d)
                    dt              = %.16e
                    norm(Q₀)        = %.16e
                    """ ArrayType FT polynomialorder numelem... dt eng0

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
  callbacks = (cbinfo,)

  if output_vtk
    # create vtk dir
    vtkdir = "vtk-risingbubble-dim$dim" *
    "-poly$(polynomialorder)-elem$(join(numelem, "_"))" *
      "-$(ArrayType)-$(FT)"
    mkpath(vtkdir)

    vtkstep = 0
    # output initial step
    do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)

    # setup the output callback
    cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
      vtkstep += 1
      do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)
    end
    callbacks = (callbacks..., cbvtk)
  end

  solve!(Q, odesolver; timeend=timeend, callbacks=callbacks)

  # final statistics
  engf = norm(Q)
  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  """ engf engf/eng0 engf-eng0
end

cosine_perturbation(r::FT, δθ_center::FT) where {FT} = δθ_center * (1 + cospi(r)) / 2

Base.@kwdef struct RisingBubbleSetup{FT}
  dim::Int
  θ_ref::FT = 300
  δθ_center::FT = 1 // 2
  bubble_radius::FT = 250
  domain_size::SVector{3, FT} = (1e3, 1e3, 1e3)
  bubble_center::SVector{3, FT} = (domain_size[1] / 2, domain_size[2] / 2, 350)
end

function (setup::RisingBubbleSetup)(state, aux, coords, t) 
  FT = eltype(state)

  x = coords .- setup.bubble_center
  @inbounds if setup.dim == 2
    r = sqrt(x[1] ^ 2 + x[3] ^ 2) / setup.bubble_radius
  else
    r = sqrt(x[1] ^ 2 + x[2] ^ 2 + x[3] ^ 2) / setup.bubble_radius
  end
  δθ = r <= 1 ? cosine_perturbation(r, setup.δθ_center) : zero(FT)
  
  p = aux.ref_state.p
  θ = setup.θ_ref + δθ
  T = (p / FT(MSLP)) ^ FT(R_d / cp_d) * θ

  state.ρ = air_density(T, p)
  state.ρu = SVector{3, FT}(0, 0, 0)
  state.ρe = state.ρ * (internal_energy(T) + aux.orientation.Φ)
end

struct RisingBubbleRefState{FT} <: ReferenceState
  setup::RisingBubbleSetup{FT}
end
vars_aux(::RisingBubbleRefState, FT) = @vars(ρ::FT, p::FT, T::FT, ρe::FT)
function atmos_init_aux!(ref::RisingBubbleRefState, atmos::AtmosModel, aux::Vars, geom::LocalGeometry)
  FT = eltype(aux)
  @inbounds z = geom.coord[3]
  θ_ref = ref.setup.θ_ref
  
  p = FT(MSLP) * (1 - FT(grav) / (θ_ref * FT(cp_d)) * z) ^ FT(cp_d / R_d)
  T = (p / FT(MSLP)) ^ FT(R_d / cp_d) * θ_ref
  ρ = air_density(T, p)

  aux.ref_state.ρ = ρ
  aux.ref_state.T = T
  aux.ref_state.p = p
  e_kin = FT(0)
  e_pot = aux.orientation.Φ
  aux.ref_state.ρe = ρ * total_energy(e_kin, e_pot, T)
end

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, model, testname = "risingbubble")
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
