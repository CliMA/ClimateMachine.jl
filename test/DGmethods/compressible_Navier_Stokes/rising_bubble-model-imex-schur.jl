# Load Packages 
using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Geometry
using CLIMA.Mesh.Filters
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.AdditiveRungeKuttaMethod
using CLIMA.AdditiveRungeKuttaMethod: NaiveVariant, LowStorageVariant
using CLIMA.SubgridScaleParameters
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks: EveryXSimulationSteps, EveryXWallTimeSeconds
using CLIMA.Atmos
using CLIMA.VariableTemplates
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.GeneralizedMinimalResidualSolver: GeneralizedMinimalResidual
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK
using Random
using CLIMA.Atmos: vars_state, ReferenceState
import CLIMA.Atmos: atmos_init_aux!, vars_aux

const ArrayType = CLIMA.array_type()

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

const FT = Float64
const dim = 2
const polynomialorder = 4
const filter = CutoffFilter
#const filter = (ExponentialFilter, 32)
const domain_start = (0, 0, 0)
const domain_end = (1000, dim == 2 ? 100 : 1000, 1000)
const Ne = (10, dim == 2 ? 1 : 10, 10)
const Δxyz = @. (domain_end - domain_start) / Ne / polynomialorder
const dt_factor = 20
const dt = dt_factor * min(Δxyz...) / soundspeed_air(300.0) / polynomialorder
const ark_scheme = ARK2GiraldoKellyConstantinescu
#const ark_scheme = ARK548L2SA2KennedyCarpenter
#const ark_scheme = ARK437L2SA1KennedyCarpenter
const split_nonlinear_linear = true
#const variant = LowStorageVariant()
const variant = NaiveVariant()
const schur = true
const tolerance = 1e-6
const smooth_bubble = false
const subgridmodel = ConstantViscosityWithDivergence(FT(0))
#const subgridmodel = Vreman{FT}(C_smag)
const moistmodel = EquilMoist()
const timeend = 500
const output_vtk = false
const outputtime = 25
#const linflux = Rusanov
const linflux = CentralNumericalFluxNonDiffusive

function initialize_risingbubble!(state::Vars, aux::Vars, (x1,x2,x3), t)
  FT            = eltype(state)
  R_gas::FT     = R_d
  c_p::FT       = cp_d
  c_v::FT       = cv_d
  γ::FT         = c_p / c_v
  p0::FT        = MSLP
  
  xc::FT        = 500
  zc::FT        = 260
  if dim == 2
    r             = sqrt((x1 - xc)^2 + (x3 - zc)^2)
  else
    r             = sqrt((x1 - xc)^2 + (x2 - xc)^2 + (x3 - zc)^2)
  end
  rc::FT        = 250
  θ_ref::FT     = 303
  Δθ::FT        = 0
  θ_c::FT = 1 // 2
 
  if smooth_bubble
    a::FT   =  50
    s::FT   = 100
    if r <= a
      Δθ = θ_c
    elseif r > a
      Δθ = θ_c * exp(-(r - a)^2 / s^2)
    end
  else
    if r <= rc 
      Δθ          = θ_c
    end
  end
  
  if t < 0
    Δθ = 0
  end

  #Perturbed state:
  θ            = θ_ref + Δθ # potential temperature
  π_exner      = FT(1) - grav / (c_p * θ) * x3 # exner pressure
  ρ            = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density
  P            = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
  T            = P / (ρ * R_gas) # temperature
  ρu           = SVector(FT(0),FT(0),FT(0))
  # energy definitions
  e_kin        = FT(0)
  e_pot        = grav * x3
  ρe_tot       = ρ * total_energy(e_kin, e_pot, T)
  state.ρ      = ρ
  state.ρu     = ρu
  state.ρe     = ρe_tot
  if moistmodel !== DryModel
    state.moisture.ρq_tot = FT(0)
  end
end
struct RisingBubbleReferenceState <: ReferenceState end
vars_aux(::RisingBubbleReferenceState, DT) = @vars(ρ::DT, p::DT, T::DT, ρe::DT)
function atmos_init_aux!(m::RisingBubbleReferenceState, atmos::AtmosModel, aux::Vars, geom::LocalGeometry)
  x1, x2, x3 = geom.coord
  FT            = eltype(aux)
  R_gas::FT     = R_d
  c_p::FT       = cp_d
  c_v::FT       = cv_d
  γ::FT         = c_p / c_v
  p0::FT        = MSLP
  θ_ref::FT     = 303

  θ            = θ_ref
  π_exner      = FT(1) - grav / (c_p * θ) * x3 # exner pressure
  ρ            = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density
  P            = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
  T            = P / (ρ * R_gas) # temperature
  e_kin        = FT(0)
  e_pot        = grav * x3
  ρe_tot       = ρ * total_energy(e_kin, e_pot, T)

  aux.ref_state.ρ = ρ
  aux.ref_state.ρe = ρe_tot
  aux.ref_state.p = P
  aux.ref_state.T = T
end

function run(mpicomm, topl, dim, Ne, polynomialorder, timeend, FT, dt)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = polynomialorder
                                         )

  model = AtmosModel(FlatOrientation(),
                     RisingBubbleReferenceState(),
                     subgridmodel,
                     moistmodel,
                     NoRadiation(),
                     Gravity(),
                     NoFluxBC(),
                     initialize_risingbubble!)

  fast_model = AtmosAcousticLinearModel(model)
  slow_model = RemainderModel(model, (fast_model,))

  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())

  fast_dg = DGModel(fast_model,
                    grid,
                    linflux(),
                    CentralNumericalFluxDiffusive(),
                    CentralGradPenalty();
                    auxstate=dg.auxstate)

  slow_dg = DGModel(slow_model,
                    grid, Rusanov(),
                    CentralNumericalFluxDiffusive(),
                    CentralGradPenalty();
                    auxstate=dg.auxstate)

  Q = init_ode_state(dg, FT(0))
  Qinit = init_ode_state(dg, FT(-1))

  if schur
    schur_complement = AtmosSchurComplement(fast_dg, Q)
    linearsolver = GeneralizedMinimalResidual(30, schur_complement.P, tolerance)
  else
    linearsolver = GeneralizedMinimalResidual(30, Q, tolerance)
    schur_complement = NoSchur()
  end

  ode_solver = ark_scheme(split_nonlinear_linear ? slow_dg : dg,
                          fast_dg, linearsolver, Q; dt = dt, t0 = 0,
                          split_nonlinear_linear=split_nonlinear_linear,
                          variant=variant,
                          schur_complement = schur_complement)

  callbacks = ()
  if output_vtk
    options = []
    push!(options, "D$dim")
    push!(options, "P$polynomialorder")
    push!(options, "DT$dt_factor")
    shortts = Dict(ARK2GiraldoKellyConstantinescu => "ARK2",
                   ARK548L2SA2KennedyCarpenter => "ARK5",
                   ARK437L2SA1KennedyCarpenter => "ARK4")
    push!(options, shortts[ark_scheme])
    push!(options, split_nonlinear_linear ? "NL" : "FL")
    push!(options, linflux isa Rusanov ? "RV" : "CT")
    push!(options, schur ? "SCH" : "NSCH")
    push!(options, smooth_bubble ? "BS" : "BD")
    push!(options, moistmodel === DryModel ? "DRY" : "MST")
    if subgridmodel isa ConstantViscosityWithDivergence
      shortsubgrid = "V0"
    elseif subgridmodel isa Vreman
      shortsubgrid = "VV"
    end
    if filter === CutoffFilter
      shortfilter = "COF"
    else
      shortfilter = "EXP$(filter[2])"
    end
    push!(options, shortfilter)
    push!(options, variant isa LowStorageVariant ? "LSV" : "NVV")

    vtkdir = "vtk-" * join(options, "-")
    mkpath(vtkdir)
    
    file = MPI.Comm_rank(mpicomm) == 0 ? "$vtkdir/log.txt" : "/dev/null"
    io = open(file, "w+")
    logger = SimpleLogger(io)
    global_logger(logger)
    
    vtkstep = 0
    # output initial step
    Qdiff = Q .- Qinit
    do_output(mpicomm, vtkdir, vtkstep, dg, Qdiff, model)

    # setup the output callback
    cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
      vtkstep += 1
      Qdiff = Q .- Qinit
      do_output(mpicomm, vtkdir, vtkstep, dg, Qdiff, model)
    end
    callbacks = (callbacks..., cbvtk)
  end

  eng0 = norm(Q)
  @info @sprintf """Starting
  dt           = %.16e
  split_nll    = %s
  linflux      = %s
  smooth       = %s
  schur        = %s
  scheme       = %s
  tolerance    = %.16e
  moistmodel   = %s
  subgridmodel = %s
  filter       = %s
  ArrayType    = %s
  FloatType    = %s
  norm(Q₀)     = %.16e""" dt "$split_nonlinear_linear" "$linflux" "$smooth_bubble" "$schur" "$ark_scheme" tolerance "$moistmodel" "$subgridmodel" "$filter" ArrayType FT eng0

  # Set up the information callback (output field dump is via vtk callback: see cbinfo)
  starttime = Ref(now())
  cbinfo = EveryXWallTimeSeconds(10, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", ODESolvers.gettime(ode_solver),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
      output_vtk && flush(io)
    end
  end
  
  if filter !== nothing
    if filter === CutoffFilter
      f = filter(grid)
    else
      filtertype, filterorder = filter
      f = filtertype(grid, filterorder)
    end
    cbfilter = EveryXSimulationSteps(1) do
      Filters.apply!(Q, 1:size(Q, 2), grid, f; horizontal=true, vertical=true)
      nothing
    end
    callbacks = (callbacks..., cbinfo, cbfilter)
  end
  callbacks = (callbacks..., cbinfo, cbfilter)

  solve!(Q, ode_solver; timeend=FT(timeend), callbacks=callbacks)
  # End of the simulation information
  engf = norm(Q)
  Qe = init_ode_state(dg, FT(timeend))
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
  output_vtk && close(io)
end

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, model, testname = "rtb")
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

    writepvtu(pvtuprefix, prefixes, statenames)

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
  brickrange = ntuple(d -> range(FT(domain_start[d]); length=Ne[d]+1, stop=domain_end[d]), 3)
  periodicity = (false, dim == 2 ? true : false, false)
  topl = StackedBrickTopology(mpicomm, brickrange, periodicity = periodicity)
  engf_eng0 = run(mpicomm, topl, dim, Ne, polynomialorder, timeend, FT, dt)
end
