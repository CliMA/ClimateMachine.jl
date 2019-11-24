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

# -------------- Problem constants ------------------- # 
const dim = 2
const polynomialorder = 4
const filterorder = 30
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
#const tolerance = sqrt(eps(Float64))
#const tolerance = 1e-3 # ok for full-linear
#const tolerance = 1e-4 # ok for nonlinear-linear with rusanov
const tolerance = 1e-6 # tbd
const smooth_bubble = false
const diffusion = false
const dry = true
const timeend = 500
const output_vtk = false
const outputtime = 25
#const linflux = Rusanov
const linflux = CentralNumericalFluxNonDiffusive

# ------------- Initial condition function ----------- # 
"""
@article{doi:10.1175/1520-0469(1993)050<1865:BCEWAS>2.0.CO;2,
author = {Robert, A},
title = {Bubble Convection Experiments with a Semi-implicit Formulation of the Euler Equations},
journal = {Journal of the Atmospheric Sciences},
volume = {50},
number = {13},
pages = {1865-1873},
year = {1993},
doi = {10.1175/1520-0469(1993)050<1865:BCEWAS>2.0.CO;2},
URL = {https://doi.org/10.1175/1520-0469(1993)050<1865:BCEWAS>2.0.CO;2},
eprint = {https://doi.org/10.1175/1520-0469(1993)050<1865:BCEWAS>2.0.CO;2},
}
"""
function Initialise_Rising_Bubble!(state::Vars, aux::Vars, (x1,x2,x3), t)
  DF            = eltype(state)
  R_gas::DF     = R_d
  c_p::DF       = cp_d
  c_v::DF       = cv_d
  γ::DF         = c_p / c_v
  p0::DF        = MSLP
  
  xc::DF        = 500
  zc::DF        = 260
  if dim == 2
    r             = sqrt((x1 - xc)^2 + (x3 - zc)^2)
  else
    r             = sqrt((x1 - xc)^2 + (x2 - xc)^2 + (x3 - zc)^2)
  end
  rc::DF        = 250
  θ_ref::DF     = 303
  Δθ::DF        = 0
  θ_c::DF = 1 // 2
 
  if smooth_bubble
    a::DF   =  50
    s::DF   = 100
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
  π_exner      = DF(1) - grav / (c_p * θ) * x3 # exner pressure
  ρ            = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density
  P            = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
  T            = P / (ρ * R_gas) # temperature
  ρu           = SVector(DF(0),DF(0),DF(0))
  # energy definitions
  e_kin        = DF(0)
  e_pot        = grav * x3
  ρe_tot       = ρ * total_energy(e_kin, e_pot, T)
  state.ρ      = ρ
  state.ρu     = ρu
  state.ρe     = ρe_tot
  if !dry
    state.moisture.ρq_tot = DF(0)
  end
end
struct RisingBubbleReferenceState <: ReferenceState end
vars_aux(::RisingBubbleReferenceState, DT) = @vars(ρ::DT, p::DT, T::DT, ρe::DT)
function atmos_init_aux!(m::RisingBubbleReferenceState, atmos::AtmosModel, aux::Vars, geom::LocalGeometry)
  x1, x2, x3 = geom.coord
  DF            = eltype(aux)
  R_gas::DF     = R_d
  c_p::DF       = cp_d
  c_v::DF       = cv_d
  γ::DF         = c_p / c_v
  p0::DF        = MSLP
  θ_ref::DF     = 303

  θ            = θ_ref
  π_exner      = DF(1) - grav / (c_p * θ) * x3 # exner pressure
  ρ            = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density
  P            = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
  T            = P / (ρ * R_gas) # temperature
  e_kin        = DF(0)
  e_pot        = grav * x3
  ρe_tot       = ρ * total_energy(e_kin, e_pot, T)

  aux.ref_state.ρ = ρ
  aux.ref_state.ρe = ρe_tot
  aux.ref_state.p = P
  aux.ref_state.T = T
end
# --------------- Driver definition ------------------ # 
function run(mpicomm, ArrayType, 
             topl, dim, Ne, polynomialorder, 
             timeend, DF, dt)
  # -------------- Define grid ----------------------------------- # 
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DF,
                                          DeviceArray = ArrayType,
                                          polynomialorder = polynomialorder
                                         )
  sgs = diffusion ? Vreman{DF}(C_smag) : ConstantViscosityWithDivergence(DF(0))
  moisture = dry ? DryModel() : EquilMoist()
  # -------------- Define model ---------------------------------- # 
  model = AtmosModel(FlatOrientation(),
                     RisingBubbleReferenceState(),
                     #HydrostaticState(IsothermalProfile(DF(T_0)),DF(0)),
                     sgs,
                     moisture, 
                     NoRadiation(),
                     Gravity(),
                     NoFluxBC(),
                     Initialise_Rising_Bubble!)

  fast_model = AtmosAcousticLinearModel(model)
  slow_model = RemainderModel(model, (fast_model,))

  # -------------- Define dgbalancelaw --------------------------- # 
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

  Q = init_ode_state(dg, DF(0))
  Qinit = init_ode_state(dg, DF(-1))

  if schur
    schur_lhs_model = SchurLHSModel()
    schur_rhs_model = SchurRHSModel()
    schur_update_model = SchurUpdateModel()


    schur_lhs_dg = DGModel(schur_lhs_model,
                           grid,
                           nothing,
                           CentralNumericalFluxDiffusive(),
                           CentralGradPenalty())
    
    #nodal_update!(schur_aux_init!, dg.grid,
    #              schur_lhs_model, schur_lhs,
    #              schur_lhs_dg.auxstate,
    #              model, Q, dg.auxstate,
    #              0)
    #grad_auxiliary_state!(schur_lhs_dg, 1, (2, 3, 4))
    
    schur_rhs_dg = DGModel(schur_rhs_model,
                           grid,
                           CentralNumericalFluxNonDiffusive(),
                           CentralNumericalFluxDiffusive(),
                           CentralGradPenalty();
                           auxstate=schur_lhs_dg.auxstate)
    
    schur_update_dg = DGModel(schur_update_model,
                              grid,
                              nothing,
                              CentralNumericalFluxDiffusive(),
                              CentralGradPenalty())
    
    #nodal_update_aux!(schur_aux_init!,
    #                  schur_update_dg, dg,
    #                  schur_update_model, model,
    #                  Q,
    #                  schur_update_dg.auxstate, dg.auxstate, 0)

    linearsolver = GeneralizedMinimalResidual(30, similar(Q, 1), tolerance)
    schur_complement = SchurComplement(schur_lhs_dg, schur_rhs_dg, schur_update_dg) 
  else
    linearsolver = GeneralizedMinimalResidual(30, Q, tolerance)
  end

  ode_solver = ark_scheme(split_nonlinear_linear ? slow_dg : dg,
                          fast_dg, linearsolver, Q; dt = dt, t0 = 0,
                          split_nonlinear_linear=split_nonlinear_linear,
                          variant=variant,
                          schur = schur ? schur_complement : NoSchur())

  callbacks = ()
  if output_vtk
    # create vtk dir
    vtkdir = "vtk-stability-rtb-dim$dim-poly$polynomialorder-dt=$(dt_factor)x" *
             "-ark=$ark_scheme-split_nonlinear_linear=$split_nonlinear_linear-linflux=$linflux-schur=$schur" *
             "-smooth=$smooth_bubble-dry=$dry-diffusion=$diffusion-filter=$filterorder-variant=$variant"
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
  dt        = %.16e
  split_nll = %s
  linflux   = %s
  smooth    = %s
  schur     = %s
  scheme    = %s
  tolerance = %.16e
  dry       = %s
  diffusion = %s
  filter    = %d
  ArrayType = %s
  FloatType = %s
  norm(Q₀)  = %.16e""" dt "$split_nonlinear_linear" "$linflux" "$smooth_bubble" "$schur" "$ark_scheme" tolerance "$dry" "$diffusion" filterorder ArrayType DF eng0

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
  
  if filterorder > 0 
	#filter = ExponentialFilter(grid, 0, filterorder, 8)
	filter = CutoffFilter(grid)
	cbfilter = EveryXSimulationSteps(1) do
	  Filters.apply!(Q, 1:size(Q, 2), grid, filter; horizontal=true, vertical=true)
	  nothing
	end
  	callbacks = (callbacks..., cbinfo, cbfilter)
  else
  	callbacks = (callbacks..., cbinfo)
  end

  solve!(Q, ode_solver; timeend=DF(timeend), callbacks=callbacks)
  # End of the simulation information
  engf = norm(Q)
  Qe = init_ode_state(dg, DF(timeend))
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
  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
    FloatType = (Float64,)
    for DF in FloatType
      brickrange = ntuple(d -> range(DF(domain_start[d]); length=Ne[d]+1, stop=domain_end[d]), 3)
      periodicity = (false, dim == 2 ? true : false, false)
      topl = StackedBrickTopology(mpicomm, brickrange, periodicity = periodicity)
      engf_eng0 = run(mpicomm, ArrayType, 
                      topl, dim, Ne, polynomialorder, 
                      timeend, DF, dt)
      #@test engf_eng0 ≈ DF(9.9999993807738441e-01)
    end
  end
end
