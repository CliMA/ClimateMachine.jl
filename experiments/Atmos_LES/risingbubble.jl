using Random
using StaticArrays
using Test

using CLIMA
using CLIMA.Atmos
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates

# ------------------------ Description ------------------------- #
# 1) Dry Rising Bubble (circular potential temperature perturbation)
# 2) Boundaries - `All Walls` : NoFluxBC (Impermeable Walls)
#                               Laterally periodic
# 3) Domain - 2500m[horizontal] x 2500m[horizontal] x 2500m[vertical]
# 4) Timeend - 1000s
# 5) Mesh Aspect Ratio (Effective resolution) 1:1
# 7) Overrides defaults for
#               `init_on_cpu`
#               `solver_type`
#               `sources`
#               `C_smag`
# 8) Default settings can be found in `src/Driver/Configurations.jl`
# ------------------------ Description ------------------------- #

function init_risingbubble!(bl, state, aux, (x,y,z), t)
  FT            = eltype(state)
  R_gas::FT     = R_d
  c_p::FT       = cp_d
  c_v::FT       = cv_d
  γ::FT         = c_p / c_v
  p0::FT        = MSLP

  xc::FT        = 1250
  yc::FT        = 1250
  zc::FT        = 1000
  r             = sqrt((x-xc)^2+(y-yc)^2+(z-zc)^2)
  rc::FT        = 500
  θ_ref::FT     = 300
  Δθ::FT        = 0

  if r <= rc
    Δθ          = FT(5) * cospi(r/rc/2)
  end

  #Perturbed state:
  θ            = θ_ref + Δθ # potential temperature
  π_exner      = FT(1) - grav / (c_p * θ) * z # exner pressure
  ρ            = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density
  q_tot        = FT(0)
  ts           = LiquidIcePotTempSHumEquil(θ, ρ, q_tot)
  q_pt         = PhasePartition(ts)

  ρu           = SVector(FT(0),FT(0),FT(0))

  #State (prognostic) variable assignment
  e_kin        = FT(0)
  e_pot        = gravitational_potential(bl.orientation, aux)
  ρe_tot       = ρ * total_energy(e_kin, e_pot, ts)
  state.ρ      = ρ
  state.ρu     = ρu
  state.ρe     = ρe_tot
  state.moisture.ρq_tot = ρ*q_pt.tot
end

function config_risingbubble(FT, N, resolution, xmax, ymax, zmax)

  # Boundary conditions
  bc = NoFluxBC()

  # Choose explicit solver
  ode_solver = CLIMA.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch)

  # Set up the model
  C_smag = FT(0.23)
  ref_state = HydrostaticState(DryAdiabaticProfile(typemin(FT), FT(300)), FT(0))
  model = AtmosModel{FT}(AtmosLESConfiguration;
                         turbulence=SmagorinskyLilly{FT}(C_smag),
                         source=(Gravity(),),
                         ref_state=ref_state,
                         init_state=init_risingbubble!)

  # Problem configuration
  config = CLIMA.Atmos_LES_Configuration("DryRisingBubble",
                                         N, resolution, xmax, ymax, zmax,
                                         init_risingbubble!,
                                         solver_type=ode_solver,
                                         model=model)
  return config
end

function main()
    CLIMA.init()

    # Working precision
    FT = Float64
    # DG polynomial order
    N = 4
    # Domain resolution and size
    Δh = FT(50)
    Δv = FT(50)
    resolution = (Δh, Δh, Δv)
    # Domain extents
    xmax = 2500
    ymax = 2500
    zmax = 2500
    # Simulation time
    t0 = FT(0)
    timeend = FT(1000)
    # Courant number
    CFL = FT(0.8)

    driver_config = config_risingbubble(FT, N, resolution, xmax, ymax, zmax)
    solver_config = CLIMA.setup_solver(t0, timeend, driver_config, init_on_cpu=true, Courant_number=CFL)

    # User defined filter (TMAR positivity preserving filter)
    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init=false)
        Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
        nothing
    end

    # Invoke solver (calls solve! function for time-integrator)
    result = CLIMA.invoke!(solver_config;
                          user_callbacks=(cbtmarfilter,),
                          check_euclidean_distance=true)

    @test isapprox(result,FT(1); atol=1.5e-3)
end

main()
