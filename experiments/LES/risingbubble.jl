using Random
using StaticArrays
using Test

using CLIMA
using CLIMA.Atmos
using CLIMA.GenericCallbacks
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates

# ------------------------ Description ------------------------- # 
# 1) Dry Rising Bubble (Closed-box configuration)
# 2) Boundaries - `Top`   : Prescribed temperature, no-slip
#                 `Bottom`: Prescribed temperature, no-slip 
# 3) Domain - 1000m[horizontal] x 1000m[horizontal] x 1000m[vertical] 
# 4) Timeend - 1000s
# 5) Mesh Aspect Ratio (Effective resolution) 5:1
# 6) Random seed in initial condition (Requires `forcecpu=true` argument)
# 7) Overrides defaults for 
#               `forcecpu`
#               `solver_type`
#               `sources`
#               `C_smag`
# 8) Default settings can be found in src/Driver/Configurations.jl
# ------------------------ Description ------------------------- # 

function init_surfacebubble!(state, aux, (x,y,z), t)
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
  P            = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
  T            = P / (ρ * R_gas) # temperature
  ρu           = SVector(FT(0),FT(0),FT(0))
  # energy definitions
  e_kin        = FT(0)
  e_pot        = grav * z
  ρe_tot       = ρ * total_energy(e_kin, e_pot, T)
  state.ρ      = ρ
  state.ρu     = ρu
  state.ρe     = ρe_tot
  state.moisture.ρq_tot = FT(0)
end

function config_surfacebubble(FT, N, resolution, xmax, ymax, zmax)
    
  # Boundary conditions
  bc = NoFluxBC()

  # Choose IMEX Solver as the default
  ode_solver = CLIMA.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch)

  config = CLIMA.LES_Configuration("DryRisingBubble", 
                                   N, resolution, xmax, ymax, zmax,
                                   init_surfacebubble!,
                                   solver_type=ode_solver,
                                   C_smag = FT(0.23),
                                   moisture=EquilMoist(),
                                   sources=Gravity(),
                                   bc=bc)
  return config
end

function main()
    CLIMA.init()
    FT = Float64
    # DG polynomial order
    N = 4
    # Domain resolution and size
    Δh = FT(50)
    Δv = FT(50)
    resolution = (Δh, Δh, Δv)
    xmax = 2500
    ymax = 2500
    zmax = 2500
    t0 = FT(0)
    timeend = FT(1000)
    CFL = FT(0.8)
    
    driver_config = config_surfacebubble(FT, N, resolution, xmax, ymax, zmax)
    solver_config = CLIMA.setup_solver(t0, timeend, driver_config, forcecpu=true, Courant_number=CFL)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init=false)
        Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
        nothing
    end

    result = CLIMA.invoke!(solver_config;
                          user_callbacks=(cbtmarfilter,),
                          check_euclidean_distance=true)

    @test isapprox(result,FT(1); atol=1.5e-3)
end

main()
