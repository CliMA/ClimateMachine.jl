using Distributions
using Random
using StaticArrays
using Test
using Printf

using CLIMA
using CLIMA.Atmos
using CLIMA.GenericCallbacks
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates

const ArrayType = CLIMA.array_type()

"""
  Initial Condition for dry Rayleigh-Benard convection with prescribed surface temperatures
"""
# ------------- Initial condition function ----------- #
function init_problem!(state::Vars, aux::Vars, (x1,x2,x3), t)
  FT            = eltype(state)
  R_gas::FT     = R_d
  c_p::FT       = cp_d
  c_v::FT       = cv_d
  γ::FT         = c_p / c_v
  p0::FT        = MSLP
  zmin          = FT(0)
  zmax          = FT(1000)
  δT            = sinpi(6*x3/(zmax-zmin)) * cospi(6*x3/(zmax-zmin))
  δw            = sinpi(6*x3/(zmax-zmin)) * cospi(6*x3/(zmax-zmin))
  T_bot         = FT(299)
  T_lapse       = - grav / c_p
  T_top         = T_bot + T_lapse*zmax
  ΔT            = T_lapse * x3 + δT
  T             = T_bot + ΔT
  P             = p0*(T/T_bot)^(grav/R_gas/T_lapse)
  ρ             = P / (R_gas * T)
  ρu, ρv, ρw    = FT(0) , FT(0) , ρ * δw
  E_int         = ρ * c_v * (T-T_0)
  E_pot         = ρ * grav * x3
  E_kin         = ρ * FT(1/2) * δw^2
  ρe_tot        = E_int + E_pot + E_kin
  state.ρ       = ρ
  state.ρu      = SVector(ρu, ρv, ρw)
  state.ρe      = ρe_tot
  state.moisture.ρq_tot = FT(0)
end

# -------------------- Configuration ---------------------- # 
function config_problem(FT, N, resolution, xmax, ymax, zmax)
    # Reference state
    T_min       = FT(289)
    T_s         = FT(290.4)
    Γ_lapse     = FT(grav/cp_d)
    T           = LinearTemperatureProfile(T_min, T_s, Γ_lapse)
    rel_hum     = FT(0)
    ref_state   = HydrostaticState(T, rel_hum)
    T_bot       = FT(299)
    T_lapse     = grav / cp_d
    T_top       = T_bot - T_lapse*zmax
    bc = RayleighBenardBC{FT}(T_bot, T_top)
    C_smag = FT(0.18)
    config = CLIMA.LES_Configuration("DryRayleighBenard", N, resolution, xmax, ymax, zmax,
                                     init_problem!,
                                     solver_type=CLIMA.ExplicitSolverType(LSRK144NiegemannDiehlBusch),
                                     ref_state=ref_state,
                                     C_smag=C_smag,
                                     moisture=EquilMoist(),
                                     radiation=NoRadiation(),
                                     subsidence=NoSubsidence{FT}(),
                                     sources=Gravity(),
                                     bc=bc)
    return config
end

# -------------------- Run Problem ------------------- #
function main()
    CLIMA.init()
    FT = Float64
    
    # -------------- Domain parameters ------------- # 
    # DG polynomial order
    N = 4
    # Domain resolution and size
    Δh = FT(50)
    Δv = FT(50)
    resolution = (Δh, Δh, Δv)
    # LES domain size
    xmax = 1000
    ymax = 1000
    zmax = 1000

    # -------------- Timestep parameters ------------- # 
    # Simulation start, end times
    t0 = FT(0)
    timeend = FT(1000)

    driver_config = config_problem(FT, N, resolution, xmax, ymax, zmax)
    solver_config = CLIMA.setup_solver(t0, timeend, driver_config, forcecpu=true)

    cb_tmarfilter = GenericCallbacks.EveryXSimulationSteps(2) do (init=false)
        Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
        nothing
    end
    
    cb_conservation = GenericCallbacks.EveryXSimulationSteps(500) do (init=false)
      @info @sprintf("""Finished
                     norm(ρ)             = %.16e
                     norm(ρu)            = %.16e
                     norm(ρv)            = %.16e
                     norm(ρw)            = %.16e
                     norm(ρe)            = %.16e""",
                     norm(solver_config.Q[:,1,:]),
                     norm(solver_config.Q[:,2,:]),
                     norm(solver_config.Q[:,3,:]),
                     norm(solver_config.Q[:,4,:]),
                     norm(solver_config.Q[:,5,:]))
      nothing
    end

    result = CLIMA.invoke!(solver_config;
                          user_callbacks=(cbtmarfilter,cb_conservation),
                          check_euclidean_distance=true)

    @testset begin
        @test result ≈ FT(0.9999734954176608)
    end
end

main()
