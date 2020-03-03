using Random
using StaticArrays
using Test
using Dates
using Printf

using CLIMA
using CLIMA.Atmos
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates
using CLIMA.Courant
using Printf

import CLIMA.DGmethods: courant
import CLIMA.Grids: VerticalDirection, HorizontalDirection

# ------------------------ Description ------------------------- #
# 1) Dry Rising Bubble (circular potential temperature perturbation)
# 2) Boundaries - `All Walls` : NoFluxBC (Impermeable Walls)
#                               Laterally periodic
# 3) Domain - 2500m[horizontal] x 2500m[horizontal] x 2500m[vertical]
# 4) Timeend - 1000s
# 5) Mesh Aspect Ratio (Effective resolution) 1:1
# 7) Overrides defaults for
#               `forcecpu`
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
  ode_solver = CLIMA.MRRKSolverType(solver_method=MultirateRungeKutta,
                                    slow_method=LSRK144NiegemannDiehlBusch,
                                    fast_method=LSRK54CarpenterKennedy,
                                    numsubsteps=100,
                                    linear_model=AtmosAcousticGravityLinearModel)
  # ode_solver = CLIMA.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch)

  # Set up the model
  vis = FT(75.0)
  ref_state = HydrostaticState(DryAdiabaticProfile(typemin(FT), FT(300)), FT(0))
  model = AtmosModel{FT}(AtmosLESConfiguration;
                         turbulence=ConstantViscosityWithDivergence{FT}(vis),
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
    Δh = FT(200)
    Δv = FT(100)
    resolution = (Δh, Δh, Δv)
    # Domain extents
    xmax = 2500
    ymax = 2500
    zmax = 2500
    # Simulation time
    t0 = FT(0)
    timeend = FT(200)
    # Courant number
    CFL = FT(30)

    driver_config = config_risingbubble(FT, N, resolution, xmax, ymax, zmax)
    solver_config = CLIMA.setup_solver(t0, timeend, driver_config,
                                       forcecpu=true, Courant_number=CFL)

    # User defined filter (TMAR positivity preserving filter)
    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init=false)
        Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
        nothing
    end

    cbcourantnumbers = GenericCallbacks.EveryXSimulationSteps(5) do
        dg =  solver_config.dg
        m = dg.balancelaw
        Q = solver_config.Q
        Δt = solver_config.dt
        cfl_v = courant(nondiffusive_courant, dg, m, Q, Δt, VerticalDirection())
        cfl_h = courant(nondiffusive_courant, dg, m, Q, Δt, HorizontalDirection())
        cfla_v = courant(advective_courant, dg, m, Q, Δt, VerticalDirection())
        cfla_h = courant(advective_courant, dg, m, Q, Δt, HorizontalDirection())

        fΔt = solver_config.solver.fast_solver.dt
        cflin_v = courant(nondiffusive_courant, dg, m, Q, fΔt, VerticalDirection())
        cflin_h = courant(nondiffusive_courant, dg, m, Q, fΔt, HorizontalDirection())
        cflain_v = courant(advective_courant, dg, m, Q, fΔt, VerticalDirection())
        cflain_h = courant(advective_courant, dg, m, Q, fΔt, HorizontalDirection())

        @info @sprintf """
        ================================
        Courant numbers:

        Outer (slow) method:
        Vertical Acoustic CFL    = %.2g
        Horizontal Acoustic CFL  = %.2g
        Vertical Advection CFL   = %.2g
        Horizontal Advection CFL = %.2g

        Inner (fast) method:
        Vertical Acoustic CFL    = %.2g
        Horizontal Acoustic CFL  = %.2g
        Vertical Advection CFL   = %.2g
        Horizontal Advection CFL = %.2g
        ================================
        """  cfl_v cfl_h cfla_v cfla_h cflin_v cflin_h cflain_v cflain_h

        return nothing
    end

    # Invoke solver (calls solve! function for time-integrator)
    starttime = Base.time()
    result = CLIMA.invoke!(solver_config;
                           user_callbacks=(cbtmarfilter, cbcourantnumbers),
                           check_euclidean_distance=true)
    endtime = Base.time()
    @info @sprintf("""FINISHED. Runtime = %s""", endtime - starttime)

    # @test isapprox(result,FT(1); atol=1.5e-3)
end

main()
