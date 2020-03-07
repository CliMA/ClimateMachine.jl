using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using Printf

using CLIMA
using CLIMA.Atmos
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates

import CLIMA.DGmethods: boundary_state!
import CLIMA.Atmos: atmos_boundary_state!
import CLIMA.DGmethods.NumericalFluxes: boundary_flux_diffusive!

# ------------------- Description ---------------------------------------- #
# 1) Dry Rayleigh Benard Convection (re-entrant channel configuration)
# 2) Boundaries - `Sides` : Periodic (Default `bctuple` used to identify bot,top walls)
#                 `Top`   : Prescribed temperature, no-slip
#                 `Bottom`: Prescribed temperature, no-slip
# 3) Domain - 250m[horizontal] x 250m[horizontal] x 500m[vertical]
# 4) Timeend - 1000s
# 5) Mesh Aspect Ratio (Effective resolution) 1:1
# 6) Random seed in initial condition (Requires `init_on_cpu=true` argument)
# 7) Overrides defaults for
#               `C_smag`
#               `Courant_number`
#               `init_on_cpu`
#               `ref_state`
#               `solver_type`
#               `bc`
#               `sources`
# 8) Default settings can be found in src/Driver/Configurations.jl

# ------------------- Begin Boundary Conditions -------------------------- #
"""
  FixedTempNoSlip <: BoundaryCondition

Fixed temperature prescription at top and bottom walls
No slip velocity boundary conditions.
Y ≡ state
Σ ≡ diff
A ≡ aux
⁺ and ⁻ refer to exterior / interior faces

# Fields
$(DocStringExtensions.FIELDS)
"""
struct FixedTempNoSlip{FT} <: BoundaryCondition
  "Prescribed bottom wall temperature `[K]`"
  T_bot::FT
  "Prescribed top wall temperature `[K]`"
  T_top::FT
end
# Rayleigh-Benard problem with two fixed walls (prescribed temperatures)
function atmos_boundary_state!(::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                               bc::FixedTempNoSlip,
                               m::AtmosModel,
                               Y⁺::Vars, A⁺::Vars,
                               n⁻,
                               Y⁻::Vars, A⁻::Vars,
                               bctype, t,_...)
  # Dry Rayleigh Benard Convection
  FT = eltype(Y⁺)
  @inbounds begin
    Y⁺.ρu = -Y⁻.ρu
    if bctype == 1
      E_int⁺ = Y⁺.ρ * cv_d * (bc.T_bot - T_0)
    else
      E_int⁺ = Y⁺.ρ * cv_d * (bc.T_top - T_0)
    end
    E_bc = (E_int⁺ + Y⁺.ρ * A⁺.coord[3] * grav)
    Y⁺.ρe = E_bc
  end
end
function atmos_boundary_flux_diffusive!(::CentralNumericalFluxDiffusive,
                                        bc::FixedTempNoSlip,
                                        m::AtmosModel, F,
                                        Y⁺::Vars, Σ⁺::Vars, ::Vars, A⁺::Vars,
                                        n⁻,
                                        Y⁻::Vars, Σ⁻::Vars, ::Vars, A⁻::Vars,
                                        bctype, t, _...)
  Σ⁺.∇h_tot = -Σ⁻.∇h_tot
end

boundary_state!(nf, m::AtmosModel, x...) =
  atmos_boundary_state!(nf, m.boundarycondition, m, x...)
boundary_flux_diffusive!(nf::NumericalFluxDiffusive,
                         atmos::AtmosModel,
                         F,
                         state⁺, diff⁺, hyperdiff⁺, aux⁺, n⁻,
                         state⁻, diff⁻, hyperdiff⁻, aux⁻,
                         bctype, t,
                         state1⁻, diff1⁻, aux1⁻) =
  atmos_boundary_flux_diffusive!(nf, atmos.boundarycondition, atmos,
                                 F,
                                 state⁺, diff⁺, hyperdiff⁺, aux⁺, n⁻,
                                 state⁻, diff⁻, hyperdiff⁻, aux⁻,
                                 bctype, t,
                                 state1⁻, diff1⁻, aux1⁻)
# ------------------- End Boundary Conditions -------------------------- #

const randomseed = MersenneTwister(1)

struct DryRayleighBenardConvectionDataConfig{FT}
  xmin::FT
  ymin::FT
  zmin::FT
  xmax::FT
  ymax::FT
  zmax::FT
  T_bot::FT
  T_lapse::FT
  T_top::FT
end

function init_problem!(bl, state, aux, (x,y,z), t)
  dc = bl.data_config
  FT            = eltype(state)
  R_gas::FT     = R_d
  c_p::FT       = cp_d
  c_v::FT       = cv_d
  γ::FT         = c_p / c_v
  p0::FT        = MSLP
  δT            = sinpi(6*z/(dc.zmax-dc.zmin)) * cospi(6*z/(dc.zmax-dc.zmin)) + rand(randomseed)
  δw            = sinpi(6*z/(dc.zmax-dc.zmin)) * cospi(6*z/(dc.zmax-dc.zmin)) + rand(randomseed)
  ΔT            = grav/cp_d * z + δT
  T             = dc.T_bot - ΔT
  P             = p0*(T/dc.T_bot)^(grav/R_gas/dc.T_lapse)
  ρ             = P / (R_gas * T)

  q_tot = FT(0)
  e_pot = gravitational_potential(bl.orientation, aux)
  ts = TemperatureSHumEquil(T, P, q_tot)

  ρu, ρv, ρw    = FT(0) , FT(0) , ρ * δw

  e_int         = internal_energy(ts)
  e_kin         = FT(1/2) * δw^2

  ρe_tot        = ρ * (e_int + e_pot + e_kin)
  state.ρ       = ρ
  state.ρu      = SVector(ρu, ρv, ρw)
  state.ρe      = ρe_tot
  state.moisture.ρq_tot = FT(0)
end

function config_problem(FT, N, resolution, xmax, ymax, zmax)

    # Boundary conditions
    T_bot = FT(299)
    T_lapse = FT(grav/cp_d)
    T_top = T_bot - T_lapse*zmax

    bc = FixedTempNoSlip{FT}(T_bot, T_top)

    # Turbulence
    C_smag = FT(0.23)
    data_config = DryRayleighBenardConvectionDataConfig{FT}(0, 0, 0,
                                                            xmax, ymax, zmax,
                                                            T_bot,
                                                            T_lapse,
                                                            FT(T_bot - T_lapse*zmax))

    # Set up the model
    model = AtmosModel{FT}(AtmosLESConfiguration;
                           turbulence=SmagorinskyLilly{FT}(C_smag),
                               source=(Gravity(),),
                    boundarycondition=bc,
                           init_state=init_problem!,
                           data_config=data_config)
    ode_solver = CLIMA.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch)
    config = CLIMA.Atmos_LES_Configuration("DryRayleighBenardConvection",
                                           N, resolution, xmax, ymax, zmax,
                                           init_problem!,
                                           solver_type=ode_solver,
                                           model=model)
    return config
end

function main()
    CLIMA.init()
    FT = Float64
    # DG polynomial order
    N = 4
    # Domain resolution and size
    Δh = FT(10)
    # Time integrator setup
    t0 = FT(0)
    CFLmax = FT(0.90)
    timeend = FT(1000)
    xmax, ymax, zmax = 250, 250, 500

    @testset "DryRayleighBenardTest" begin
      for Δh in Δh
        Δv = Δh
        resolution = (Δh, Δh, Δv)
        driver_config = config_problem(FT, N, resolution, xmax, ymax, zmax)
        solver_config = CLIMA.setup_solver(t0, timeend, driver_config,
                                           init_on_cpu=true, Courant_number=CFLmax)
        # User defined callbacks (TMAR positivity preserving filter)
        cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init=false)
            Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
            nothing
        end
        result = CLIMA.invoke!(solver_config;
                              user_callbacks=(cbtmarfilter,),
                              check_euclidean_distance=true)
        # result == engf/eng0
        @test isapprox(result,FT(1); atol=1.5e-2)
      end
    end
end

main()
