using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra

using CLIMA
using CLIMA.Atmos
using CLIMA.GenericCallbacks
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.ODESolvers
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates

import CLIMA.DGmethods: boundary_state!
import CLIMA.Atmos: atmos_boundary_state!, atmos_boundary_flux_diffusive!, flux_diffusive!
import CLIMA.DGmethods.NumericalFluxes: boundary_flux_diffusive!

# -------------------- Surface Driven Bubble ----------------- #
# Rising thermals driven by a prescribed surface heat flux.
# 1) Boundary Conditions:
#       Laterally periodic with no flow penetration through top
#       and bottom wall boundaries.
#       Momentum: No flow penetration [NoFluxBC()]
#       Energy:   Spatially varying non-zero heat flux up to time t₁
# 2) Domain: 1250m × 1250m × 1000m
# Configuration defaults are in `src/Driver/Configurations.jl`

"""
  SurfaceDrivenBubbleBC <: BoundaryCondition
Y ≡ state vars
Σ ≡ diffusive vars
A ≡ auxiliary vars
X⁺ and X⁻ refer to exterior, interior faces
X₁ refers to the first interior node

# Fields
$(DocStringExtensions.FIELDS)
"""
struct SurfaceDrivenBubbleBC{FT} <: BoundaryCondition
  "Prescribed MSEF Magnitude `[W/m^2]`"
  F₀::FT
  "Time Cutoff `[s]`"
  t₁::FT
  "Plume wavelength scaling"
  x₀::FT
end
function atmos_boundary_state!(nf::Union{NumericalFluxNonDiffusive,NumericalFluxGradient},
                               bc::SurfaceDrivenBubbleBC,
                               m::AtmosModel,
                               Y⁺::Vars, A⁺::Vars,
                               n⁻,
                               Y⁻::Vars, A⁻::Vars,
                               bctype, t,_...)
  # Use default NoFluxBC()
  atmos_boundary_state!(nf, NoFluxBC(), m,
                        Y⁺, A⁺, n⁻,
                        Y⁻, A⁻,
                        bctype, t)
end
function atmos_boundary_flux_diffusive!(nf::CentralNumericalFluxDiffusive,
                                        bc::SurfaceDrivenBubbleBC,
                                        m::AtmosModel,
                                        F,
                                        Y⁺::Vars, Σ⁺::Vars, HD⁺::Vars, A⁺::Vars,
                                        n⁻,
                                        Y⁻::Vars, Σ⁻::Vars, HD⁻::Vars, A⁻::Vars,
                                        bctype, t, Y₁⁻, Σ₁⁻, A₁⁻)
  # Working precision
  FT = eltype(Y⁻)
  # Assign vertical unit vector and coordinates
  k̂  = vertical_unit_vector(m.orientation, A⁻)
  x = A⁻.coord[1]
  y = A⁻.coord[2]
  # Unpack fields
  t₁    = bc.t₁
  x₀    = bc.x₀
  F₀ =  t < t₁ ? bc.F₀ : -zero(FT)
  # Apply boundary condition per face (1 == bottom wall)
  if bctype != 1
    atmos_boundary_flux_diffusive!(nf, NoFluxBC(), m, F,
                                   Y⁺, Σ⁺, HD⁺, A⁺,
                                   n⁻,
                                   Y⁻, Σ⁻, HD⁻, A⁻,
                                   bctype, t,
                                   Y₁⁻, Σ₁⁻, A₁⁻)
  else
    atmos_boundary_state!(nf, NoFluxBC(), m,
                          Y⁺, Σ⁺, A⁺,
                          n⁻,
                          Y⁻, Σ⁻, A⁻,
                          bctype, t)
    MSEF    = F₀ * (cospi(2*x/x₀))^2 * (cospi(2*y/x₀))^2
    ∇h_tot⁺ = MSEF * k̂
    _,τ⁺ = turbulence_tensors(m.turbulence, Y⁺, Σ⁺, A⁺, t)
    flux_diffusive!(m, F, Y⁺, τ⁺, ∇h_tot⁺)
  end
end

"""
  Surface Driven Thermal Bubble
"""
function init_surfacebubble!(bl, state, aux, (x,y,z), t)
  FT            = eltype(state)
  R_gas::FT     = R_d
  c_p::FT       = cp_d
  c_v::FT       = cv_d
  γ::FT         = c_p / c_v
  p0::FT        = MSLP

  xc::FT        = 1250
  yc::FT        = 1250
  zc::FT        = 1250
  θ_ref::FT     = 300
  Δθ::FT        = 0

  #Perturbed state:
  θ            = θ_ref + Δθ # potential temperature
  π_exner      = FT(1) - grav / (c_p * θ) * z # exner pressure
  ρ            = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density

  q_tot        = FT(0)
  ts           = LiquidIcePotTempSHumEquil(θ, ρ, q_tot)
  q_pt         = PhasePartition(ts)

  ρu           = SVector(FT(0),FT(0),FT(0))
  # energy definitions
  e_kin        = FT(0)
  e_pot        = gravitational_potential(bl.orientation, aux)
  ρe_tot       = ρ * total_energy(e_kin, e_pot, ts)
  state.ρ      = ρ
  state.ρu     = ρu
  state.ρe     = ρe_tot
  state.moisture.ρq_tot = ρ*q_pt.tot
end

function config_surfacebubble(FT, N, resolution, xmax, ymax, zmax)

  # Boundary conditions
  # Heat Flux Peak Magnitude
  F₀ = FT(100)
  # Time [s] at which `heater` turns off
  t₁ = FT(500)
  # Plume wavelength scaling
  x₀ = xmax
  bc = SurfaceDrivenBubbleBC{FT}(F₀, t₁, x₀)

  C_smag = FT(0.23)

  imex_solver = CLIMA.DefaultSolverType()
  explicit_solver = CLIMA.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch)

  model = AtmosModel{FT}(AtmosLESConfiguration;
                         turbulence=SmagorinskyLilly{FT}(C_smag),
                         source=(Gravity(),),
                         boundarycondition=bc,
                         moisture=EquilMoist{FT}(),
                         init_state=init_surfacebubble!)
  config = CLIMA.Atmos_LES_Configuration("SurfaceDrivenBubble",
                                   N, resolution, xmax, ymax, zmax,
                                   init_surfacebubble!,
                                   solver_type=explicit_solver,
                                   model=model)
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
  xmax = 2000
  ymax = 2000
  zmax = 2000
  t0 = FT(0)
  timeend = FT(2000)

  CFL_max = FT(0.4)

  driver_config = config_surfacebubble(FT, N, resolution, xmax, ymax, zmax)
  solver_config = CLIMA.setup_solver(t0, timeend, Courant_number=CFL_max, driver_config, init_on_cpu=true)

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
