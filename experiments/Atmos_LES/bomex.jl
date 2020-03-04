#=
@article{doi:10.1175/1520-0469(2003)60<1201:ALESIS>2.0.CO;2,
author = {Siebesma, A. Pier and Bretherton, 
          Christopher S. and Brown, 
          Andrew and Chlond, 
          Andreas and Cuxart, 
          Joan and Duynkerke, 
          Peter G. and Jiang, 
          Hongli and Khairoutdinov, 
          Marat and Lewellen, 
          David and Moeng, 
          Chin-Hoh and Sanchez, 
          Enrique and Stevens, 
          Bjorn and Stevens, 
          David E.},
title = {A Large Eddy Simulation Intercomparison Study of Shallow Cumulus Convection},
journal = {Journal of the Atmospheric Sciences},
volume = {60},
number = {10},
pages = {1201-1219},
year = {2003},
doi = {10.1175/1520-0469(2003)60<1201:ALESIS>2.0.CO;2},
URL = {https://journals.ametsoc.org/doi/abs/10.1175/1520-0469%282003%2960%3C1201%3AALESIS%3E2.0.CO%3B2},
eprint = {https://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%282003%2960%3C1201%3AALESIS%3E2.0.CO%3B2}
=# 

using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra

using CLIMA
using CLIMA.Atmos
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.GenericCallbacks
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates

import CLIMA.DGmethods: vars_state, vars_aux
import CLIMA.Atmos: source!, atmos_source!, altitude
import CLIMA.DGmethods: boundary_state!
import CLIMA.Atmos: atmos_boundary_state!, atmos_boundary_flux_diffusive!, flux_diffusive!, thermo_state

# ---------------------------- Begin Boundary Conditions ----------------- #
"""
  BOMEX_BC <: BoundaryCondition
  Prescribes boundary conditions for Barbados Oceanographic and Meteorological Experiment (BOMEX)
#Fields
$(DocStringExtensions.FIELDS)
"""
struct BOMEX_BC{FT} <: BoundaryCondition
  "Friction velocity"
  u_star::FT  
  "Sensible Heat Flux"
  w′θ′::FT
  "Latent Heat Flux"
  w′qt′::FT
end

"""
    atmos_boundary_state!(nf::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                          bc::BOMEX_BC, args...)

For the non-diffussive and gradient terms we just use the `NoFluxBC`
"""
atmos_boundary_state!(nf::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                      bc::BOMEX_BC, 
                      args...) = atmos_boundary_state!(nf, NoFluxBC(), args...)

"""
    atmos_boundary_flux_diffusive!(nf::NumericalFluxDiffusive,
                                   bc::BOMEX_BC, atmos::AtmosModel,
                                   F,
                                   state⁺, diff⁺, aux⁺, n⁻,
                                   state⁻, diff⁻, aux⁻,
                                   bctype, t,
                                   state₁⁻, diff₁⁻, aux₁⁻)
When `bctype == 1` the `NoFluxBC` otherwise the specialized BOMEX BC is used
"""
#TODO This needs to be in sync with the new boundary condition interfaces
function atmos_boundary_flux_diffusive!(nf::CentralNumericalFluxDiffusive,
                                        bc::BOMEX_BC, 
                                        atmos::AtmosModel, F,
                                        state⁺, diff⁺, hyperdiff⁺, aux⁺,
                                        n⁻,
                                        state⁻, diff⁻, hyperdiff⁻, aux⁻,
                                        bctype, t,
                                        state₁⁻, diff₁⁻, aux₁⁻)
  
  # Floatint point precision
  FT = eltype(state⁺)

  # Establish the thermodynamic state based on the prognostic variable state
  TS = thermo_state(atmos.moisture, atmos.orientation, state⁺, aux⁺)

  # Air temperature given current thermodynamic state
  temperature = air_temperature(TS)

  # Boundary condition for bottom wall
  if bctype != 1
    atmos_boundary_flux_diffusive!(nf, NoFluxBC(), atmos, F,
                                   state⁺, diff⁺, hyperdiff⁺, aux⁺, n⁻,
                                   state⁻, diff⁻, hyperdiff⁻, aux⁻,
                                   bctype, t,
                                   state₁⁻, diff₁⁻, aux₁⁻)
  else
    # Start with the noflux BC and then build custom flux from there
    atmos_boundary_state!(nf, NoFluxBC(), atmos,
                          state⁺, diff⁺, aux⁺, n⁻,
                          state⁻, diff⁻, aux⁻,
                          bctype, t)

    # Interior state velocities [u₀]
    u₁ = state₁⁻.ρu / state₁⁻.ρ
    # Windspeed at first interior node
    # [Impenetrable flow, normal component of velocity is zero, horizontal components may be non-zero]
    windspeed₁ = norm(u₁)
    # Turbulence tensors in the interior state
    _, τ⁻ = turbulence_tensors(atmos.turbulence, state⁻, diff⁻, aux⁻, t)
    u_star = bc.u_star # Constant value for friction-velocity u_star == u_star
    @inbounds begin
      τ13⁺ = - u_star^2 * windspeed₁[1] / norm(windspeed₁) # ⟨u′w′⟩ 
      τ23⁺ = - u_star^2 * windspeed₁[2] / norm(windspeed₁) # ⟨v′w′⟩
      τ21⁺ = τ⁻[2,1]
    end
    # Momentum boundary condition [This is topography specific #FIXME: Simon's interface will improve this]
    τ⁺ = SHermitianCompact{3, FT, 6}(SVector(0   ,
                                             τ21⁺, τ13⁺,
                                             0   , τ23⁺, 0))
    # Moisture boundary condition ⟨w′qt′⟩
    d_q_tot⁺  = SVector(0, 
                        0, 
                        state⁺.ρ * bc.w′qt′)
    # Heat flux boundary condition
    d_h_tot⁺ = SVector(0, 
                       0, 
                       state⁺.ρ * (bc.w′θ′ * cp_m(TS) + bc.w′qt′ * latent_heat_vapor(temperature)))
    # Set the flux using the now defined plus-side data
    flux_diffusive!(atmos, F, state⁺, τ⁺, d_h_tot⁺)
    flux_diffusive!(atmos.moisture, F, state⁺, d_q_tot⁺)
  end
end
# ------------------------ End Boundary Condition --------------------- # 

"""
  Bomex Geostrophic Forcing (Source)
"""
struct BomexGeostrophic{FT} <: Source
  "Coriolis parameter [s⁻¹]"
  f_coriolis::FT
  "Eastward geostrophic velocity `[m/s]` (Base)"
  u_geostrophic::FT
  "Eastward geostrophic velocity `[m/s]` (Slope)"
  u_slope::FT
  "Northward geostrophic velocity `[m/s]`"
  v_geostrophic::FT
end
function atmos_source!(s::BomexGeostrophic, atmos::AtmosModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, t::Real)

  f_coriolis    = s.f_coriolis
  u_geostrophic = s.u_geostrophic
  u_slope       = s.u_slope
  v_geostrophic = s.v_geostrophic

  z          = altitude(atmos.orientation,aux)
  # Note z dependence of eastward geostrophic velocity
  u_geo      = SVector(u_geostrophic + u_slope * z, v_geostrophic, 0)
  ẑ          = vertical_unit_vector(atmos.orientation, aux)
  fkvector   = f_coriolis * ẑ
  # Accumulate sources
  source.ρu -= fkvector × (state.ρu .- state.ρ*u_geo)
  return nothing
end

"""
  Bomex Sponge (Source)
"""
struct BomexSponge{FT} <: Source
  "Maximum domain altitude (m)"
  z_max::FT
  "Altitude at with sponge starts (m)"
  z_sponge::FT
  "Sponge Strength 0 ⩽ α_max ⩽ 1"
  α_max::FT
  "Sponge exponent"
  γ::FT
  "Eastward geostrophic velocity `[m/s]` (Base)"
  u_geostrophic::FT
  "Eastward geostrophic velocity `[m/s]` (Slope)"
  u_slope::FT
  "Northward geostrophic velocity `[m/s]`"
  v_geostrophic::FT
end
function atmos_source!(s::BomexSponge, atmos::AtmosModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, t::Real)

  z_max = s.z_max
  z_sponge = s.z_sponge
  α_max = s.α_max
  γ = s.γ
  u_geostrophic = s.u_geostrophic
  u_slope       = s.u_slope
  v_geostrophic = s.v_geostrophic

  z          = altitude(atmos.orientation,aux)
  u_geo      = SVector(u_geostrophic + u_slope * z, v_geostrophic, 0)
  ẑ          = vertical_unit_vector(atmos.orientation, aux)
  # Accumulate sources
  if z_sponge <= z
    r = (z - z_sponge)/(z_max-z_sponge)
    β_sponge = α_max * sinpi(r/2)^s.γ
    source.ρu -= β_sponge * (state.ρu .- state.ρ * u_geo)
  end
  return nothing
end

"""
  BomexTendencies (Source)
Moisture, Temperature and Subsidence tendencies
"""
struct BomexTendencies{FT} <: Source
  "Advection tendency in total moisture `[s⁻¹]`"
  ∂qt∂t_peak::FT
  "Lower extent of piecewise profile (moisture term) `[m]`"
  zl_moisture::FT   
  "Upper extent of piecewise profile (moisture term) `[m]`"
  zh_moisture::FT
  "Cooling rate `[K/s]`"
  ∂θ∂t_peak::FT
  "Lower extent of piecewise profile (subsidence term) `[m]`"
  zl_sub::FT
  "Upper extent of piecewise profile (subsidence term) `[m]`"
  zh_sub::FT
  "Subsidence peak velocity"
  w_sub::FT
  "Max height in domain"
  z_max::FT
end
function atmos_source!(s::BomexTendencies, atmos::AtmosModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  FT    = eltype(state)
  ρ     = state.ρ
  z     = altitude(atmos.orientation,aux)
  q_tot = state.moisture.ρq_tot / state.ρ
  TS    = thermo_state(atmos.moisture, atmos.orientation, state, aux)
  
  # Moisture tendencey (sink term) 
  # Temperature tendency (Radiative cooling)
  # Large scale subsidence
  # Unpack struct
  zl_moisture = s.zl_moisture
  zh_moisture = s.zh_moisture
  z_max       = s.z_max
  zl_sub      = s.zl_sub
  zh_sub      = s.zh_sub
  w_sub       = s.w_sub
  ∂qt∂t_peak  = s.∂qt∂t_peak
  ∂θ∂t_peak   = s.∂θ∂t_peak
  k̂           = vertical_unit_vector(atmos.orientation, aux)

  # Piecewise profile for advective moisture forcing
  P     = air_pressure(TS)

  # Thermodynamic state identification
  q_pt  = PhasePartition(TS)
  cvm   = cv_m(TS)
  
  linscale_moisture = (z-zl_moisture)/(zh_moisture-zl_moisture)
  linscale_temp     = (z-zl_sub) / (z_max-zl_sub)
  linscale_sub      = (z-zl_sub) / (zh_sub-zl_sub)

  # Piecewise term for moisture tendency
  if z <= zl_moisture
    ρ∂qt∂t = ρ*∂qt∂t_peak
  elseif zl_moisture < z <= zh_moisture
    ρ∂qt∂t = ρ*(∂qt∂t_peak - ∂qt∂t_peak * linscale_moisture)
  else
    ρ∂qt∂t = -zero(FT)
  end
  
  # Piecewise term for internal energy tendency
  if z <= zl_sub
    ρ∂θ∂t = ρ*∂θ∂t_peak
  elseif  zh_sub < z <= z_max
    ρ∂θ∂t = ρ*(∂θ∂t_peak - ∂θ∂t_peak*linscale_temp)
  else
    ρ∂θ∂t = -zero(FT)
  end
  
  # Moisture tendency source
  source.moisture.ρq_tot += ρ∂qt∂t
  # Internal energy tendency source
  source.ρe += cvm*ρ∂θ∂t*exner(TS) + e_int_v0*ρ∂qt∂t
  
  wₛ = -zero(FT)
  if z <= zl_sub
    wₛ = -zero(FT) + z*(w_sub)/(zl_sub)
  elseif zl_sub < z <= zh_sub
    wₛ = w_sub - (w_sub)*linscale_sub
  else
    wₛ = -zero(FT)
  end
  
  # Large scale subsidence tendency
  source.ρe -= ρ * wₛ * dot(k̂, diffusive.∇h_tot)
  source.moisture.ρq_tot -= ρ * wₛ * dot(k̂, diffusive.moisture.∇q_tot)
  return nothing
end

"""
  Initial Condition for BOMEX LES
"""
#TODO merge with new data_config feature for atmosmodel to avoid global constants
seed = MersenneTwister(0)
function init_bomex!(bl, state, aux, (x,y,z), t)
  # This experiment runs the BOMEX LES Configuration
  # (Shallow cumulus cloud regime)
  # x,y,z imply eastward, northward and altitude coordinates in `[m]`
  
  # Problem floating point precision
  FT        = eltype(state)
  
  P_sfc::FT = 1.015e5 # Surface air pressure
  qg::FT    = 22.45e-3 # Total moisture at surface
  q_pt_sfc  = PhasePartition(qg) # Surface moisture partitioning
  Rm_sfc    = FT(gas_constant_air(q_pt_sfc)) # Moist gas constant
  θ_liq_sfc = FT(299.1) # Prescribed θ_liq at surface
  T_sfc     = FT(300.4) # Ground air temperature
  
  # Initialise speeds [u = Eastward, v = Northward, w = Vertical]
  u::FT     = 0
  v::FT     = 0
  w::FT     = 0 
  
  # Prescribed altitudes for piece-wise profile construction
  zl1::FT   = 520
  zl2::FT   = 1480
  zl3::FT   = 2000
  zl4::FT   = 3000

  # Assign piecewise quantities to θ_liq and q_tot 
  θ_liq::FT = 0 
  q_tot::FT = 0 

  # Piecewise functions for potential temperature and total moisture
  if FT(0) <= z <= zl1
    # Well mixed layer
    θ_liq = 298.7
    q_tot = 17.0 + (z/zl1)*(16.3-17.0)
  elseif z > zl1 && z <= zl2
    # Conditionally unstable layer
    θ_liq = 298.7 + (z-zl1) * (302.4-298.7)/(zl2-zl1)
    q_tot = 16.3 + (z-zl1) * (10.7-16.3)/(zl2-zl1)
  elseif z > zl2 && z <= zl3
    # Absolutely stable inversion
    θ_liq = 302.4 + (z-zl2) * (308.2-302.4)/(zl3-zl2)
    q_tot = 10.7 + (z-zl2) * (4.2-10.7)/(zl3-zl2)
  else
    θ_liq = 308.2 + (z-zl3) * (311.85-308.2)/(zl4-zl3)
    q_tot = 4.2 + (z-zl3) * (3.0-4.2)/(zl4-zl3)
  end
  
  # Set velocity profiles - piecewise profile for u
  zlv::FT = 700
  if z <= zlv
    u = -8.75
  else
    u = -8.75 + (z - zlv) * (-4.61 + 8.75)/(zl4 - zlv)
  end
  
  # Convert total specific humidity to kg/kg
  q_tot /= 1000 
  # Scale height based on surface parameters
  H     = Rm_sfc * T_sfc / grav
  # Pressure based on scale height
  P     = P_sfc * exp(-z / H)   

  # Establish thermodynamic state and moist phase partitioning
  TS = LiquidIcePotTempSHumEquil_given_pressure(θ_liq, P, q_tot)
  T = air_temperature(TS)
  ρ = air_density(TS)
  q_pt = PhasePartition(TS)

  # Compute momentum contributions
  ρu          = ρ * u
  ρv          = ρ * v
  ρw          = ρ * w
  
  # Compute energy contributions
  e_kin       = FT(1//2) * (u^2 + v^2 + w^2)
  e_pot       = FT(grav) * z
  ρe_tot      = ρ * total_energy(e_kin, e_pot, T, q_pt)

  # Assign initial conditions for prognostic state variables
  state.ρ     = ρ
  state.ρu    = SVector(ρu, ρv, ρw) 
  state.ρe    = ρe_tot 
  state.moisture.ρq_tot = ρ * q_tot 
  
  if z <= FT(200) # Add random perturbations to bottom 200m of model
    state.ρe += rand(seed)*ρe_tot/100
    state.moisture.ρq_tot += rand(seed)*ρ*q_tot/100
  end
end

function config_bomex(FT, N, resolution, xmax, ymax, zmax)
  
  ics = init_bomex!     # Initial conditions 

  C_smag = FT(0.23)     # Smagorinsky coefficient

  u_star = FT(0.28)     # Friction velocity
  w′θ′   = FT(8e-3)     # Potential temperature flux
  w′qt′  = FT(5.2e-5)   # Total moisture flux

  bc     = BOMEX_BC{FT}(u_star, w′θ′, w′qt′) # Boundary conditions
  
  ∂qt∂t_peak = FT(-1.2e-8)  # Moisture tendency (energy source)
  zl_qt = FT(300)           # Low altitude limit for piecewise function (moisture source)
  zh_qt = FT(500)           # High altitude limit for piecewise function (moisture source)
  ∂θ∂t_peak  = FT(-2/day)   # Potential temperature tendency (energy source)

  z_sponge = FT(2400)       # Start of sponge layer
  α_max    = FT(1.0)        # Strength of sponge layer (timescale)
  γ        = 2              # Strength of sponge layer (exponent)
  u_geostrophic = FT(-10)        # Eastward relaxation speed
  u_slope       = FT(1.8e-3)     # Slope of altitude-dependent relaxation speed
  v_geostrophic = FT(0)          # Northward relaxation speed

  zl_sub = FT(1500)         # Low altitude for piecewise function (subsidence source)
  zh_sub = FT(2100)         # High altitude for piecewise function (subsidence source)
  w_sub  = FT(-0.65e-2)     # Subsidence velocity peak value

  f_coriolis = FT(0.376e-4) # Coriolis parameter
  
  # Assemble source components
  source = (
            Gravity(),
            BomexTendencies{FT}(∂qt∂t_peak, zl_qt, zh_qt, ∂θ∂t_peak, zl_sub, zh_sub, w_sub, zmax),
            BomexSponge{FT}(zmax, z_sponge, α_max, γ, u_geostrophic, u_slope, v_geostrophic),
            BomexGeostrophic{FT}(f_coriolis, u_geostrophic, u_slope, v_geostrophic)
           )

  # Assemble timestepper components
  ode_solver_type = CLIMA.DefaultSolverType()

  # Assemble model components
  model = AtmosModel{FT}(AtmosLESConfiguration;
                         turbulence        = SmagorinskyLilly{FT}(C_smag),
                         moisture          = EquilMoist{FT}(;maxiter=15,
                                                             tolerance=0.1),
                         source            = source,
                         boundarycondition = bc,
                         init_state        = ics)
  
  # Assemble configuration
  config = CLIMA.Atmos_LES_Configuration("BOMEX", N, resolution,
                                         xmax, ymax, zmax,
                                         init_bomex!,
                                         solver_type=ode_solver_type,
                                         model=model)
    return config
end

function main()
  CLIMA.init()

  FT = Float64

  # DG polynomial order
  N = 4
  # Domain resolution and size
  Δh = FT(100)
  Δv = FT(40)

  resolution = (Δh, Δh, Δv)
  
  # Prescribe domain parameters
  xmax = 6400
  ymax = 6400
  zmax = 3000

  t0 = FT(0)
  timeend = FT(3600*6)
  CFLmax  = FT(0.5)

  driver_config = config_bomex(FT, N, resolution, xmax, ymax, zmax)
  solver_config = CLIMA.setup_solver(t0, timeend, driver_config, forcecpu=true, Courant_number=CFLmax)

  cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init=false)
      Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
      nothing
  end
    
  result = CLIMA.invoke!(solver_config;
                        user_callbacks=(cbtmarfilter,),
                        check_euclidean_distance=true)
end

main()
