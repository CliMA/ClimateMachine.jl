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
using CLIMA.ODESolvers
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates

import CLIMA.DGmethods: vars_state, vars_aux,
                        vars_integrals, vars_reverse_integrals,
                        indefinite_stack_integral!,
                        reverse_indefinite_stack_integral!,
                        integral_load_aux!, integral_set_aux!,
                        reverse_integral_load_aux!,
                        reverse_integral_set_aux!

import CLIMA.DGmethods: boundary_state!
import CLIMA.Atmos: atmos_boundary_state!, atmos_boundary_flux_diffusive!, flux_diffusive!
import CLIMA.DGmethods.NumericalFluxes: boundary_flux_diffusive!

# -------------------- Radiation Model -------------------------- # 
vars_state(::RadiationModel, FT) = @vars()
vars_aux(::RadiationModel, FT) = @vars()
vars_integrals(::RadiationModel, FT) = @vars()
vars_reverse_integrals(::RadiationModel, FT) = @vars()

function atmos_nodal_update_aux!(::RadiationModel, ::AtmosModel, state::Vars, aux::Vars, t::Real) end
function preodefun!(::RadiationModel, aux::Vars, state::Vars, t::Real) end
function integral_load_aux!(::RadiationModel, integ::Vars, state::Vars, aux::Vars) end
function integral_set_aux!(::RadiationModel, aux::Vars, integ::Vars) end
function reverse_integral_load_aux!(::RadiationModel, integ::Vars, state::Vars, aux::Vars) end
function reverse_integral_set_aux!(::RadiationModel, aux::Vars, integ::Vars) end
function flux_radiation!(::RadiationModel, flux::Grad, state::Vars, aux::Vars, t::Real) end


# ---------------------------- Begin Boundary Conditions ----------------- #
"""
  DYCOMS_BC <: BoundaryCondition
  Prescribes boundary conditions for Dynamics of Marine Stratocumulus Case
#Fields
$(DocStringExtensions.FIELDS)
"""
struct DYCOMS_BC{FT} <: BoundaryCondition
  "Drag coefficient"
  C_drag::FT
  "Latent Heat Flux"
  LHF::FT
  "Sensible Heat Flux"
  SHF::FT
end

"""
    atmos_boundary_state!(nf::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                          bc::DYCOMS_BC, args...)

For the non-diffussive and gradient terms we just use the `NoFluxBC`
"""
atmos_boundary_state!(nf::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                      bc::DYCOMS_BC, 
                      args...) = atmos_boundary_state!(nf, NoFluxBC(), args...)

"""
    atmos_boundary_flux_diffusive!(nf::NumericalFluxDiffusive,
                                   bc::DYCOMS_BC, atmos::AtmosModel,
                                   F,
                                   state⁺, diff⁺, aux⁺, n⁻,
                                   state⁻, diff⁻, aux⁻,
                                   bctype, t,
                                   state1⁻, diff1⁻, aux1⁻)

When `bctype == 1` the `NoFluxBC` otherwise the specialized DYCOMS BC is used
"""
function atmos_boundary_flux_diffusive!(nf::CentralNumericalFluxDiffusive,
                                        bc::DYCOMS_BC, 
                                        atmos::AtmosModel, F,
                                        state⁺, diff⁺, hyperdiff⁺, aux⁺,
                                        n⁻,
                                        state⁻, diff⁻, hyperdiff⁻, aux⁻,
                                        bctype, t,
                                        state1⁻, diff1⁻, aux1⁻)
  if bctype != 1
    atmos_boundary_flux_diffusive!(nf, NoFluxBC(), atmos, F,
                                   state⁺, diff⁺, hyperdiff⁺, aux⁺, n⁻,
                                   state⁻, diff⁻, hyperdiff⁻, aux⁻,
                                   bctype, t,
                                   state1⁻, diff1⁻, aux1⁻)
  else
    # Start with the noflux BC and then build custom flux from there
    atmos_boundary_state!(nf, NoFluxBC(), atmos,
                          state⁺, diff⁺, aux⁺, n⁻,
                          state⁻, diff⁻, aux⁻,
                          bctype, t)

    # ------------------------------------------------------------------------
    # (<var>_FN) First node values (First interior node from bottom wall)
    # ------------------------------------------------------------------------
    u_FN = state1⁻.ρu / state1⁻.ρ
    windspeed_FN = norm(u_FN)

    # ----------------------------------------------------------
    # Extract components of diffusive momentum flux (minus-side)
    # ----------------------------------------------------------
    _, τ⁻ = turbulence_tensors(atmos.turbulence, state⁻, diff⁻, aux⁻, t)

    # ----------------------------------------------------------
    # Boundary momentum fluxes
    # ----------------------------------------------------------
    # Case specific for flat bottom topography, normal vector is n⃗ = k⃗ = [0, 0, 1]ᵀ
    # A more general implementation requires (n⃗ ⋅ ∇A) to be defined where A is
    # replaced by the appropriate flux terms
    C_drag = bc.C_drag
    @inbounds begin
      τ13⁺ = - C_drag * windspeed_FN * u_FN[1]
      τ23⁺ = - C_drag * windspeed_FN * u_FN[2]
      τ21⁺ = τ⁻[2,1]
    end

    # Assign diffusive momentum and moisture fluxes
    # (i.e. ρ𝛕 terms)
    FT = eltype(state⁺)
    τ⁺ = SHermitianCompact{3, FT, 6}(SVector(0   ,
                                             τ21⁺, τ13⁺,
                                             0   , τ23⁺, 0))

    # ----------------------------------------------------------
    # Boundary moisture fluxes
    # ----------------------------------------------------------
    # really ∇q_tot is being used to store d_q_tot
    d_q_tot⁺  = SVector(0, 0, bc.LHF/(LH_v0))

    # ----------------------------------------------------------
    # Boundary energy fluxes
    # ----------------------------------------------------------
    # Assign diffusive enthalpy flux (i.e. ρ(J+D) terms)
    d_h_tot⁺ = SVector(0, 0, bc.LHF + bc.SHF)

    # Set the flux using the now defined plus-side data
    flux_diffusive!(atmos, F, state⁺, τ⁺, d_h_tot⁺)
    flux_diffusive!(atmos.moisture, F, state⁺, d_q_tot⁺)
  end
end
# ------------------------ End Boundary Condition --------------------- # 


# ------------------------ Begin Radiation Model ---------------------- #
"""
  DYCOMSRadiation <: RadiationModel

Stevens et. al (2005) approximation of longwave radiative fluxes in DYCOMS.
Analytical description as a function of the liquid water path and inversion height zᵢ

* Stevens, B. et. al. (2005) "Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine Stratocumulus". Mon. Wea. Rev., 133, 1443–1462, https://doi.org/10.1175/MWR2930.1
"""
struct DYCOMSRadiation{FT} <: RadiationModel
  "mass absorption coefficient `[m^2/kg]`"
  κ::FT
  "Troposphere cooling parameter `[m^(-4/3)]`"
  α_z::FT
  "Inversion height `[m]`"
  z_i::FT
  "Density"
  ρ_i::FT
  "Large scale divergence `[s^(-1)]`"
  D_subsidence::FT
  "Radiative flux parameter `[W/m^2]`"
  F_0::FT
  "Radiative flux parameter `[W/m^2]`"
  F_1::FT
end

vars_aux(m::DYCOMSRadiation, FT) = @vars(Rad_flux::FT)

vars_integrals(m::DYCOMSRadiation, FT) = @vars(attenuation_coeff::FT)
function integral_load_aux!(m::DYCOMSRadiation, integrand::Vars, state::Vars, aux::Vars)
  FT = eltype(state)
  integrand.radiation.attenuation_coeff = state.ρ * m.κ * aux.moisture.q_liq
end
function integral_set_aux!(m::DYCOMSRadiation, aux::Vars, integrand::Vars)
  integrand = integrand.radiation.attenuation_coeff
  aux.∫dz.radiation.attenuation_coeff = integrand
end

vars_reverse_integrals(m::DYCOMSRadiation, FT) = @vars(attenuation_coeff::FT)
function reverse_integral_load_aux!(m::DYCOMSRadiation, integrand::Vars, state::Vars, aux::Vars)
  FT = eltype(state)
  integrand.radiation.attenuation_coeff = state.ρ * m.κ * aux.moisture.q_liq
end
function reverse_integral_set_aux!(m::DYCOMSRadiation, aux::Vars, integrand::Vars)
  aux.∫dnz.radiation.attenuation_coeff = integrand.radiation.attenuation_coeff
end

function flux_radiation!(m::DYCOMSRadiation, atmos::AtmosModel, flux::Grad, state::Vars,
                         aux::Vars, t::Real)
  FT = eltype(flux)
  z = altitude(atmos.orientation, aux)
  Δz_i = max(z - m.z_i, -zero(FT))
  # Constants
  upward_flux_from_cloud  = m.F_0 * exp(-aux.∫dnz.radiation.attenuation_coeff)  
  upward_flux_from_sfc = m.F_1 * exp(-aux.∫dz.radiation.attenuation_coeff)
  free_troposphere_flux = m.ρ_i * FT(cp_d) * m.D_subsidence * m.α_z * cbrt(Δz_i) * (Δz_i/4 + m.z_i)
  F_rad = upward_flux_from_sfc + upward_flux_from_cloud + free_troposphere_flux
  ẑ = vertical_unit_vector(atmos.orientation, aux)
  flux.ρe += F_rad * ẑ
end
function preodefun!(m::DYCOMSRadiation, aux::Vars, state::Vars, t::Real)
end
# -------------------------- End Radiation Model ------------------------ # 

"""
  Initial Condition for DYCOMS_RF01 LES
@article{doi:10.1175/MWR2930.1,
author = {Stevens, Bjorn and Moeng, Chin-Hoh and Ackerman,
          Andrew S. and Bretherton, Christopher S. and Chlond,
          Andreas and de Roode, Stephan and Edwards, James and Golaz,
          Jean-Christophe and Jiang, Hongli and Khairoutdinov,
          Marat and Kirkpatrick, Michael P. and Lewellen, David C. and Lock, Adrian and
          Maeller, Frank and Stevens, David E. and Whelan, Eoin and Zhu, Ping},
title = {Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine Stratocumulus},
journal = {Monthly Weather Review},
volume = {133},
number = {6},
pages = {1443-1462},
year = {2005},
doi = {10.1175/MWR2930.1},
URL = {https://doi.org/10.1175/MWR2930.1},
eprint = {https://doi.org/10.1175/MWR2930.1}
}
"""
function init_dycoms!(bl, state, aux, (x,y,z), t)
    FT = eltype(state)

    z = altitude(bl.orientation, aux)

    # These constants are those used by Stevens et al. (2005)
    qref       = FT(9.0e-3)
    q_pt_sfc   = PhasePartition(qref)
    Rm_sfc     = FT(gas_constant_air(q_pt_sfc))
    T_sfc      = FT(290.4)
    P_sfc      = FT(MSLP)

    # Specify moisture profiles
    q_liq      = FT(0)
    q_ice      = FT(0)
    zb         = FT(600)         # initial cloud bottom
    zi         = FT(840)         # initial cloud top

    if z <= zi
        θ_liq  = FT(289.0)
        q_tot  = qref
    else
        θ_liq  = FT(297.0) + (z - zi)^(FT(1/3))
        q_tot  = FT(1.5e-3)
    end

    ugeo = FT(7)
    vgeo = FT(-5.5)
    u, v, w = ugeo, vgeo, FT(0)

    # Perturb initial state to break symmetry and trigger turbulent convection
    r1 = FT(rand(Uniform(-0.002, 0.002)))
    if z <= 200.0
        θ_liq += r1 * θ_liq
    end

    # Pressure
    H     = Rm_sfc * T_sfc / grav
    p     = P_sfc * exp(-z / H)

    # Density, Temperature
    ts    = LiquidIcePotTempSHumEquil_given_pressure(θ_liq, p, q_tot)
    ρ     = air_density(ts)

    e_kin = FT(1/2) * FT((u^2 + v^2 + w^2))
    e_pot = gravitational_potential(bl.orientation, aux)
    E     = ρ * total_energy(e_kin, e_pot, ts)

    state.ρ               = ρ
    state.ρu              = SVector(ρ*u, ρ*v, ρ*w)
    state.ρe              = E
    state.moisture.ρq_tot = ρ * q_tot

    return nothing
end

function config_dycoms(FT, N, resolution, xmax, ymax, zmax)
    # Reference state
    T_min   = FT(289)
    T_s     = FT(290.4)
    Γ_lapse = FT(grav/cp_d)
    T       = LinearTemperatureProfile(T_min, T_s, Γ_lapse)
    rel_hum = FT(0)
    ref_state = HydrostaticState(T, rel_hum)

    # Radiation model
    κ             = FT(85)
    α_z           = FT(1)
    z_i           = FT(840)
    ρ_i           = FT(1.13)
    D_subsidence  = FT(0) # 0 for stable testing, 3.75e-6 in practice
    F_0           = FT(70)
    F_1           = FT(22)
    radiation = DYCOMSRadiation{FT}(κ, α_z, z_i, ρ_i, D_subsidence, F_0, F_1)

    # Sources
    f_coriolis    = FT(1.03e-4)
    u_geostrophic = FT(7.0)
    v_geostrophic = FT(-5.5)
    w_ref         = FT(0)
    u_relaxation  = SVector(u_geostrophic, v_geostrophic, w_ref)
    # Sponge
    c_sponge = 1
    # Rayleigh damping
    zsponge = FT(1500.0)
    rayleigh_sponge = RayleighSponge{FT}(zmax, zsponge, c_sponge, u_relaxation, 2)
    # Geostrophic forcing
    geostrophic_forcing = GeostrophicForcing{FT}(f_coriolis, u_geostrophic, v_geostrophic)

    # Boundary conditions
    # SGS Filter constants
    C_smag = FT(0.21) # 0.21 for stable testing, 0.18 in practice
    C_drag = FT(0.0011)
    LHF    = FT(115)
    SHF    = FT(15)
    bc = DYCOMS_BC{FT}(C_drag, LHF, SHF)
    ics = init_dycoms!
    source = (Gravity(),
              rayleigh_sponge,
              Subsidence{FT}(D_subsidence),
              geostrophic_forcing)

    model = AtmosModel{FT}(AtmosLESConfiguration;
                           ref_state=ref_state,
                          turbulence=SmagorinskyLilly{FT}(C_smag),
                            moisture=EquilMoist{FT}(;maxiter=5),
                           radiation=radiation,
                              source=source,
                   boundarycondition=bc,
                          init_state=ics)

    config = CLIMA.Atmos_LES_Configuration("DYCOMS", N, resolution, xmax, ymax, zmax,
                                           init_dycoms!,
                                           solver_type=CLIMA.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch),
                                           model=model)

    return config
end

function main()
    CLIMA.init()

    FT = Float64

    # DG polynomial order
    N = 4

    # Domain resolution and size
    Δh = FT(40)
    Δv = FT(20)
    resolution = (Δh, Δh, Δv)

    xmax = 1000
    ymax = 1000
    zmax = 2500

    t0 = FT(0)
    timeend = FT(100)

    driver_config = config_dycoms(FT, N, resolution, xmax, ymax, zmax)
    solver_config = CLIMA.setup_solver(t0, timeend, driver_config, init_on_cpu=true)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(2) do (init=false)
        Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
        nothing
    end
      
    result = CLIMA.invoke!(solver_config;
                          user_callbacks=(cbtmarfilter,),
                          check_euclidean_distance=true)
end

main()
