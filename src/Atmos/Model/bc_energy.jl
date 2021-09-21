abstract type EnergyBC end

using ..TurbulenceClosures
using SurfaceFluxes: get_energy_flux, surface_conditions, DGScheme

"""
    Insulating() :: EnergyBC
No energy flux across the boundary.
"""
struct Insulating <: EnergyBC end
function atmos_energy_boundary_state!(nf, bc_energy::Insulating, atmos, _...) end
function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::Insulating,
    atmos,
    _...,
) end


"""
    PrescribedTemperature(fn) :: EnergyBC
Prescribe the temperature at the boundary by `fn`, a function with signature
`fn(state, aux, t)` returning the temperature (in K).
"""
struct PrescribedTemperature{FN} <: EnergyBC
    fn::FN
end
function atmos_energy_boundary_state!(
    nf,
    bc_energy::PrescribedTemperature,
    atmos,
    state⁺,
    args,
)
    @unpack aux⁻, state⁻, t = args
    FT = eltype(aux⁻)
    param_set = parameter_set(atmos)
    _T_0::FT = T_0(param_set)
    _cv_d::FT = cv_d(param_set)

    T = bc_energy.fn(state⁻, aux⁻, t)
    E_int⁺ = state⁺.ρ * _cv_d * (T - _T_0)
    state⁺.energy.ρe =
        E_int⁺ + state⁺.ρ * gravitational_potential(atmos.orientation, aux⁻)
end

function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::PrescribedTemperature,
    atmos,
    fluxᵀn,
    args,
)
    @unpack state⁻, aux⁻, diffusive⁻, hyperdiff⁻, t, n⁻ = args
    tend_type = Flux{SecondOrder}()
    _args⁻ = (;
        state = state⁻,
        aux = aux⁻,
        t,
        diffusive = diffusive⁻,
        hyperdiffusive = hyperdiff⁻,
    )
    pargs = merge(_args⁻, (precomputed = precompute(atmos, _args⁻, tend_type),))
    total_flux =
        Σfluxes(Energy(), eq_tends(Energy(), atmos, tend_type), atmos, pargs)
    nd_ρh_tot = dot(n⁻, total_flux)
    # both sides involve projections of normals, so signs are consistent
    fluxᵀn.energy.ρe += nd_ρh_tot
end


"""
    PrescribedEnergyFlux(fn) :: EnergyBC
Prescribe the net inward energy flux across the boundary by `fn`, a function
with signature `fn(state, aux, t)`, returning the flux (in W/m^2).
"""
struct PrescribedEnergyFlux{FN} <: EnergyBC
    fn::FN
end
function atmos_energy_boundary_state!(
    nf,
    bc_energy::PrescribedEnergyFlux,
    atmos,
    _...,
) end
function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::PrescribedEnergyFlux,
    atmos,
    fluxᵀn,
    args,
)
    @unpack state⁻, aux⁻, t = args

    # DG normal is defined in the outward direction
    # we want to prescribe the inward flux
    fluxᵀn.energy.ρe -= bc_energy.fn(state⁻, aux⁻, t)
end

"""
    Adiabaticθ(fn) :: EnergyBC
Prescribe the net inward potential temperature flux
across the boundary by `fn`, a function with signature
`fn(state, aux, t)`, returning the flux (in kgK/m^2).
"""
struct Adiabaticθ{FN} <: EnergyBC
    fn::FN
end
function atmos_energy_boundary_state!(nf, bc_energy::Adiabaticθ, atmos, _...) end
function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::Adiabaticθ,
    atmos,
    fluxᵀn,
    args,
)
    @unpack state⁻, aux⁻, t = args

    # DG normal is defined in the outward direction
    # we want to prescribe the inward flux
    fluxᵀn.energy.ρθ_liq_ice -= bc_energy.fn(state⁻, aux⁻, t)
end

"""
    BulkFormulaEnergy(fn) :: EnergyBC
Calculate the net inward energy flux across the boundary. The drag
coefficient is `C_h = fn_C_h(atmos, state, aux, t, normu_int_tan)`. The surface
temperature and q_tot are `T, q_tot = fn_T_and_q_tot(atmos, state, aux, t)`.
Return the flux (in W m^-2).
"""
struct BulkFormulaEnergy{FNX, FNTM} <: EnergyBC
    fn_C_h::FNX
    fn_T_and_q_tot::FNTM
end
function atmos_energy_boundary_state!(
    nf,
    bc_energy::BulkFormulaEnergy,
    atmos,
    _...,
) end
function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::BulkFormulaEnergy,
    atmos,
    fluxᵀn,
    args,
)
    @unpack state⁻, aux⁻, t, state_int⁻, aux_int⁻ = args

    u_int⁻ = state_int⁻.ρu / state_int⁻.ρ
    u_int⁻_tan = projection_tangential(atmos, aux_int⁻, u_int⁻)
    normu_int⁻_tan = norm(u_int⁻_tan)
    C_h = bc_energy.fn_C_h(atmos, state⁻, aux⁻, t, normu_int⁻_tan)
    T, q_tot = bc_energy.fn_T_and_q_tot(atmos, state⁻, aux⁻, t)
    param_set = parameter_set(atmos)

    # calculate MSE from the states at the boundary and at the interior point
    ts = PhaseEquil_ρTq(param_set, state⁻.ρ, T, q_tot)
    ts_int = recover_thermo_state(atmos, state_int⁻, aux_int⁻)
    e_pot = gravitational_potential(atmos.orientation, aux⁻)
    e_pot_int = gravitational_potential(atmos.orientation, aux_int⁻)
    MSE = moist_static_energy(ts, e_pot)
    MSE_int = moist_static_energy(ts_int, e_pot_int)

    # TODO: use the correct density at the surface
    ρ_avg = average_density(state⁻.ρ, state_int⁻.ρ)
    # NOTE: difference from design docs since normal points outwards
    fluxᵀn.energy.ρe -= C_h * ρ_avg * normu_int⁻_tan * (MSE - MSE_int)
end

"""
    NishizawaEnergyFlux(fn) :: EnergyBC
Calculate the net inward energy flux across the boundary following Nishizawa and Kitamura (2018).
Return the flux (in W m^-2).
"""
struct NishizawaEnergyFlux{FNTM, FNX} <: EnergyBC
    fn_z0::FNX
    fn_T_and_q_tot::FNTM
end
function atmos_energy_boundary_state!(
    nf,
    bc_energy::NishizawaEnergyFlux,
    atmos,
    _...,
) end
function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::NishizawaEnergyFlux,
    atmos,
    fluxᵀn,
    args,
)
    @unpack state⁻, aux⁻, t, aux_int⁻, state_int⁻ = args
    param_set = parameter_set(atmos)
    FT = eltype(state⁻)
    # Interior state
    u_int⁻ = state_int⁻.ρu / state_int⁻.ρ
    u_int⁻_tan = projection_tangential(atmos, aux_int⁻, u_int⁻)
    normu_int⁻_tan = norm(u_int⁻_tan)
    q_tot_int =
        moisture_model(atmos) isa DryModel ? FT(0) :
        state_int⁻.moisture.ρq_tot / state_int⁻.ρ
    # recover thermo state
    ts_int = recover_thermo_state(atmos, state_int⁻, aux_int⁻)
    x_in =
        MArray{Tuple{3}, FT}(FT[normu_int⁻_tan, dry_pottemp(ts_int), q_tot_int])

    # Boundary state
    T, q_tot = bc_energy.fn_T_and_q_tot(atmos, state⁻, aux⁻, t)
    x_s = MArray{Tuple{3}, FT}(FT[FT(0), T, q_tot])

    ## Initial guesses for MO parameters, these should be a function of state.
    LMO_init = FT(100) # Initial value so that ξ_init<<1
    u_star_init = FT(0.1) * normu_int⁻_tan
    th_star_init = T
    qt_star_init = q_tot
    MO_param_guess = MArray{Tuple{4}, FT}(FT[
        LMO_init,
        u_star_init,
        th_star_init,
        qt_star_init,
    ])

    # Roughness and interior heights
    z_0 = bc_energy.fn_z0(atmos, state⁻, aux⁻, t, normu_int⁻_tan)
    z_in = altitude(atmos, aux_int⁻)

    θ_flux, q_tot_flux = get_energy_flux(surface_conditions(
        param_set,
        MO_param_guess,
        x_in,
        x_s,
        z_0,
        T,
        z_in,
        DGScheme(),
    ))

    # recover thermo state
    ts_surf = PhaseEquil_ρTq(param_set, state⁻.ρ, T, q_tot)
    # Add sensible heat flux
    fluxᵀn.energy.ρe -= state⁻.ρ * θ_flux * cp_m(ts_surf)
    # Add latent heat flux
    fluxᵀn.energy.ρe -= state⁻.ρ * q_tot_flux * latent_heat_vapor(param_set, T)
end
