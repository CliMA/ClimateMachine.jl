using ..TurbulenceClosures
using ClimateMachine.SurfaceFluxes:
    get_energy_flux, surface_conditions, DGScheme

"""
    Insulating()
No energy flux across the boundary.
"""
struct Insulating{PV <: Energy} <: BCDef{PV} end
Insulating() = Insulating{Energy}()

bc_val(bc::Insulating, atmos::AtmosModel, ::NF12∇, args) = DefaultBCValue()

function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::Insulating,
    atmos,
    args...,
) end


"""
    PrescribedTemperature(fn)
Prescribe the temperature at the boundary by `fn`, a function with signature
`fn(state, aux, t)` returning the temperature (in K).
"""
struct PrescribedTemperature{PV <: Energy, FN} <: BCDef{PV}
    fn::FN
end
PrescribedTemperature(fn::FN) where {FN} = PrescribedTemperature{Energy, FN}(fn)

function bc_val(
    bc::PrescribedTemperature{Energy},
    atmos::AtmosModel,
    ::Union{NF1, NF∇},
    args,
)
    @unpack state, aux, t, n = args
    FT = eltype(aux)
    _T_0::FT = T_0(atmos.param_set)
    _cv_d::FT = cv_d(atmos.param_set)

    T = bc.fn(state, aux, t)
    return state.ρ * _cv_d * (T - _T_0) +
           state.ρ * gravitational_potential(atmos.orientation, aux)
end

function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::PrescribedTemperature,
    atmos,
    fluxᵀn,
    n⁻,
    state⁻,
    diffusive⁻,
    hyperdiffusive⁻,
    aux⁻,
    state⁺,
    diffusive⁺,
    hyperdiffusive⁺,
    aux⁺,
    t,
    args...,
)

    # TODO: figure out a better way...
    ν, D_t, _ = turbulence_tensors(atmos, state⁻, diffusive⁻, aux⁻, t)
    d_h_tot = -D_t .* diffusive⁻.energy.∇h_tot
    nd_h_tot = dot(n⁻, d_h_tot)
    # both sides involve projections of normals, so signs are consistent
    fluxᵀn.energy.ρe += nd_h_tot * state⁻.ρ
end


# TODO: Rename to PrescribedFlux
"""
    PrescribedEnergyFlux(fn)
Prescribe the net inward energy flux across the boundary by `fn`, a function
with signature `fn(state, aux, t)`, returning the flux (in W/m^2).
"""
struct PrescribedEnergyFlux{PV <: Energy, FN} <: BCDef{PV}
    fn::FN
end
PrescribedEnergyFlux(fn::FN) where {FN} = PrescribedEnergyFlux{Energy, FN}(fn)

function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::PrescribedEnergyFlux,
    atmos,
    fluxᵀn,
    n⁻,
    state⁻,
    diffusive⁻,
    hyperdiffusive⁻,
    aux⁻,
    state⁺,
    diffusive⁺,
    hyperdiffusive⁺,
    aux⁺,
    t,
    args...,
)

    # DG normal is defined in the outward direction
    # we want to prescribe the inward flux
    fluxᵀn.energy.ρe -= bc_energy.fn(state⁻, aux⁻, t)
end

"""
    Adiabaticθ(fn)
Prescribe the net inward potential temperature flux
across the boundary by `fn`, a function with signature
`fn(state, aux, t)`, returning the flux (in kgK/m^2).
"""
struct Adiabaticθ{PV <: ρθ_liq_ice, FN} <: BCDef{PV}
    fn::FN
end
Adiabaticθ(fn::FN) where {FN} = Adiabaticθ{ρθ_liq_ice, FN}(fn)

function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::Adiabaticθ,
    atmos,
    fluxᵀn,
    n⁻,
    state⁻,
    diffusive⁻,
    hyperdiffusive⁻,
    aux⁻,
    state⁺,
    diffusive⁺,
    hyperdiffusive⁺,
    aux⁺,
    t,
    args...,
)
    # DG normal is defined in the outward direction
    # we want to prescribe the inward flux
    fluxᵀn.energy.ρθ_liq_ice -= bc_energy.fn(state⁻, aux⁻, t)
end

# TODO: rename to BulkFormula
"""
    BulkFormulaEnergy(fn)
Calculate the net inward energy flux across the boundary. The drag
coefficient is `C_h = fn_C_h(atmos, state, aux, t, normu_int_tan)`. The surface
temperature and q_tot are `T, q_tot = fn_T_and_q_tot(atmos, state, aux, t)`.
Return the flux (in W m^-2).
"""
struct BulkFormulaEnergy{PV <: Energy, FNX, FNTM} <: BCDef{PV}
    fn_C_h::FNX
    fn_T_and_q_tot::FNTM
end
BulkFormulaEnergy(fn_C_h::FNX, fn_T_and_q_tot::FNTM) where {FNX, FNTM} =
    BulkFormulaEnergy{Energy, FNX, FNTM}(fn_C_h, fn_T_and_q_tot)

function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::BulkFormulaEnergy,
    atmos,
    fluxᵀn,
    n⁻,
    state⁻,
    diffusive⁻,
    hyperdiffusive⁻,
    aux⁻,
    state⁺,
    diffusive⁺,
    hyperdiffusive⁺,
    aux⁺,
    t,
    state_int⁻,
    diffusive_int⁻,
    aux_int⁻,
)

    u_int⁻ = state_int⁻.ρu / state_int⁻.ρ
    u_int⁻_tan = projection_tangential(atmos, aux_int⁻, u_int⁻)
    normu_int⁻_tan = norm(u_int⁻_tan)
    C_h = bc_energy.fn_C_h(atmos, state⁻, aux⁻, t, normu_int⁻_tan)
    T, q_tot = bc_energy.fn_T_and_q_tot(atmos, state⁻, aux⁻, t)

    # calculate MSE from the states at the boundary and at the interior point
    ts = PhaseEquil_ρTq(atmos.param_set, state⁻.ρ, T, q_tot)
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
    NishizawaEnergyFlux(fn)
Calculate the net inward energy flux across the boundary following Nishizawa and Kitamura (2018).
Return the flux (in W m^-2).
"""
struct NishizawaEnergyFlux{PV <: Energy, FNX, FNTM} <: BCDef{PV}
    fn_z0::FNX
    fn_T_and_q_tot::FNTM
end
NishizawaEnergyFlux(fn_z0::FNX, fn_T_and_q_tot::FNTM) where {FNX, FNTM} =
    NishizawaEnergyFlux{Energy, FNX, FNTM}(fn_z0, fn_T_and_q_tot)

function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::NishizawaEnergyFlux,
    atmos,
    fluxᵀn,
    n⁻,
    state⁻,
    diffusive⁻,
    hyperdiffusive⁻,
    aux⁻,
    state⁺,
    diffusive⁺,
    hyperdiffusive⁺,
    aux⁺,
    t,
    state_int⁻,
    diffusive_int⁻,
    aux_int⁻,
)
    FT = eltype(state⁻)
    # Interior state
    u_int⁻ = state_int⁻.ρu / state_int⁻.ρ
    u_int⁻_tan = projection_tangential(atmos, aux_int⁻, u_int⁻)
    normu_int⁻_tan = norm(u_int⁻_tan)
    q_tot_int =
        atmos.moisture isa DryModel ? FT(0) :
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
        atmos.param_set,
        MO_param_guess,
        x_in,
        x_s,
        z_0,
        T,
        z_in,
        DGScheme(),
    ))

    # recover thermo state
    ts_surf = PhaseEquil_ρTq(atmos.param_set, state⁻.ρ, T, q_tot)
    # Add sensible heat flux
    fluxᵀn.energy.ρe -= state⁻.ρ * θ_flux * cp_m(ts_surf)
    # Add latent heat flux
    fluxᵀn.energy.ρe -=
        state⁻.ρ * q_tot_flux * latent_heat_vapor(atmos.param_set, T)
end
