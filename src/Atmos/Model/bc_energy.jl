abstract type EnergyBC end

using ..TurbulenceClosures

"""
    Insulating() :: EnergyBC

No energy flux across the boundary.
"""
struct Insulating <: EnergyBC end
function atmos_energy_boundary_state!(nf, bc_energy::Insulating, atmos, args...) end
function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::Insulating,
    atmos,
    args...,
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
    aux⁺,
    n,
    state⁻,
    aux⁻,
    bctype,
    t,
    args...,
)
    FT = eltype(aux⁻)
    _T_0::FT = T_0(atmos.param_set)
    _cv_d::FT = cv_d(atmos.param_set)

    T = bc_energy.fn(state⁻, aux⁻, t)
    E_int⁺ = state⁺.ρ * _cv_d * (T - _T_0)
    state⁺.ρe =
        E_int⁺ + state⁺.ρ * gravitational_potential(atmos.orientation, aux⁻)
end
function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::PrescribedTemperature,
    atmos,
    fluxᵀn,
    n⁻,
    state⁻,
    diff⁻,
    hyperdiff⁻,
    aux⁻,
    state⁺,
    diff⁺,
    hyperdiff⁺,
    aux⁺,
    bctype,
    t,
    args...,
)

    # TODO: figure out a better way...
    ν, D_t, _ = turbulence_tensors(atmos, state⁻, diff⁻, aux⁻, t)
    d_h_tot = -D_t .* diff⁻.∇h_tot
    nd_h_tot = dot(n⁻, d_h_tot)
    # both sides involve projections of normals, so signs are consistent
    fluxᵀn.ρe += nd_h_tot * state⁻.ρ
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
    args...,
) end
function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::PrescribedEnergyFlux,
    atmos,
    fluxᵀn,
    n⁻,
    state⁻,
    diff⁻,
    hyperdiff⁻,
    aux⁻,
    state⁺,
    diff⁺,
    hyperdiff⁺,
    aux⁺,
    bctype,
    t,
    args...,
)

    # DG normal is defined in the outward direction
    # we want to prescribe the inward flux
    fluxᵀn.ρe -= bc_energy.fn(state⁻, aux⁻, t)
end

"""
    BulkFormulaEnergy(fn) :: EnergyBC

Calculate the net inward energy flux across the boundary.
The drag coefficient is `C_h = fn_C_h(state, aux, t, normu_int_tan)`.
The surface temp is `T_sfc= fn_T_sfc(state, aux, t)`.
The surface q_tot is `q_tot_sfc = fn_q_tot_sfc(state, aux, t)`.
Return the flux (in W m^-2).
"""
struct BulkFormulaEnergy{FNX, FNT, FNM} <: EnergyBC
    fn_C_h::FNX
    fn_T_sfc::FNT
    fn_q_tot_sfc::FNM
end
function atmos_energy_boundary_state!(
    nf,
    bc_energy::BulkFormulaEnergy,
    atmos,
    args...,
) end
function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::BulkFormulaEnergy,
    atmos,
    fluxᵀn,
    n⁻,
    state_sfc⁻,
    diff_sfc⁻,
    hyperdiff_sfc⁻,
    aux_sfc⁻,
    state_sfc⁺,
    diff_sfc⁺,
    hyperdiff_sfc⁺,
    aux_sfc⁺,
    bctype,
    t,
    state_int⁻,
    diff_int⁻,
    aux_int⁻,
)

    u_int⁻ = state_int⁻.ρu / state_int⁻.ρ
    u_int⁻_tan = projection_tangential(atmos, aux_int⁻, u_int⁻)
    normu_int⁻_tan = norm(u_int⁻_tan)
    C_h = bc_energy.fn_C_h(state_sfc⁻, aux_sfc⁻, t, normu_int⁻_tan)
    T_sfc = bc_energy.fn_T_sfc(state_sfc⁻, aux_sfc⁻, t)
    q_tot_sfc = bc_energy.fn_q_tot_sfc(state_sfc⁻, aux_sfc⁻, t)

    # calculate MSE from the states at the surface and at the interior point
    ts_sfc =
        TemperatureSHumEquil(atmos.param_set, T_sfc, state_sfc⁻.ρ, q_tot_sfc)
    ts_int = thermo_state(atmos, atmos.moisture, state_int⁻, aux_int⁻)
    e_pot_sfc = gravitational_potential(atmos.orientation, aux_sfc⁻)
    e_pot_int = gravitational_potential(atmos.orientation, aux_int⁻)
    MSE_sfc = moist_static_energy(ts_sfc, e_pot_sfc)
    MSE_int = moist_static_energy(ts_int, e_pot_int)

    ρ_avg = average_density_sfc_int(state_sfc⁻.ρ, state_int⁻.ρ)
    # NOTE: difference from design docs since normal points outwards
    fluxᵀn.ρe -= C_h * ρ_avg * normu_int⁻_tan * (MSE_sfc - MSE_int)
end
