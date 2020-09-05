abstract type EnergyBC <: BoundaryCondition end

using ..TurbulenceClosures

"""
    Insulating() :: EnergyBC

No energy flux across the boundary.
"""
struct Insulating <: EnergyBC end
function boundary_state!(nf, bc_energy::Insulating, atmos::AtmosModel, args...) end


"""
    PrescribedTemperature(fn) :: EnergyBC

Prescribe the temperature at the boundary by `fn`, a function with signature
`fn(state, aux, t)` returning the temperature (in K).
"""
struct PrescribedTemperature{FN} <: EnergyBC
    fn::FN
end
function boundary_state!(
    nf,
    bc_energy::PrescribedTemperature,
    atmos::AtmosModel,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
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


"""
    PrescribedEnergyFlux(fn) :: EnergyBC

Prescribe the net inward energy flux across the boundary by `fn`, a function
with signature `fn(state, aux, t)`, returning the flux (in W/m^2).
"""
struct PrescribedEnergyFlux{FN} <: EnergyBC
    fn::FN
end
function boundary_state!(
    nf,
    bc_energy::PrescribedEnergyFlux,
    atmos::AtmosModel,
    args...,
)
    nothing
end
function numerical_boundary_flux_second_order!(
    nf,
    bc_energy::PrescribedEnergyFlux,
    atmos::AtmosModel,
    fluxᵀn::Vars,
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
    fluxᵀn.ρe -= bc_energy.fn(state⁻, aux⁻, t)
end

"""
    BulkFormulaEnergy(fn) :: EnergyBC

Calculate the net inward energy flux across the boundary. The drag
coefficient is `C_h = fn_C_h(state, aux, t, normu_int_tan)`. The surface
temperature and q_tot are `T, q_tot = fn_T_and_q_tot(state, aux, t)`.

Return the flux (in W m^-2).
"""
struct BulkFormulaEnergy{FNX, FNTM} <: EnergyBC
    fn_C_h::FNX
    fn_T_and_q_tot::FNTM
end
function boundary_state!(
    nf,
    bc_energy::BulkFormulaEnergy,
    atmos,
    args...,
) end
function numerical_boundary_flux_second_order!(
    nf,
    bc_energy::BulkFormulaEnergy,
    atmos::AtmosModel,
    fluxᵀn::Vars,
    n⁻,
    state⁻,
    diffusive⁻,
    hyperdiffusive⁻,
    aux⁻,
    state⁺,
    diffusive⁺,
    hyperdiffusive⁺,
    aux⁺,
    bctype,
    t,
    state_int⁻,
    diffusive_int⁻,
    aux_int⁻,
)

    u_int⁻ = state_int⁻.ρu / state_int⁻.ρ
    u_int⁻_tan = projection_tangential(atmos, aux_int⁻, u_int⁻)
    normu_int⁻_tan = norm(u_int⁻_tan)
    C_h = bc_energy.fn_C_h(state⁻, aux⁻, t, normu_int⁻_tan)
    T, q_tot = bc_energy.fn_T_and_q_tot(state⁻, aux⁻, t)

    # calculate MSE from the states at the boundary and at the interior point
    ts = TemperatureSHumEquil(atmos.param_set, T, state⁻.ρ, q_tot)
    ts_int = thermo_state(atmos, atmos.moisture, state_int⁻, aux_int⁻)
    e_pot = gravitational_potential(atmos.orientation, aux⁻)
    e_pot_int = gravitational_potential(atmos.orientation, aux_int⁻)
    MSE = moist_static_energy(ts, e_pot)
    MSE_int = moist_static_energy(ts_int, e_pot_int)

    # TODO: use the correct density at the surface
    ρ_avg = average_density(state⁻.ρ, state_int⁻.ρ)
    # NOTE: difference from design docs since normal points outwards
    fluxᵀn.ρe -= C_h * ρ_avg * normu_int⁻_tan * (MSE - MSE_int)
end
