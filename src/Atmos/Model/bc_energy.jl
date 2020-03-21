abstract type EnergyBC end

"""
    Insulating() :: EnergyBC

No energy flux across the boundary.
"""
struct Insulating <: EnergyBC end
function atmos_energy_boundary_state!(nf, bc_energy::Insulating, atmos, args...) end
function atmos_energy_normal_boundary_flux_diffusive!(
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
    T = bc_energy.fn(state⁻, aux⁻, t)
    E_int⁺ = state⁺.ρ * cv_d * (T - T_0)
    state⁺.ρe =
        E_int⁺ + state⁺.ρ * gravitational_potential(atmos.orientation, aux⁻)
end
function atmos_energy_normal_boundary_flux_diffusive!(
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
    ν, _ = turbulence_tensors(atmos.turbulence, state⁻, diff⁻, aux⁻, t)
    D_t = (ν isa Real ? ν : diag(ν)) * inv_Pr_turb
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
function atmos_energy_normal_boundary_flux_diffusive!(
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
