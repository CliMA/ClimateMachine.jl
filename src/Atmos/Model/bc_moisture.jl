abstract type MoistureBC end

"""
    Impermeable() :: MoistureBC

No moisture flux.
"""
struct Impermeable <: MoistureBC end
function atmos_moisture_boundary_state!(
    nf,
    bc_moisture::Impermeable,
    atmos,
    args...,
) end
function atmos_moisture_normal_boundary_flux_second_order!(
    nf,
    bc_moisture::Impermeable,
    atmos,
    args...,
) end


"""
    PrescribedMoistureFlux(fn) :: MoistureBC

Prescribe the net inward moisture flux across the boundary by `fn`, a function
with signature `fn(state, aux, t)`, returning the flux (in kg/m^2).
"""
struct PrescribedMoistureFlux{FN} <: MoistureBC
    fn::FN
end
function atmos_moisture_boundary_state!(
    nf,
    bc_moisture::PrescribedMoistureFlux,
    atmos,
    args...,
) end
function atmos_moisture_normal_boundary_flux_second_order!(
    nf,
    bc_moisture::PrescribedMoistureFlux,
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

    nρd_q_tot = -bc_moisture.fn(state⁻, aux⁻, t)
    fluxᵀn.ρ += nρd_q_tot
    fluxᵀn.ρu += nρd_q_tot / state⁻.ρ .* state⁻.ρu
    # assumes EquilMoist
    fluxᵀn.moisture.ρq_tot += nρd_q_tot
end

struct BulkFormulationMoisture{FN} <: MoistureBC
    fn::FN
end
function atmos_moisture_boundary_state!(
    nf,
    bc_moisture::BulkFormulationMoisture,
    atmos,
    args...,
) end
function atmos_moisture_normal_boundary_flux_diffusive!(
    nf,
    bc_moisture::BulkFormulationMoisture,
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
    state1⁻,
    args...,
)
    FT = eltype(state⁺)
    u1⁻ = state1⁻.ρu / state1⁻.ρ
    Pu1⁻ = u1⁻ - dot(u1⁻, n⁻) .* SVector(n⁻)
    normPu1⁻ = norm(Pu1⁻)
    # NOTE: difference from design docs since normal points outwards
    C = bc_moisture.fn(state⁻, aux⁻, t, normPu1⁻)
    τe = C * normPu1⁻
    TS = thermo_state(atmos, atmos.moisture, state⁻, aux⁻)
    q_surf = state⁻.moisture.ρq_tot / state⁻.ρ#q_vap_saturation(TS)
    # both sides involve projections of normals, so signs are consistent
    fluxᵀn.moisture.ρq_tot -=
        state⁻.ρ * τe * (q_surf - state1⁻.moisture.ρq_tot / state1⁻.ρ)
    fluxᵀn.ρ -= state⁻.ρ * τe * (q_surf - state1⁻.moisture.ρq_tot / state1⁻.ρ)
    fluxᵀn.ρu -= τe * (q_surf - state1⁻.moisture.ρq_tot / state1⁻.ρ) .* state⁻.ρu
end
