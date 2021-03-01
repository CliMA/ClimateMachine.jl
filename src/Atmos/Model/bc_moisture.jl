"""
    Impermeable{PV <: Moisture} <: BCDef{PV}

No moisture flux.
"""
struct Impermeable{PV <: Moisture} <: BCDef{PV} end

# TODO: Remove this definition, since it's unclear which prognostic
# variable this is defined for.
Impermeable() = Impermeable{TotalMoisture}()

bc_val(bc::Impermeable, atmos::AtmosModel, ::NF12∇, args) = DefaultBCValue()

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
struct PrescribedMoistureFlux{PV <: TotalMoisture, FN} <: BCDef{PV}
    fn::FN
end
PrescribedMoistureFlux(fn::FN) where {FN} =
    PrescribedMoistureFlux{TotalMoisture, FN}(fn)

function atmos_moisture_normal_boundary_flux_second_order!(
    nf,
    bc_moisture::PrescribedMoistureFlux,
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

    nρd_q_tot = -bc_moisture.fn(state⁻, aux⁻, t)
    fluxᵀn.ρ += nρd_q_tot
    fluxᵀn.ρu += nρd_q_tot / state⁻.ρ .* state⁻.ρu
    # assumes EquilMoist
    fluxᵀn.moisture.ρq_tot += nρd_q_tot
end

"""
    BulkFormulaMoisture(fn) :: MoistureBC

Calculate the net inward moisture flux across the boundary using
the bulk formula. The drag coefficient is `C_q = fn_C_q(state, aux,
t, normu_int_tan)`. The surface q_tot at the boundary is `q_tot =
fn_q_tot(state, aux, t)`.

Return the flux (in kg m^-2 s^-1).
"""
struct BulkFormulaMoisture{PV <: TotalMoisture, FNX, FNM} <: BCDef{PV}
    fn_C_q::FNX
    fn_q_tot::FNM
end
BulkFormulaMoisture(fn_C_q::FNX, fn_q_tot::FNM) where {FNX, FNM} =
    BulkFormulaMoisture{TotalMoisture, FNX, FNM}(fn_C_q, fn_q_tot)

function atmos_moisture_normal_boundary_flux_second_order!(
    nf,
    bc_moisture::BulkFormulaMoisture,
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
    C_q = bc_moisture.fn_C_q(state⁻, aux⁻, t, normu_int⁻_tan)
    q_tot = bc_moisture.fn_q_tot(state⁻, aux⁻, t)
    q_tot_int = state_int⁻.moisture.ρq_tot / state_int⁻.ρ

    # TODO: use the correct density at the surface
    ρ_avg = average_density(state⁻.ρ, state_int⁻.ρ)
    # NOTE: difference from design docs since normal points outwards
    fluxᵀn.moisture.ρq_tot -= C_q * ρ_avg * normu_int⁻_tan * (q_tot - q_tot_int)
    fluxᵀn.ρ -= C_q * ρ_avg * normu_int⁻_tan * (q_tot - q_tot_int)
    fluxᵀn.ρu -= C_q * normu_int⁻_tan * (q_tot - q_tot_int) .* state_int⁻.ρu
end
