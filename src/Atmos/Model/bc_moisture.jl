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



"""
    BulkFormulaMoisture(fn) :: MoistureBC

Bulk formula moisture boundary condition calculation using exchange coefficients from the Monin Obukhov 
Surface Flux parameterisation
"""
struct BulkFormulaMoisture{FN} <: MoistureBC
    fn::FN
end
function atmos_momentum_boundary_state!(
    nf,
    bc_moisture::BulkFormulaMoisture,
    atmos,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    bctype,
    t,
    args...,
) end
function atmos_momentum_normal_boundary_flux_second_order!(
    nf,
    bc_moisture::BulkFormulaMoisture,
    atmos,
    fluxᵀn,
    n,
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
    diff1⁻,
    aux1⁻,
)

#    u1⁻ = state1⁻.ρu / state1⁻.ρ
#    Pu1⁻ = u1⁻ - dot(u1⁻, n⁻) .* SVector(n⁻)
#    normPu1⁻ = norm(Pu1⁻)
#    # DG normal is defined in the outward direction
#    # we want to prescribe the inward flux
#    sfc = surfaceconditions(...);
#    #TODO fill in surface conditions
#    C_q = sfc.K_exchange[3]; # Exchange coefficient for energy
#    Δρq_tot = state1⁻.moisture.ρq_tot - state⁻.ρ * q_tot 
#    fluxᵀn.moisture.ρq_tot -= -C_q * normPu1⁻ * Δρq_tot
end
