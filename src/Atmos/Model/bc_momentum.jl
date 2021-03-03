abstract type MomentumBC end

"""
    ImpenetrableFreeSlip <: MomentumBC

Defines an impenetrable free-slip wall model for momentum. This implies:
  - no flow in the direction normal to the boundary, and
  - No surface drag on momentum parallel to the boundary.
"""
struct ImpenetrableFreeSlip <: MomentumBC end

function atmos_momentum_boundary_state!(
    nf::NumericalFluxFirstOrder,
    bc_momentum::ImpenetrableFreeSlip,
    atmos,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    t,
    args...,
)
    state⁺.ρu -= 2 * dot(state⁻.ρu, n) .* SVector(n)
end
function atmos_momentum_boundary_state!(
    nf::NumericalFluxGradient,
    bc_momentum::ImpenetrableFreeSlip,
    atmos,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    t,
    args...,
)
    state⁺.ρu -= dot(state⁻.ρu, n) .* SVector(n)
end
function atmos_momentum_normal_boundary_flux_second_order!(
    nf,
    bc_momentum::ImpenetrableFreeSlip,
    atmos,
    args...,
) end


"""
    ImpenetrableNoSlip <: MomentumBC

Defines an impenetrable no-slip wall model for momentum. This implies:
  - no flow in the direction normal to the boundary, and
  - Zero momentum at the boundary.
"""
struct ImpenetrableNoSlip <: MomentumBC end

function atmos_momentum_boundary_state!(
    nf::NumericalFluxFirstOrder,
    bc_momentum::ImpenetrableNoSlip,
    atmos,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    t,
    args...,
)
    state⁺.ρu = -state⁻.ρu
end
function atmos_momentum_boundary_state!(
    nf::NumericalFluxGradient,
    bc_momentum::ImpenetrableNoSlip,
    atmos,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    t,
    args...,
)
    state⁺.ρu = zero(state⁺.ρu)
end
function atmos_momentum_normal_boundary_flux_second_order!(
    nf,
    bc_momentum::ImpenetrableNoSlip,
    atmos,
    args...,
) end


"""
    ImpenetrableDragLaw(fn) :: MomentumBC

Defines an impenetrable drag law wall model for momentum. This implies:
  - no flow in the direction normal to the boundary, and
  - Drag law for momentum parallel to the boundary.

The drag coefficient is
`C = fn(state, aux, t, normu_int_tan)`, where `normu_int_tan` is the internal speed
parallel to the boundary.
`_int` refers to the first interior node.
"""
struct ImpenetrableDragLaw{FN} <: MomentumBC
    fn::FN
end

function atmos_momentum_boundary_state!(
    nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    bc_momentum::ImpenetrableDragLaw,
    atmos,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    t,
    args...,
)
    atmos_momentum_boundary_state!(
        nf,
        ImpenetrableFreeSlip(),
        atmos,
        state⁺,
        aux⁺,
        n,
        state⁻,
        aux⁻,
        t,
        args...,
    )
end
function atmos_momentum_normal_boundary_flux_second_order!(
    nf,
    bc_momentum::ImpenetrableDragLaw,
    atmos,
    fluxᵀn,
    n,
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

    u1⁻ = state_int⁻.ρu / state_int⁻.ρ
    u_int⁻_tan = u1⁻ - dot(u1⁻, n) .* SVector(n)
    normu_int⁻_tan = norm(u_int⁻_tan)
    # NOTE: difference from design docs since normal points outwards
    C = bc_momentum.fn(state⁻, aux⁻, t, normu_int⁻_tan)
    τn = C * normu_int⁻_tan * u_int⁻_tan
    # both sides involve projections of normals, so signs are consistent
    fluxᵀn.ρu += state⁻.ρ * τn
    fluxᵀn.energy.ρe += state⁻.ρu' * τn
end
