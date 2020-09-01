abstract type MomentumBC end
abstract type MomentumDragBC end

"""
    Impenetrable(drag::MomentumDragBC) :: MomentumBC

Defines an impenetrable wall model for momentum. This implies:
  - no flow in the direction normal to the boundary, and
  - flow parallel to the boundary is subject to the `drag` condition.
"""
struct Impenetrable{D <: MomentumDragBC} <: MomentumBC
    drag::D
end

"""
    FreeSlip() :: MomentumDragBC

No surface drag on momentum parallel to the boundary.
"""
struct FreeSlip <: MomentumDragBC end



function atmos_momentum_boundary_state!(
    nf::NumericalFluxFirstOrder,
    bc_momentum::Impenetrable{FreeSlip},
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
    state⁺.ρu -= 2 * dot(state⁻.ρu, n) .* SVector(n)
end
function atmos_momentum_boundary_state!(
    nf::NumericalFluxGradient,
    bc_momentum::Impenetrable{FreeSlip},
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
    state⁺.ρu -= dot(state⁻.ρu, n) .* SVector(n)
end
function atmos_momentum_normal_boundary_flux_second_order!(
    nf,
    bc_momentum::Impenetrable{FreeSlip},
    atmos,
    args...,
) end



"""
    NoSlip() :: MomentumDragBC

Zero momentum at the boundary.
"""
struct NoSlip <: MomentumDragBC end

function atmos_momentum_boundary_state!(
    nf::NumericalFluxFirstOrder,
    bc_momentum::Impenetrable{NoSlip},
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
    state⁺.ρu = -state⁻.ρu
end
function atmos_momentum_boundary_state!(
    nf::NumericalFluxGradient,
    bc_momentum::Impenetrable{NoSlip},
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
    state⁺.ρu = zero(state⁺.ρu)
end
function atmos_momentum_normal_boundary_flux_second_order!(
    nf,
    bc_momentum::Impenetrable{NoSlip},
    atmos,
    args...,
) end


"""
    DragLaw(fn) :: MomentumDragBC

Drag law for momentum parallel to the boundary. The drag coefficient is
`C = fn(state, aux, t, normu_int_tan)`, where `normu_int_tan` is the internal speed
parallel to the boundary.
`_int` refers to the first interior node.
"""
struct DragLaw{FN} <: MomentumDragBC
    fn::FN
end
function atmos_momentum_boundary_state!(
    nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    bc_momentum::Impenetrable{DL},
    atmos,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    bctype,
    t,
    args...,
) where {DL <: DragLaw}
    atmos_momentum_boundary_state!(
        nf,
        Impenetrable(FreeSlip()),
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
end
function atmos_momentum_normal_boundary_flux_second_order!(
    nf,
    bc_momentum::Impenetrable{DL},
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
    bctype,
    t,
    state_int⁻,
    diffusive_int⁻,
    aux_int⁻,
) where {DL <: DragLaw}

    u1⁻ = state_int⁻.ρu / state_int⁻.ρ
    u_int⁻_tan = u1⁻ - dot(u1⁻, n) .* SVector(n)
    normu_int⁻_tan = norm(u_int⁻_tan)
    # NOTE: difference from design docs since normal points outwards
    C = bc_momentum.drag.fn(state⁻, aux⁻, t, normu_int⁻_tan)
    τn = C * normu_int⁻_tan * u_int⁻_tan
    # both sides involve projections of normals, so signs are consistent
    fluxᵀn.ρu += state⁻.ρ * τn
    fluxᵀn.ρe += state⁻.ρu' * τn
end
