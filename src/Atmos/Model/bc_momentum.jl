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
`C = fn(state, aux, t, normPu_int)`, where `normPu_int⁻` is the internal speed
parallel to the boundary.
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
    bc_extra_data,
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
        bc_extra_data,
    )
end

function bottom1_velocity(atmos, bc_extra_data)
    elem = bc_extra_data.elem
    face_dof = bc_extra_data.face_dof
    offset = bc_extra_data.offset
    state_conservative = bc_extra_data.state_conservative
    FT = eltype(state_conservative)
    @inbounds @views begin
        state = state_conservative[face_dof + offset, :, elem]
        ρ = Vars{vars_state_conservative(atmos, FT)}(state).ρ
        ρu = Vars{vars_state_conservative(atmos, FT)}(state).ρu
    end
    return ρu / ρ
end

function atmos_momentum_normal_boundary_flux_second_order!(
    nf,
    bc_momentum::Impenetrable{DL},
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
    bc_extra_data,
) where {DL <: DragLaw}
    u1⁻ = bottom1_velocity(atmos, bc_extra_data)

    Pu1⁻ = u1⁻ - dot(u1⁻, n) .* SVector(n)
    normPu1⁻ = norm(Pu1⁻)
    # NOTE: difference from design docs since normal points outwards
    C = bc_momentum.drag.fn(state⁻, aux⁻, t, normPu1⁻)
    τn = C * normPu1⁻ * Pu1⁻
    # both sides involve projections of normals, so signs are consistent
    fluxᵀn.ρu += state⁻.ρ * τn
    fluxᵀn.ρe += state⁻.ρu' * τn
end
