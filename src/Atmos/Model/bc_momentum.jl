abstract type MomentumBC <: BoundaryCondition end
abstract type MomentumDragBC <: BoundaryCondition end

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



function boundary_state!(
    nf::NumericalFluxFirstOrder,
    bc_momentum::Impenetrable{FreeSlip},
    atmos::AtmosModel,
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
function boundary_state!(
    nf::NumericalFluxGradient,
    bc_momentum::Impenetrable{FreeSlip},
    atmos::AtmosModel,
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
function numerical_boundary_flux_second_order!(
    nf,
    bc_momentum::Impenetrable{FreeSlip},
    atmos::AtmosModel,
    fluxᵀn::Vars,
    args...,
)
    nothing
end



"""
    NoSlip() :: MomentumDragBC

Zero momentum at the boundary.
"""
struct NoSlip <: MomentumDragBC end

function boundary_state!(
    nf::NumericalFluxFirstOrder,
    bc_momentum::Impenetrable{NoSlip},
    atmos::AtmosModel,
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
function boundary_state!(
    nf::NumericalFluxGradient,
    bc_momentum::Impenetrable{NoSlip},
    atmos::AtmosModel,
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
function numerical_boundary_flux_second_order!(
    nf,
    bc_momentum::Impenetrable{NoSlip},
    atmos::AtmosModel,
    fluxᵀn::Vars,
    args...,
) 
    nothing
end


"""
    DragLaw(fn) :: MomentumDragBC

Drag law for momentum parallel to the boundary. The drag coefficient is
`C = fn(state, aux, t, normPu_int)`, where `normPu_int⁻` is the internal speed
parallel to the boundary.
"""
struct DragLaw{FN} <: MomentumDragBC
    fn::FN
end
function boundary_state!(
    nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    bc_momentum::Impenetrable{DL},
    atmos::AtmosModel,
    args...,
) where {DL <: DragLaw}
    boundary_state!(
        nf,
        Impenetrable(FreeSlip()),
        atmos,
        args...,
    )
end
function numerical_boundary_flux_second_order!(
    nf,
    bc_momentum::Impenetrable{DL},
    atmos::AtmosModel,
    fluxᵀn::Vars,
    n,
    state⁻,
    diff⁻,
    hyperdiff⁻,
    aux⁻,
    state⁺,
    diff⁺,
    hyperdiff⁺,
    aux⁺,
    t,
    state1⁻,
    diff1⁻,
    aux1⁻,
) where {DL <: DragLaw}

    u1⁻ = state1⁻.ρu / state1⁻.ρ
    Pu1⁻ = u1⁻ - dot(u1⁻, n) .* SVector(n)
    normPu1⁻ = norm(Pu1⁻)
    # NOTE: difference from design docs since normal points outwards
    C = bc_momentum.drag.fn(state⁻, aux⁻, t, normPu1⁻)
    τn = C * normPu1⁻ * Pu1⁻
    # both sides involve projections of normals, so signs are consistent
    fluxᵀn.ρu += state⁻.ρ * τn
    fluxᵀn.ρe += state⁻.ρu' * τn
end
