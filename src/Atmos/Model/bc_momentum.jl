abstract type MomentumBC end
abstract type MomentumDragBC end

"""
    Impenetrable(drag::MomentumDragBC) :: MomentumBC

Defines an impenetrable wall model for momentum. This implies:
  - no flow in the direction normal to the boundary, and
  - flow parallel to the boundary is subject to the `drag` condition.
"""
struct Impenetrable{D, PV <: Union{Momentum, Energy}} <: BCDef{PV}
    drag::D
end

Impenetrable{PV}(drag::D) where {D, PV} = Impenetrable{D, PV}(drag)
Impenetrable(drag::D) where {D} =
    (Impenetrable{D, Momentum}(drag), Impenetrable{D, Energy}(drag))

"""
    FreeSlip() :: MomentumDragBC

No surface drag on momentum parallel to the boundary.
"""
struct FreeSlip <: MomentumDragBC end

function bc_val(::Impenetrable{FreeSlip, Momentum}, ::AtmosModel, ::NF1, args)
    @unpack state, n = args
    return state.ρu - 2 * dot(state.ρu, n) .* SVector(n)
end

function bc_val(::Impenetrable{FreeSlip, Momentum}, ::AtmosModel, ::NF∇, args)
    @unpack state, n = args
    return state.ρu - dot(state.ρu, n) .* SVector(n)
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

function bc_val(::Impenetrable{NoSlip, Momentum}, ::AtmosModel, ::NF1, args)
    return -args.state⁻.ρu
end

function bc_val(::Impenetrable{NoSlip, Momentum}, ::AtmosModel, ::NF2, args)
    @unpack state, n = args
    return state.ρu - dot(state.ρu, n) .* SVector(n)
end

function bc_val(::Impenetrable{NoSlip, Momentum}, ::AtmosModel, ::NF∇, args)
    return zero(args.state.ρu)
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
    fluxᵀn.energy.ρe += state⁻.ρu' * τn
end

function bc_val(
    bc::Impenetrable{D, Momentum},
    atmos::AtmosModel,
    nf::Union{NF1, NF∇},
    args,
) where {D <: DragLaw}
    return bc_val(Impenetrable{FreeSlip, Momentum}(FreeSlip()), atmos, nf, args)
end

function compute_τn(bc, args)
    @unpack state, state_int, n = args

    u1⁻ = state_int.ρu / state_int.ρ
    u_int_tan = u1 - dot(u1, n) .* SVector(n)
    normu_int_tan = norm(u_int_tan)
    # NOTE: difference from design docs since normal points outwards
    C = bc.drag.fn(state, aux, t, normu_int_tan)
    τn = C * normu_int_tan * u_int_tan
end

function bc_val(
    bc::Impenetrable{D, Momentum},
    ::AtmosModel,
    ::NF2,
    args,
) where {D <: DragLaw}
    @unpack state = args
    τn = compute_τn(bc, args)
    # both sides involve projections of normals, so signs are consistent
    return state.ρ + state.ρ * τn
end

function bc_val(
    ::Impenetrable{D, Energy},
    ::AtmosModel,
    ::NF2,
    args,
) where {D <: DragLaw}
    @unpack state = args

    τn = compute_τn(bc, args)
    # both sides involve projections of normals, so signs are consistent
    return state.energy.ρe + state.ρu' * τn
end
