"""
    ImpenetrableFreeSlip() :: BCDef

Defines an impenetrable wall model for momentum. This implies:
  - no flow in the direction normal to the boundary, and
  - free slip for the tangential components
"""
struct ImpenetrableFreeSlip{PV <: Union{Momentum, Energy}} <: BCDef{PV} end

ImpenetrableFreeSlip() =
    (ImpenetrableFreeSlip{Momentum}(), ImpenetrableFreeSlip{Energy}())

function bc_val(::ImpenetrableFreeSlip{Momentum}, ::AtmosModel, ::NF1, args)
    @unpack state, n = args
    return state.ρu - 2 * dot(state.ρu, n) .* SVector(n)
end

function bc_val(::ImpenetrableFreeSlip{Momentum}, ::AtmosModel, ::NF∇, args)
    @unpack state, n = args
    return state.ρu - dot(state.ρu, n) .* SVector(n)
end

function atmos_momentum_normal_boundary_flux_second_order!(
    nf,
    bc_momentum::ImpenetrableFreeSlip,
    atmos,
    args...,
) end


"""
    ImpenetrableNoSlip() :: BCDef

Defines an impenetrable wall model for momentum. This implies:
  - no flow in the direction normal to the boundary, and
  - no slip for the tangential components
"""
struct ImpenetrableNoSlip{PV <: Union{Momentum, Energy}} <: BCDef{PV} end

ImpenetrableNoSlip() = (ImpenetrableNoSlip{Momentum}(),)

function bc_val(::ImpenetrableNoSlip{Momentum}, ::AtmosModel, ::NF1, args)
    return -args.state⁻.ρu
end

function bc_val(::ImpenetrableNoSlip{Momentum}, ::AtmosModel, ::NF2, args)
    @unpack state, n = args
    return state.ρu - dot(state.ρu, n) .* SVector(n)
end

function bc_val(::ImpenetrableNoSlip{Momentum}, ::AtmosModel, ::NF∇, args)
    return zero(args.state.ρu)
end

function atmos_momentum_normal_boundary_flux_second_order!(
    nf,
    bc_momentum::ImpenetrableNoSlip,
    atmos,
    args...,
) end

"""
    ImpenetrableDragLaw(drag) :: BCDef

Defines an impenetrable wall model for momentum. This implies:
  - no flow in the direction normal to the boundary, and
  - tangential components follow the given drag law

Drag law for momentum parallel to the boundary. The drag coefficient is
`C = fn(state, aux, t, normu_int_tan)`, where `normu_int_tan` is the internal speed
parallel to the boundary.
`_int` refers to the first interior node.
"""
struct ImpenetrableDragLaw{PV <: Union{Momentum, Energy}, D} <: BCDef{PV}
    drag::D
end

ImpenetrableDragLaw{PV}(drag::D) where {PV, D} =
    ImpenetrableDragLaw{PV, D}(drag)

ImpenetrableDragLaw(drag::D) where {D} = (
    ImpenetrableDragLaw{Momentum, D}(drag),
    ImpenetrableDragLaw{Energy, D}(drag),
)

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
    C = bc_momentum.drag(state⁻, aux⁻, t, normu_int⁻_tan)
    τn = C * normu_int⁻_tan * u_int⁻_tan
    # both sides involve projections of normals, so signs are consistent
    fluxᵀn.ρu += state⁻.ρ * τn
    fluxᵀn.energy.ρe += state⁻.ρu' * τn
end

function bc_val(
    bc::ImpenetrableDragLaw{Momentum},
    atmos::AtmosModel,
    nf::Union{NF1, NF∇},
    args,
)
    return bc_val(ImpenetrableFreeSlip{Momentum}(), atmos, nf, args)
end

function compute_τn(bc, args)
    @unpack state, state_int, n = args

    u1⁻ = state_int.ρu / state_int.ρ
    u_int_tan = u1 - dot(u1, n) .* SVector(n)
    normu_int_tan = norm(u_int_tan)
    # NOTE: difference from design docs since normal points outwards
    C = bc.drag(state, aux, t, normu_int_tan)
    τn = C * normu_int_tan * u_int_tan
    return τn
end

function bc_val(bc::ImpenetrableDragLaw{Momentum}, ::AtmosModel, ::NF2, args)
    @unpack state = args
    τn = compute_τn(bc, args)
    # both sides involve projections of normals, so signs are consistent
    return state.ρ + state.ρ * τn
end

function bc_val(::ImpenetrableDragLaw{Energy}, ::AtmosModel, ::NF2, args)
    @unpack state = args

    τn = compute_τn(bc, args)
    # both sides involve projections of normals, so signs are consistent
    return state.energy.ρe + state.ρu' * τn
end
