"""
    calc_boundary_state!(::NumericalFluxFirstOrder, ::Impenetrable{FreeSlip}, ::ModelSetup)
apply free slip boundary condition for velocity
sets reflective ghost point
"""
@inline function calc_boundary_state!(
    ::NumericalFluxFirstOrder,
    ::Impenetrable{FreeSlip},
    ::ModelSetup,
    ::Union{AbstractDiffusion, Nothing},
    state⁺,
    aux⁺,
    n⁻,
    state⁻,
    aux⁻,
    t,
    _...,
)
    state⁺.ρ = state⁻.ρ

    ρu⁻ = state⁻.ρu
    state⁺.ρu = ρu⁻ - 2 * n⁻ ⋅ ρu⁻ .* SVector(n⁻)

    return nothing
end

"""
    calc_boundary_state!(::NumericalFluxFirstOrder, ::Impenetrable{NoSlip}, ::ModelSetup)
apply no slip boundary condition for velocity
sets reflective ghost point
"""
@inline function calc_boundary_state!(
    ::NumericalFluxFirstOrder,
    ::Impenetrable{NoSlip},
    ::ModelSetup,
    ::Union{AbstractDiffusion, Nothing},
    state⁺,
    aux⁺,
    n⁻,
    state⁻,
    aux⁻,
    t,
    _...,
)
    state⁺.ρ = state⁻.ρ
    state⁺.ρu = -state⁻.ρu

    return nothing
end

"""
    calc_boundary_state!(::NumericalFluxFirstOrder, ::Insulating, ::HBModel)

apply insulating boundary condition for temperature
sets transmissive ghost point
"""
@inline function calc_boundary_state!(
    ::NumericalFluxFirstOrder,
    ::Insulating,
    ::ModelSetup,
    ::Union{AbstractDiffusion, Nothing},
    state⁺,
    aux⁺,
    n⁻,
    state⁻,
    aux⁻,
    t,
    _...,
)
    state⁺.ρθ = state⁻.ρθ

    nothing
end

# """
#     calc_boundary_state!(::NumericalFluxFirstOrder, ::Penetrable{FreeSlip}, ::ModelSetup)
# no mass boundary condition for penetrable
# """
# @inline calc_boundary_state!(
#     ::NumericalFluxFirstOrder,
#     ::Penetrable{FreeSlip},
#     ::ModelSetup,
#     ::ConstantViscosity,
#     _...,
# ) = nothing


# """
#     calc_boundary_state!(::NumericalFluxFirstOrder, ::Impenetrable{MomentumFlux}, ::HBModel)
# apply kinematic stress boundary condition for velocity
# applies free slip conditions for first-order and gradient fluxes
# """
# @inline function calc_boundary_state!(
#     nf::NumericalFluxFirstOrder,
#     ::Impenetrable{<:MomentumFlux},
#     model::ModelSetup,
#     diff::AbstractDiffusion,
#     args...,
# )
#     return calc_boundary_state!(
#         nf,
#         Impenetrable(FreeSlip()),
#         model,
#         diff,
#         args...,
#     )
# end

# """
#     calc_boundary_state!(::NumericalFluxFirstOrder, ::Penetrable{MomentumFlux}, ::HBModel)
# apply kinematic stress boundary condition for velocity
# applies free slip conditions for first-order and gradient fluxes
# """
# @inline function calc_boundary_state!(
#     nf::NumericalFluxFirstOrder,
#     ::Penetrable{<:MomentumFlux},
#     model::ModelSetup,
#     diff::AbstractDiffusion,
#     args...,
# )
#     return calc_boundary_state!(
#         nf,
#         Penetrable(FreeSlip()),
#         model,
#         diff,
#         args...,
#     )
# end

#     return nothing
# end

# """
#     calc_boundary_state!(::NumericalFluxFirstOrder, ::TemperatureFlux, ::HBModel)

# apply temperature flux boundary condition for velocity
# applies insulating conditions for first-order and gradient fluxes
# """
# @inline function calc_boundary_state!(
#     nf::NumericalFluxFirstOrder,
#     ::TemperatureFlux,
#     model::ModelSetup,
#     args...,
# )
#     return calc_boundary_state!(nf, Insulating(), model, args...)
# end