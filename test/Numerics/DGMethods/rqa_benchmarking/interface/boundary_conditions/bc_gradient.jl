"""
    calc_boundary_state!(::NumericalFluxGradient, ::Impenetrable{FreeSlip}, ::ModelSetup)
apply free slip boundary condition for velocity
sets non-reflective ghost point
"""
function calc_boundary_state!(
    ::NumericalFluxGradient,
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
    state⁺.ρu = ρu⁻ - n⁻ ⋅ ρu⁻ .* SVector(n⁻)

    return nothing
end

"""
    calc_boundary_state!(::NumericalFluxGradient, ::Impenetrable{NoSlip}, ::ModelSetup)
apply no slip boundary condition for velocity
set numerical flux to zero for U
"""
@inline function calc_boundary_state!(
    ::NumericalFluxGradient,
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
    FT = eltype(state⁺)
    state⁺.ρu = @SVector zeros(FT, 3)

    return nothing
end

"""
    calc_boundary_state!(::NumericalFluxGradient, ::Insulating, ::HBModel)

apply insulating boundary condition for temperature
sets transmissive ghost point
"""
@inline function calc_boundary_state!(
    ::NumericalFluxGradient,
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

    return nothing
end

# """
#     calc_boundary_state!(::NumericalFluxGradient, ::Penetrable{FreeSlip}, ::ModelSetup)
# no mass boundary condition for penetrable
# """
# @inline calc_boundary_state!(
#     ::NumericalFluxGradient,
#     ::Penetrable{FreeSlip},
#     ::ModelSetup,
#     ::ConstantViscosity,
#     _...,
# ) = nothing
# """
#     calc_boundary_state!(::NumericalFluxGradient, ::Impenetrable{MomentumFlux}, ::HBModel)
# apply kinematic stress boundary condition for velocity
# applies free slip conditions for first-order and gradient fluxes
# """
# @inline function calc_boundary_state!(
#     nf::NumericalFluxGradient,
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
#     calc_boundary_state!(::NumericalFluxGradient, ::Penetrable{MomentumFlux}, ::HBModel)
# apply kinematic stress boundary condition for velocity
# applies free slip conditions for first-order and gradient fluxes
# """
# @inline function calc_boundary_state!(
#     nf::NumericalFluxGradient,
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

# """
#     calc_boundary_state!(::NumericalFluxGradient, ::TemperatureFlux, ::HBModel)

# apply temperature flux boundary condition for velocity
# applies insulating conditions for first-order and gradient fluxes
# """
# @inline function calc_boundary_state!(
#     nf::NumericalFluxGradient,
#     ::TemperatureFlux,
#     model::ModelSetup,
#     args...,
# )
#     return calc_boundary_state!(nf, Insulating(), model, args...)
# end