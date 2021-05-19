using ClimateMachine.Orientations

import ClimateMachine.Orientations: vertical_unit_vector

# function orientation_gradient(
#     model::AbstractFluidModel,
#     ::Orientation,
#     state_auxiliary,
#     grid,
#     direction,
# )
#     auxiliary_field_gradient!(
#         model,
#         state_auxiliary,
#         ("orientation.∇Φ",),
#         state_auxiliary,
#         ("orientation.Φ",),
#         grid,
#         direction,
#     )

#     return nothing
# end

# function orientation_gradient(::AbstractFluidModel, ::NoOrientation, _...)
#     return nothing
# end

# function orientation_nodal_init_aux!(
#     ::NoOrientation,
#     # domain::AbstractDomain,
#     aux::Vars,
#     geom::LocalGeometry,
# )
#     return nothing
# end

# function orientation_nodal_init_aux!(
#     ::FlatOrientation,
#     # domain::AbstractDomain,
#     aux::Vars,
#     geom::LocalGeometry,
# )
#     @inbounds aux.orientation.Φ = geom.coord[3]

#     return nothing
# end

# function orientation_nodal_init_aux!(
#     ::SphericalOrientation,
#     # domain::AbstractDomain,
#     aux::Vars,
#     geom::LocalGeometry,
# )
#     norm_R = norm(geom.coord)
#     # @inbounds aux.orientation.Φ = norm_R - domain.radius
#     @inbounds aux.orientation.Φ = norm_R 

#     return nothing
# end

function init_state_auxiliary!(
    ::ModelSetup,
    ::FlatOrientation,
    state_auxiliary,
    geom,
)
    FT = eltype(state_auxiliary)
    _grav = FT(grav(param_set))
    @inbounds r = geom.coord[3]
    state_auxiliary.x = geom.coord[1]
    state_auxiliary.y = geom.coord[2]
    state_auxiliary.z = geom.coord[3]
    state_auxiliary.Φ = _grav * r
    state_auxiliary.∇Φ = SVector{3, FT}(0, 0, _grav)
end

function init_state_auxiliary!(
    ::ModelSetup,
    ::SphericalOrientation,
    state_auxiliary,
    geom,
)
    FT = eltype(state_auxiliary)
    _grav = FT(grav(param_set))
    r = norm(geom.coord)
    state_auxiliary.x = geom.coord[1]
    state_auxiliary.y = geom.coord[2]
    state_auxiliary.z = geom.coord[3]
    state_auxiliary.Φ = _grav * r
    state_auxiliary.∇Φ = _grav * geom.coord / r
end

@inline vertical_unit_vector(::Orientation, aux) = aux.∇Φ / grav(param_set)
@inline vertical_unit_vector(::NoOrientation, aux) = @SVector [0, 0, 1]