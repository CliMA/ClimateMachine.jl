using ClimateMachine.Orientations

import ClimateMachine.Orientations: vertical_unit_vector

function init_state_auxiliary!(
    model::ModelSetup,
    ::FlatOrientation,
    state_auxiliary,
    geom,
)
    FT = eltype(state_auxiliary)
    _grav = model.physics.parameters.g
    @inbounds r = geom.coord[3]
    state_auxiliary.x = geom.coord[1]
    state_auxiliary.y = geom.coord[2]
    state_auxiliary.z = geom.coord[3]
    state_auxiliary.Φ = _grav * r
    state_auxiliary.∇Φ = SVector{3, FT}(0, 0, _grav)
end

function init_state_auxiliary!(
    model::ModelSetup,
    ::SphericalOrientation,
    state_auxiliary,
    geom,
)
    _grav = model.physics.parameters.g
    r = norm(geom.coord)
    state_auxiliary.x = geom.coord[1]
    state_auxiliary.y = geom.coord[2]
    state_auxiliary.z = geom.coord[3]
    state_auxiliary.Φ = _grav * r
    state_auxiliary.∇Φ = _grav * geom.coord / r
end

@inline vertical_unit_vector(::Orientation, aux) = aux.∇Φ / norm(aux.∇Φ)
@inline vertical_unit_vector(::NoOrientation, aux) = @SVector [0, 0, 1]