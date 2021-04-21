using ClimateMachine.Orientations

import ClimateMachine.Orientations: vertical_unit_vector

function orientation_gradient(
    model::AbstractFluidModel,
    ::Orientation,
    state_auxiliary,
    grid,
    direction,
)
    auxiliary_field_gradient!(
        model,
        state_auxiliary,
        ("orientation.∇Φ",),
        state_auxiliary,
        ("orientation.Φ",),
        grid,
        direction,
    )

    return nothing
end

function orientation_gradient(::AbstractFluidModel, ::NoOrientation, _...)
    return nothing
end

function orientation_nodal_init_aux!(
    ::NoOrientation,
    # domain::AbstractDomain,
    aux::Vars,
    geom::LocalGeometry,
)
    return nothing
end

function orientation_nodal_init_aux!(
    ::FlatOrientation,
    # domain::AbstractDomain,
    aux::Vars,
    geom::LocalGeometry,
)
    @inbounds aux.orientation.Φ = geom.coord[3]

    return nothing
end

function orientation_nodal_init_aux!(
    ::SphericalOrientation,
    # domain::AbstractDomain,
    aux::Vars,
    geom::LocalGeometry,
)
    norm_R = norm(geom.coord)
    # @inbounds aux.orientation.Φ = norm_R - domain.radius
    @inbounds aux.orientation.Φ = norm_R 

    return nothing
end

@inline vertical_unit_vector(::Orientation, aux) = aux.orientation.∇Φ
@inline vertical_unit_vector(::NoOrientation, aux) = @SVector [0, 0, 1]