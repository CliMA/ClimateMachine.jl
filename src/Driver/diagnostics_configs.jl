using ..Diagnostics
using ..Mesh.Interpolation

mutable struct DiagnosticsConfiguration
    groups::Array{DiagnosticsGroup, 1}

    DiagnosticsConfiguration(groups::Array{DiagnosticsGroup, 1}) = new(groups)
end

function InterpolationConfiguration(
    driver_config::DriverConfiguration,
    boundaries::Array,
    resolution::Tuple,
)
    grid = driver_config.grid
    if isa(grid.topology, StackedBrickTopology)

        axes = (
            collect(range(
                boundaries[1, 1],
                boundaries[2, 1],
                step = resolution[1],
            )),
            collect(range(
                boundaries[1, 2],
                boundaries[2, 2],
                step = resolution[2],
            )),
            collect(range(
                boundaries[1, 3],
                boundaries[2, 3],
                step = resolution[3],
            )),
        )
        return InterpolationBrick(grid, boundaries, axes[1], axes[2], axes[3])

    elseif isa(grid.topology, StackedCubedSphereTopology)

        FT = eltype(grid)
        info = driver_config.config_info
        vert_range = grid1d(
            FT(planet_radius),
            FT(planet_radius + info.domain_height),
            nelem = info.nelem_vert,
        )

        axes = (
            collect(range(
                boundaries[1, 1],
                boundaries[2, 1],
                step = resolution[1],
            )),
            collect(range(
                boundaries[1, 2],
                boundaries[2, 2],
                step = resolution[2],
            )),
            collect(range(
                boundaries[1, 3],
                boundaries[2, 3],
                step = resolution[3],
            )),
        )
        return InterpolationCubedSphere(
            grid,
            collect(vert_range),
            info.nelem_horz,
            axes[1],
            axes[2],
            axes[3],
        )
    else
        @error "Cannot set up interpolation for this topology."
    end
end
