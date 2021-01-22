using CLIMAParameters
using CLIMAParameters.Planet: planet_radius
using ..Diagnostics
using ..Mesh.Interpolation

"""
    DiagnosticsConfiguration

Container for all the `DiagnosticsGroup`s to be used for a simulation.
"""
mutable struct DiagnosticsConfiguration
    groups::Array{DiagnosticsGroup, 1}

    DiagnosticsConfiguration(
        groups::Array{DG, 1},
    ) where {DG <: DiagnosticsGroup} = new(groups)
end

function InterpolationConfiguration(
    ::StackedBrickTopology,
    driver_config,
    boundaries,
    axes,
)
    grid = driver_config.grid
    return InterpolationBrick(grid, boundaries, axes[1], axes[2], axes[3])
end

function InterpolationConfiguration(
    ::StackedCubedSphereTopology,
    driver_config,
    boundaries,
    axes;
    nr_toler = nothing,
)
    FT = eltype(driver_config.grid)
    param_set = driver_config.bl.param_set
    grid = driver_config.grid
    info = driver_config.config_info

    _planet_radius::FT = planet_radius(param_set)
    vert_range = grid1d(
        _planet_radius,
        FT(_planet_radius + info.domain_height),
        nelem = info.nelem_vert,
    )

    return InterpolationCubedSphere(
        grid,
        collect(vert_range),
        info.nelem_horz,
        axes[1],
        axes[2],
        axes[3];
        nr_toler = nr_toler,
    )
end

"""
    InterpolationConfiguration(
        driver_config::DriverConfiguration,
        boundaries::Array,
        resolution = nothing;
        axes = nothing;
    )

Creates an `InterpolationTopology` (either an `InterpolationBrick` or an
`InterpolationCubedSphere`) to be used with a `DiagnosticsGroup`. Either
`resolution` is specified, in which case the axes are set up with
equi-distant points, or the `axes` may be specified directly (in
lat/lon/lvl or x/y/z order).
"""
function InterpolationConfiguration(
    driver_config::DriverConfiguration,
    boundaries::Array,
    resolution = nothing;
    axes = nothing,
)
    @assert isnothing(resolution) || isnothing(axes)
    if isnothing(axes)
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
    end

    return InterpolationConfiguration(
        driver_config.grid.topology,
        driver_config,
        boundaries,
        axes,
    )
end
