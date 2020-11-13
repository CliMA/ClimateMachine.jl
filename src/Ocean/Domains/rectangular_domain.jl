using MPI

using ClimateMachine.Mesh.Grids:
    DiscontinuousSpectralElementGrid, polynomialorder

using ClimateMachine.Mesh.Topologies: StackedBrickTopology

#####
##### RectangularDomain
#####

struct RectangularDomain{FT, G} <: AbstractDomain
    grid::G
    Np::Int
    Ne::NamedTuple{(:x, :y, :z), NTuple{3, Int}}
    L::NamedTuple{(:x, :y, :z), NTuple{3, FT}}
    x::NTuple{2, FT}
    y::NTuple{2, FT}
    z::NTuple{2, FT}
end

Base.eltype(::RectangularDomain{FT}) where {FT} = FT

Base.show(io::IO, domain::RectangularDomain{FT, G}) where {FT, G} = print(
    io,
    "RectangularDomain{$FT, $(G.name.wrapper)}:",
    '\n',
    "    Np = ",
    domain.Np,
    ", Ne = ",
    domain.Ne,
    '\n',
    @sprintf(
        "    L = (x = %.2e, y = %.2e, z = %.2e)",
        domain.L.x,
        domain.L.y,
        domain.L.z
    ),
    '\n',
    @sprintf(
        "    x = (%.2e, %.2e), y = (%.2e, %.2e), z = (%.2e, %.2e)",
        domain.x[1],
        domain.x[2],
        domain.y[1],
        domain.y[2],
        domain.z[1],
        domain.z[2]
    )
)

name_it(Ne::NamedTuple{(:x, :y, :z)}) = Ne
name_it(Ne) = (x = Ne[1], y = Ne[2], z = Ne[3])

"""
    RectangularDomain(grid::DiscontinuousSpectralElementGrid, Ne)

Inverts the volume geometry information in `grid.vgeo` to construct
a `RectangularDomain`.
"""
function RectangularDomain(grid::DiscontinuousSpectralElementGrid, Ne)
    Ne = name_it(Ne)

    # Unwind volume geometry
    volume_geometry = grid.vgeo

    # Check number of elements
    prod(Ne) === size(volume_geometry, 3) ||
    error("prod(Ne) must match the total number of grid elements.")

    Np = polynomialorder(grid)

    x = view(volume_geometry, :, grid.x1id, :)
    y = view(volume_geometry, :, grid.x2id, :)
    z = view(volume_geometry, :, grid.x3id, :)

    xlims = (minimum(x), maximum(x))
    ylims = (minimum(y), maximum(y))
    zlims = (minimum(z), maximum(z))

    L = (
        x = xlims[2] - xlims[1],
        y = ylims[2] - ylims[1],
        z = zlims[2] - zlims[1],
    )

    return RectangularDomain(grid, Np, Ne, L, xlims, ylims, zlims)
end

"""
    RectangularDomain(FT=Float64;
                      elements,
                      polynomialorder,
                      x = (-1, 1),
                      y = (-1, 1),
                      z = (-1, 1),
                      periodicity = (true, true, false),
                      boundary = ((0, 0), (0, 0), (1, 2)),
                      array_type = Settings.array_type,
                      message_communicator = MPI.COMM_WORLD)

Returns a `RectangularDomain` representing the product of `x, y, z` intervals,
specified by 2-tuples.

The `RectangularDomain` is meshed with a simple `DiscontinuousSpectralElementGrid`
with an isotropic `polynomialorder` and a 3-tuple of `elements`
giving the number of elements in `x, y, z`.

Additional arguments are:

- `periodicity`: a 3-tuple that indicates periodic dimensions with `true`, 

- `boundary`: specifies the boundary condition on each boundary with a
              boundary condition `tag`

- `array_type`: either `Array` for CPU computations or `CuArray` for
                GPU computations

- `message_communicator`: the world communicator for message passing between
                          nodes in a distributed memory configuration.

Example
=======

```jldoctest
julia> using ClimateMachine; ClimateMachine.init()

julia> using ClimateMachine.Ocean.RectangularDomains: RectangularDomain

julia> domain = RectangularDomain(elements=(7, 8, 9), polynomialorder=4, x=(0, 1), y=(0, 1), z=(0, 1))
RectangularDomain{Float64, ClimateMachine.Mesh.Grids.DiscontinuousSpectralElementGrid}:
    Np = 4, Ne = (x = 7, y = 8, z = 9)
    L = (x = 1.00e+00, y = 1.00e+00, z = 1.00e+00)
    x = (0.00e+00, 1.00e+00), y = (0.00e+00, 1.00e+00), z = (1.00e+00, 0.00e+00)
```
"""
function RectangularDomain(
    FT = Float64;
    elements,
    polynomialorder,
    x::Tuple{<:Number, <:Number},
    y::Tuple{<:Number, <:Number},
    z::Tuple{<:Number, <:Number},
    periodicity = (true, true, false),
    boundary = ((0, 0), (0, 0), (1, 2)),
    array_type = ClimateMachine.Settings.array_type,
    message_communicator = MPI.COMM_WORLD,
)

    Ne = name_it(elements)

    west, east = FT.(x)
    south, north = FT.(y)
    bottom, top = FT.(z)

    east > west || error("Domain x-limits must be increasing!")
    north > south || error("Domain y-limits must be increasing!")
    top > bottom || error("Domain z-limits must be increasing!")

    L = (x = east - west, y = north - south, z = top - bottom)

    element_coordinates = (
        range(west, east, length = Ne.x + 1),
        range(south, north, length = Ne.y + 1),
        range(bottom, top, length = Ne.z + 1),
    )

    topology = StackedBrickTopology(
        message_communicator,
        element_coordinates;
        periodicity = periodicity,
        boundary = boundary,
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = polynomialorder,
    )

    return RectangularDomain{FT, typeof(grid)}(
        grid,
        polynomialorder,
        Ne,
        L,
        (west, east),
        (south, north),
        (bottom, top),
    )
end

array_type(domain::RectangularDomain) = Array #array_type(domain.grid)
eltype(::RectangularDomain{FT}) where {FT} = FT
communicator(args...) = MPI.COMM_WORLD
