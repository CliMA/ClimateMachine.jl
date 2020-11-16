using MPI

using ClimateMachine: Settings

using ClimateMachine.Mesh.Grids: polynomialorder

using ClimateMachine.Mesh.Topologies: StackedBrickTopology

import ClimateMachine.Mesh.Grids: DiscontinuousSpectralElementGrid

#####
##### RectangularDomain
#####

struct RectangularDomain{FT} <: AbstractDomain
    Np::Int
    Ne::NamedTuple{(:x, :y, :z), NTuple{3, Int}}
    L::NamedTuple{(:x, :y, :z), NTuple{3, FT}}
    x::NTuple{2, FT}
    y::NTuple{2, FT}
    z::NTuple{2, FT}
    periodicity::NamedTuple{(:x, :y, :z), NTuple{3, Bool}}
end

Base.eltype(::RectangularDomain{FT}) where {FT} = FT

function Base.show(io::IO, domain::RectangularDomain{FT}) where {FT}
    Np = domain.Np
    Ne = domain.Ne
    L = domain.L

    first = "RectangularDomain{$FT}\n"
    second = "    Np = $Np, Ne = $Ne\n"
    third = "    L = $L\n"

    return print(io, first, second, third)
end
    
name_it(Ne::NamedTuple{(:x, :y, :z)}) = Ne
name_it(Ne) = (x = Ne[1], y = Ne[2], z = Ne[3])

"""
    RectangularDomain(FT=Float64;
                      Ne,
                      Np,
                      x = (-1, 1),
                      y = (-1, 1),
                      z = (-1, 1),
                      periodicity = (true, true, false))

Returns a `RectangularDomain` representing the product of `x, y, z` intervals,
specified by 2-tuples.

The `RectangularDomain` is meshed with a simple `DiscontinuousSpectralElementGrid`
with an isotropic polynomial order `Np` and a 3-tuple of `Ne`lements
giving the number of elements in `x, y, z`.

Additional arguments are:

- `periodicity`: a 3-tuple that indicates periodic dimensions with `true`, 

- `boundary`: specifies the boundary condition on each boundary with a
              boundary condition `tag`

- `array_type`: either `Array` for CPU computations or `CuArray` for
                GPU computations

- `mpicomm`: communicator for sending data across nodes in a distributed memory
             configuration using the Message Passing Interface (MPI).
             See https://pages.tacc.utexas.edu/~eijkhout/pcse/html/mpi-comm.html

Example
=======

```julia
julia> using ClimateMachine; ClimateMachine.init()

julia> using ClimateMachine.Ocean.RectangularDomains: RectangularDomain

julia> domain = RectangularDomain(Ne=(7, 8, 9), Np=4, x=(0, 1), y=(0, 1), z=(0, 1))
RectangularDomain{Float64}:
    Np = 4, Ne = (x = 7, y = 8, z = 9)
    L = (x = 1.00e+00, y = 1.00e+00, z = 1.00e+00)
    x = (0.00e+00, 1.00e+00), y = (0.00e+00, 1.00e+00), z = (1.00e+00, 0.00e+00)
```
"""
function RectangularDomain(
    FT = Float64;
    Ne,
    Np,
    x::Tuple{<:Number, <:Number},
    y::Tuple{<:Number, <:Number},
    z::Tuple{<:Number, <:Number},
    periodicity = (true, true, false),
)

    Ne = name_it(Ne)
    periodicity = name_it(periodicity)

    west, east = FT.(x)
    south, north = FT.(y)
    bottom, top = FT.(z)

    east > west || error("Domain x-limits must be increasing!")
    north > south || error("Domain y-limits must be increasing!")
    top > bottom || error("Domain z-limits must be increasing!")

    L = (x = east - west, y = north - south, z = top - bottom)

    return RectangularDomain(
        Np,
        Ne,
        L,
        (west, east),
        (south, north),
        (bottom, top),
        periodicity,
    )
end

array_type(domain::RectangularDomain) = Settings.array_type
eltype(::RectangularDomain{FT}) where {FT} = FT

function DiscontinuousSpectralElementGrid(
    domain::RectangularDomain{FT};
    boundary_tags = ((0, 0), (0, 0), (1, 2)),
    array_type = Settings.array_type,
    mpicomm = MPI.COMM_WORLD,
) where {FT}

    west, east = domain.x
    south, north = domain.y
    bottom, top = domain.z

    element_coordinates = (
        range(west, east, length = domain.Ne.x + 1),
        range(south, north, length = domain.Ne.y + 1),
        range(bottom, top, length = domain.Ne.z + 1),
    )

    topology = StackedBrickTopology(
        mpicomm,
        element_coordinates;
        periodicity = tuple(domain.periodicity...),
        boundary = boundary_tags,
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = domain.Np,
    )

    return grid
end
