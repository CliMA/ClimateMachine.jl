import ClimateMachine.Mesh.Grids: SpectralElementGrid

function coordinates(grid::SpectralElementGrid)
    x = view(grid.vgeo, :, grid.x1id, :)   # x-direction	
    y = view(grid.vgeo, :, grid.x2id, :)   # y-direction	
    z = view(grid.vgeo, :, grid.x3id, :)   # z-direction
    return x, y, z
end

# some convenience functions
function convention(
    a::NamedTuple{(:vertical, :horizontal), T},
    ::Val{3},
) where {T}
    return (a.horizontal, a.horizontal, a.vertical)
end

function convention(a::Number, ::Val{3})
    return (a, a, a)
end

function convention(
    a::NamedTuple{(:vertical, :horizontal), T},
    ::Val{2},
) where {T}
    return (a.horizontal, a.vertical)
end

function convention(a::Number, ::Val{2})
    return (a, a)
end

function convention(a::Tuple, b)
    return a
end

# brick range brickbuilder
function uniform_brick_builder(domain, elements; FT = Float64)
    dimension = ndims(domain)

    tuple_ranges = []
    for i in 1:dimension
        push!(
            tuple_ranges,
            range(
                FT(domain[i].min);
                length = elements[i] + 1,
                stop = FT(domain[i].max),
            ),
        )
    end

    brickrange = Tuple(tuple_ranges)
    return brickrange
end

# Grid Constructor
"""
function SpectralElementGrid(domain::ProductDomain; elements = nothing, polynomialorder = nothing)
# Description 
Computes a SpectralElementGrid as specified by a product domain
# Arguments
-`domain`: A product domain object
# Keyword Arguments 
-`elements`: A tuple of integers ordered by (Nx, Ny, Nz) for number of elements
-`polynomialorder`: A tupe of integers ordered by (npx, npy, npz) for polynomial order
-`FT`: floattype, assumed Float64 unless otherwise specified
-`topology`: default = StackedBrickTopology
-`mpicomm`: default = MPI.COMM_WORLD
-`array`: default = ClimateMachine.array_type()
-`brickbuilder`: default = uniform_brick_builder, 
  brickrange=uniform_brick_builder(domain, elements)
# Return 
A SpectralElementGrid object
"""
function SpectralElementGrid(
    domain::ProductDomain;
    elements = nothing,
    polynomialorder = nothing,
    FT = Float64,
    mpicomm = MPI.COMM_WORLD,
    array = ClimateMachine.array_type(),
    topology = StackedBrickTopology,
    brick_builder = uniform_brick_builder,
)

    if elements == nothing
        error_message = "Please specify the number of elements as a tuple whose size is commensurate with the domain,"
        error_message *= " e.g., a 3 dimensional domain would need a specification like elements = (10,10,10)."
        error_message *= " or elements = (vertical = 8, horizontal = 5)"

        @error(error_message)
        return nothing
    end

    if polynomialorder == nothing
        error_message = "Please specify the polynomial order as a tuple whose size is commensurate with the domain,"
        error_message *= "e.g., a 3 dimensional domain would need a specification like polynomialorder = (3,3,3)."
        error_message *= " or polynomialorder = (vertical = 8, horizontal = 5)"

        @error(error_message)
        return nothing
    end

    dimension = ndims(domain)

    if (dimension < 2) || (dimension > 3)
        error_message = "SpectralElementGrid only works with dimensions 2 or 3. "
        error_message *= "The current dimension is " * string(ndims(domain))

        println("The domain is ", domain)
        @error(error_message)
        return nothing
    end

    elements = convention(elements, Val(dimension))
    if ndims(domain) != length(elements)
        @error("Incorrectly specified elements for the dimension of the domain")
        return nothing
    end

    polynomialorder = convention(polynomialorder, Val(dimension))
    if ndims(domain) != length(polynomialorder)
        @error("Incorrectly specified polynomialorders for the dimension of the domain")
        return nothing
    end

    brickrange = brick_builder(domain, elements, FT = FT)

    if dimension == 2
        boundary = ((1, 2), (3, 4))
    else
        boundary = ((1, 2), (3, 4), (5, 6))
    end

    periodicity = periodicityof(domain)
    connectivity = dimension == 2 ? :face : :full

    topl = topology(
        mpicomm,
        brickrange;
        periodicity = periodicity,
        boundary = boundary,
        connectivity = connectivity,
    )

    grid = SpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = array,
        polynomialorder = polynomialorder,
    )

    return grid
end

abstract type AbstractDiscretizedDomain end

struct DiscretizedDomain{𝒜, ℬ, 𝒞} <: AbstractDiscretizedDomain
    domain::𝒜
    resolution::ℬ
    numerical::𝒞
end

function DiscretizedDomain(
    domain::ProductDomain;
    elements = nothing,
    polynomial_order = nothing,
    overintegration_order = nothing,
    FT = Float64,
    mpicomm = MPI.COMM_WORLD,
    array = ClimateMachine.array_type(),
    topology = StackedBrickTopology,
    brick_builder = uniform_brick_builder,
)

    grid = SpectralElementGrid(
        domain,
        elements = elements,
        polynomialorder = polynomial_order .+ overintegration_order,
        FT = FT,
        mpicomm = mpicomm,
        array = array,
        topology = topology,
        brick_builder = brick_builder,
    )
    return DiscretizedDomain(
        domain,
        (; elements, polynomial_order, overintegration_order),
        grid,
    )
end
