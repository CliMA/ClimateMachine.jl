import ClimateMachine.Mesh.Grids: DiscontinuousSpectralElementGrid
using ClimateMachine.Mesh.Topologies
export DiscontinuousSpectralElementGrid

"""
function DiscontinuousSpectralElementGrid(Ω::ProductDomain; elements = nothing, polynomialorder = nothing)
# Description 
Computes a DiscontinuousSpectralElementGrid as specified by a product domain
# Arguments
-`Ω`: A product domain object
# Keyword Arguments TODO: Add brickrange and topology as keyword arguments
-`elements`: A tuple of integers ordered by (Nx, Ny, Nz) for number of elements
-`polynomialorder`: A tupe of integers ordered by (npx, npy, npz) for polynomial order
-`FT`: floattype, assumed Float64 unless otherwise specified
-`mpicomm`: default = MPI.COMM_WORLD
-`array`: default = Array, but should generally be ArrayType
# Return 
A DiscontinuousSpectralElementGrid object
"""
function DiscontinuousSpectralElementGrid(Ω; periodicity = (false, false, false), elements = nothing, polynomialorder = nothing, boundary = nothing, FT=Float64, mpicomm=MPI.COMM_WORLD, array = Array, dimension = 3)
    tuple_ranges = []
    for i in 1:dimension
        push!(tuple_ranges,range(FT(Ω[i][1]); length = elements[i] + 1,
            stop = FT(Ω[i][2])))
    end

    brickrange = Tuple(tuple_ranges)
    if boundary==nothing
        boundary = (ntuple(j -> (1, 2), dimension - 1)...,(3, 4),)
    end

    topl = StackedBrickTopology(
                            mpicomm,
                            brickrange;
                            periodicity = periodicity,
                            boundary = boundary
    )

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = array,
        polynomialorder = polynomialorder,
    )
    return grid
end