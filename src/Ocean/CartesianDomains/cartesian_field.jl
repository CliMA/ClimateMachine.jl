using ClimateMachine.MPIStateArrays: MPIStateArray

using ..Fields: AbstractField

import ..Fields: field

#####
##### CartesianField
#####

struct CartesianField{E, D} <: AbstractField
    elements::E
    domain::D
end

function Base.collect(field::CartesianField)
    elements = [
        collect(field.elements[i, j, k])
        for
        i in 1:(field.domain.Ne.x),
        j in 1:(field.domain.Ne.y), k in 1:(field.domain.Ne.z)
    ]
    return CartesianField(elements, field)
end

Base.maximum(f, field::CartesianField) =
    maximum([maximum(f, el) for el in field.elements])
Base.minimum(f, field::CartesianField) =
    minimum([minimum(f, el) for el in field.elements])

Base.maximum(field::CartesianField) =
    maximum([maximum(el) for el in field.elements])
Base.minimum(field::CartesianField) =
    minimum([minimum(el) for el in field.elements])

Base.show(io::IO, field::CartesianField{FT}) where {FT} =
    print(io, "CartesianField{$FT}")

Base.@propagate_inbounds Base.getindex(field::CartesianField, i, j, k) =
    field.elements[i, j, k]

Base.size(field::CartesianField) = size(field.elements)

"""
    CartesianField(domain::CartesianDomain, state::MPIStateArray, variable_index::Int)

Returns an abstracted Cartesian `view` into `state.realdata[:, variable_index, :]`
that assumes `state.realdata` lives on `CartesianDomain`.

`CartesianField.elements` is a three-dimensional array of `RectangularElements`.
"""
function CartesianField(
    domain::CartesianDomain,
    state::MPIStateArray,
    variable_index::Int,
)

    # Unwind the data in solver
    data = view(state.realdata, :, variable_index, :)

    # Unwind volume geometry
    volume_geometry = domain.grid.vgeo

    Ne = domain.Ne
    Te = prod(domain.Ne)
    Np = domain.Np

    grid = domain.grid

    # Extract coordinate arrays with size (xnode, ynode, znode, element)
    x = reshape(volume_geometry[:, grid.x1id, :], Np + 1, Np + 1, Np + 1, Te)
    y = reshape(volume_geometry[:, grid.x2id, :], Np + 1, Np + 1, Np + 1, Te)
    z = reshape(volume_geometry[:, grid.x3id, :], Np + 1, Np + 1, Np + 1, Te)

    # Reshape data as coordinate arrays
    data = reshape(data, Np + 1, Np + 1, Np + 1, Te)

    # Construct a list of elements assuming Cartesian geometry
    element_list = [
        RectangularElement(
            view(data, :, :, :, i),
            view(x, :, 1, 1, i),
            view(y, 1, :, 1, i),
            view(z, 1, 1, :, i),
        ) for i in 1:Te
    ]

    function linear_coordinate(elem)
        Δx = elem.x[1] - domain.x[1]
        Δy = elem.y[1] - domain.y[1]
        Δz = elem.z[1] - domain.z[1]

        coordinate = (
            Δz / domain.Lz * domain.Ne.z * domain.Ne.y * domain.Lx +
            Δy / domain.Ly * domain.Ne.y * domain.Lx +
            Δx
        )

        return coordinate
    end

    sort!(element_list, by = linear_coordinate)

    # Reshape and permute dims to get an array where i, j, k correspond to x, y, z
    element_array = reshape(element_list, Ne.x, Ne.y, Ne.z)

    return CartesianField(element_array, domain)
end

field(domain::CartesianDomain, state, index) =
    CartesianField(domain, state, index)
