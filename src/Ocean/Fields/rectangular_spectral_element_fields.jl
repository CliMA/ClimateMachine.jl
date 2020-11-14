using ClimateMachine.MPIStateArrays: MPIStateArray

using ..Domains: RectangularDomain

"""
    SpectralElementField(domain::RectangularDomain, state::MPIStateArray, variable_index::Int)

Returns a Cartesian `view` into `state.realdata[:, variable_index, :]`
that assumes `state.realdata` lives on `RectangularDomain`.

`SpectralElementField.elements` is a three-dimensional array of `RectangularElements`.
"""
function SpectralElementField(
    domain::RectangularDomain{FT},
    state::MPIStateArray,
    variable_index::Int,
) where {FT}

    # Unwind the data in solver
    data = view(state.realdata, :, variable_index, :)

    # Unwind volume geometry
    volume_geometry = domain.grid.vgeo

    Ne = domain.Ne
    Te = prod(domain.Ne) # total number of elements
    Te = prod(domain.Ne)
    Np = domain.Np

    grid = domain.grid

    # Transfer data to host
    x = zeros(FT, (Np + 1)^3, Te)
    y = zeros(FT, (Np + 1)^3, Te)
    z = zeros(FT, (Np + 1)^3, Te)

    x .= volume_geometry[:, grid.x1id, :]
    y .= volume_geometry[:, grid.x2id, :]
    z .= volume_geometry[:, grid.x3id, :]

    # Reshape x, y, z to (xnode, ynode, znode, element)
    x = reshape(x, Np + 1, Np + 1, Np + 1, Te)
    y = reshape(y, Np + 1, Np + 1, Np + 1, Te)
    z = reshape(z, Np + 1, Np + 1, Np + 1, Te)

    # Unwind the data in state.realdata
    data = view(state.realdata, :, variable_index, :)

    # Reshape data as coordinate arrays
    data = reshape(data, Np + 1, Np + 1, Np + 1, Te)

    # Construct a list of elements assuming Cartesian geometry
    element_list = [
        RectangularElement(
            view(data, :, :, :, i),
            view(x, :, :, :, i),
            view(y, :, :, :, i),
            view(z, :, :, :, i),
        ) for i in 1:Te
    ]

    function linear_coordinate(elem)
        Δx = elem.x[1] - domain.x[1]
        Δy = elem.y[1] - domain.y[1]
        Δz = elem.z[1] - domain.z[1]

        coordinate = (
            Δz / domain.L.z * domain.Ne.z * domain.Ne.y * domain.L.x +
            Δy / domain.L.y * domain.Ne.y * domain.L.x +
            Δx
        )

        return coordinate
    end

    sort!(element_list, by = linear_coordinate)

    # Reshape and permute dims to get an array where i, j, k correspond to x, y, z
    element_array = reshape(element_list, Ne.x, Ne.y, Ne.z)

    return SpectralElementField(element_array, domain)
end
