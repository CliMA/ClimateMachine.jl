using ..Domains: RectangularDomain

""" Like a linear index... """
function linear_coordinate(elem, domain)
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

"""
    SpectralElementField(domain::RectangularDomain, realdata::AbstractArray)

Returns a `SpectralElementField` whose `elements` provide a Cartesian `view`
into `realdata`, assuming that `realdata` lives on `domain::RectangularDomain`.
"""
function SpectralElementField(domain::RectangularDomain{FT}, realdata::AbstractArray) where FT

    # Unwind volume geometry
    volume_geometry = domain.grid.vgeo

    Ne = domain.Ne
    Te = prod(domain.Ne) # total number of elements
    Np = domain.Np

    grid = domain.grid

    # Transfer coordinate data to host
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

    # Reshape realdata as coordinate arrays
    reshaped_realdata = reshape(realdata, Np + 1, Np + 1, Np + 1, Te)

    element_list = [
        RectangularElement(
            view(reshaped_realdata, :, :, :, i),
            view(x, :, :, :, i),
            view(y, :, :, :, i),
            view(z, :, :, :, i),
        ) for i in 1:Te
    ]
    
    sort!(element_list, by = elem -> linear_coordinate(elem, domain))

    # Reshape and permute dims to get an array where i, j, k correspond to x, y, z
    element_array = reshape(element_list, Ne.x, Ne.y, Ne.z)

    return SpectralElementField(realdata, domain, element_array)
end
