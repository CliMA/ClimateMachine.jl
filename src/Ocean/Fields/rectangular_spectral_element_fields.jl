using ..Domains: RectangularDomain

""" Like a linear index... """
function linear_coordinate(elem, domain, cpu_x, cpu_y, cpu_z)

    corner_x = cpu_x[1, elem.index]
    corner_y = cpu_y[1, elem.index]
    corner_z = cpu_z[1, elem.index]

    Δx = corner_x - domain.x[1]
    Δy = corner_y - domain.y[1]
    Δz = corner_z - domain.z[1]

    coordinate = (
        Δz / domain.L.z * domain.Ne.z * domain.Ne.y * domain.L.x +
        Δy / domain.L.y * domain.Ne.y * domain.L.x +
        Δx
    )

    return coordinate
end

"""
    SpectralElementField(domain::RectangularDomain, grid, realdata::AbstractArray)

Returns a `SpectralElementField` whose `elements` provide a Cartesian `view`
into `realdata`, assuming that `realdata` lives on `domain::RectangularDomain`.
"""
function SpectralElementField(domain::RectangularDomain{FT}, grid, realdata::AbstractArray) where FT

    # Build element list
    Te = prod(domain.Ne) # total number of elements
    
    element_list = [RectangularElement(domain, grid, realdata, i) for i in 1:Te]
    
    # Transfer coordinate data to host for element sorting
    volume_geometry = grid.vgeo

    x = Array(volume_geometry[:, grid.x1id, :])
    y = Array(volume_geometry[:, grid.x2id, :])
    z = Array(volume_geometry[:, grid.x3id, :])

    # Sort elements by linear coordinate for x, y, z structuring
    sort!(element_list, by = elem -> linear_coordinate(elem, domain, x, y, z))

    # Reshape and permute dims to obtain array where i, j, k correspond to x, y, z
    Ne = domain.Ne
    element_array = reshape(element_list, Ne.x, Ne.y, Ne.z)

    return SpectralElementField(element_array, domain, grid, realdata)
end
