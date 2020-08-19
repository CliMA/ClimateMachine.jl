"""
function cell_average(Q; M = nothing)

# Description
Compute the cell-average of Q given the mass matrix M.
Assumes that Q and M are the same size

# Arguments
`Q`: MPIStateArrays (array)

# Keyword Arguments
`M`: Mass Matrix (array)

# Return
The cell-average of Q
"""
function cell_average(Q; M = nothing)
    if M==nothing
        return nothing
    end
    return (sum(M .* Q, dims = 1) ./ sum(M , dims = 1))[:]
end

"""
function coordinates(grid::DiscontinuousSpectralElementGrid)

# Description
Gets the (x,y,z) coordinates corresponding to the grid

# Arguments
- `grid`: DiscontinuousSpectralElementGrid

# Return
- `x, y, z`: views of x, y, z coordinates
"""
function coordinates(grid::DiscontinuousSpectralElementGrid)
    x = view(grid.vgeo, :, grid.x1id, :)   # x-direction	
    y = view(grid.vgeo, :, grid.x2id, :)   # y-direction	
    z = view(grid.vgeo, :, grid.x3id, :)   # z-direction
    return x, y, z
end

"""
function cell_average(Q; M = nothing)

# Description
Get the cell-centers of every element in the grid

# Arguments
- `grid`: DiscontinuousSpectralElementGrid

# Return
- Tuple of cell-centers
"""
function cell_centers(grid::DiscontinuousSpectralElementGrid)
    x, y, z = coordinates(grid)
    M = view(grid.vgeo, :, grid.Mid, :)  # mass matrix
    xC = cell_average(x, M = M)
    yC = cell_average(y, M = M)
    zC = cell_average(z, M = M)
    return xC[:], yC[:], zC[:]
end