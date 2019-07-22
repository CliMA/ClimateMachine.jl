"""
    Grids

A simple 1-dimensional uniform grid
for finite difference method.
"""
module Grids

export Grid, over_elems, over_elems_real, over_elems_ghost
export first_elem_above_surface, get_z

"""
    Grid{T}

A simple 1-dimensional uniform grid of
type `T` for finite difference method.
"""
struct Grid{T}
  """
  Number of physical elements
  """
  n_elem :: Int

  """
  Element size
  """
  dz :: T

  """
  Inverse of element size squared
  """
  dzi :: T

  """
  Inverse of element size squared
  """
  dzi2 :: T

  """
  z-coordinate
  """
  z :: Vector{T}

  """
  Number of ghost points per side
  """
  n_ghost :: Int

  """
  z-coordinate at the cell center at the top of the domain
  """
  zc_top :: T

  """
  z-coordinate at the top domain boundary
  """
  zn_top :: T

  """
  z-coordinate at the cell center at the bottom of the domain
  """
  zc_surf :: T

  """
  z-coordinate at the bottom domain boundary
  """
  zn_surf :: T
end

"""
    Grid(z_min::T, z_max::T, n_elem_real::Int, n_ghost::Int = 1)

A simple grid implementation that accepts the domain interval
`z_min` and `z_max`, the number of elements `n_elem_real` and
the number of ghost points per side `n_ghost`.
"""
function Grid(z_min::T, z_max::T, n_elem_real::Int, n_ghost::Int = 1) where T
  n_elem = n_elem_real+2*n_ghost
  dz = (z_max-z_min)/n_elem_real
  dzi = inv(dz)
  dzi2 = dzi^2
  z = [(i-n_ghost)*dz-dz/2 for i in 1:n_elem]
  zc_surf = z[1+n_ghost]
  zn_surf = (z[n_ghost]+z[1+n_ghost])/2
  zc_top = z[end-n_ghost]
  zn_top = (z[end-n_ghost]+z[end-n_ghost+1])/2
  return Grid(n_elem, dz, dzi, dzi2, z, n_ghost, zc_top, zn_top, zc_surf, zn_surf)
end

function Base.show(io::IO, grid::Grid)
  println(io, "----------------- Grid")
  println(io, "z       = ",grid.z)
  println(io, "n_elem  = ",grid.n_elem)
  println(io, "n_ghost = ",grid.n_ghost)
  println(io, "zc_top  = ",grid.zc_top)
  println(io, "zn_top  = ",grid.zn_top)
  println(io, "zc_surf = ",grid.zc_surf)
  println(io, "zn_surf = ",grid.zn_surf)
  println(io, "-----------------")
end

"""
    get_z(grid::Grid, k::Int)

Get the z-coordinate given element index
"""
@inline get_z(grid::Grid, k::Int) = grid.z[k]

"""
    over_elems(grid::Grid)

Get the range of indexes to traverse real and ghost grid elements
"""
@inline over_elems(grid::Grid) = 1:grid.n_elem

"""
    over_elems_real(grid::Grid)

Get the range of indexes to traverse only real grid elements
"""
@inline over_elems_real(grid::Grid) = 1+grid.n_ghost : grid.n_elem-grid.n_ghost

"""
    over_elems_ghost(grid::Grid)

Get the range of indexes to traverse only ghost grid elements
"""
@inline over_elems_ghost(grid::Grid) = setdiff(over_elems(grid), over_elems_real(grid))

"""
    first_elem_above_surface(grid::Grid)

Get the first element index above the surface
"""
@inline first_elem_above_surface(grid::Grid) = 1+grid.n_ghost

end
