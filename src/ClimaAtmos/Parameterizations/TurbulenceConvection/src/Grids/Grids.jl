module Grids

import Base
export Grid, over_elems, over_elems_real, over_elems_ghost
export first_cell_above_surface, get_z

struct Grid{T}
  n_elem :: Int
  dz :: T
  dzi :: T
  dzi2 :: T
  z :: Vector{T}
  n_ghost :: Int
  zc_top :: T
  zn_top :: T
  zc_surf :: T
  zn_surf :: T
end

function Grid(z_min::T, z_max::T, n_elem_real::Int, n_ghost::Int = 1) where T
  n_elem = n_elem_real+2*n_ghost
  dz = T((z_max-z_min)/T(n_elem_real))
  dzi = T(1)/dz
  dzi2 = dzi*dzi
  z = Vector{T}([T(i-n_ghost)*dz-dz/2 for i in 1:n_elem])
  zc_surf = z[1+n_ghost]
  zn_surf = (z[n_ghost]+z[1+n_ghost])/2
  zc_top = z[end-n_ghost]
  zn_top = (z[end-n_ghost]+z[end-n_ghost+1])/2
  return Grid(n_elem, dz, dzi, dzi2, z, n_ghost, zc_top, zn_top, zc_surf, zn_surf)
end

function Base.show(io::IO, grid::Grid)
  println("----------------- Grid")
  println("z       = ",grid.z)
  println("n_elem  = ",grid.n_elem)
  println("n_ghost = ",grid.n_ghost)
  println("zc_top  = ",grid.zc_top)
  println("zn_top  = ",grid.zn_top)
  println("zc_surf = ",grid.zc_surf)
  println("zn_surf = ",grid.zn_surf)
  println("-----------------")
end

@inline function get_z(grid::Grid, k::Int)
  return grid.z[k]
end

@inline function over_elems(grid::Grid)
  return 1:grid.n_elem
end

@inline function over_elems_real(grid::Grid)
  return 1+grid.n_ghost:grid.n_elem-grid.n_ghost
end

@inline function over_elems_ghost(grid::Grid)
  return setdiff(over_elems(grid), over_elems_real(grid))
end

@inline function first_cell_above_surface(grid::Grid)
  return 1+grid.n_ghost
end

end