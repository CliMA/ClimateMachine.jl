module Grids

include("BoundaryConditions.jl")
include("FieldTypes.jl")

export Grid
export over_nodes, over_nodes_real, over_centers, over_centers_real
export Center, Node, get_type, suffix

# New interface:
export elems, overelems

using .FieldTypes: Center, Node, get_type, suffix

struct Grid{T}
  n_nodes :: Int
  n_elem :: Int
  dz :: T
  dzi :: T
  dzi2 :: T
  zc :: Vector{T}
  zn :: Vector{T}
  z :: Dict{UnionAll, Vector{T}}
  gw :: Int
  zc_top :: T
  zn_top :: T
  zc_surf :: T
  zn_surf :: T
  zn_range :: T
  inds :: CartesianIndices
  Np :: Int
end

function Grid(a::T, b::T, N::Int, Np = 1) where T
  n_nodes = N+3
  n_elem = N+2
  dz = T((b-a)/T(N))
  dzi = T(1)/dz
  dzi2 = dzi*dzi
  zn = Vector{T}([T(i-2)*dz for i in 1:n_nodes])
  zc = Vector{T}([T(i-1)*dz-T(0.5)*dz for i in 1:n_elem])
  z = Dict(Node => zn, Center => zc)
  gw = typeof(N)(2)
  zc_top = zc[end-gw]
  zn_top = zn[end-gw]
  zc_surf = zc[gw]
  zn_surf = zn[gw]
  zn_range = zn_top - zn_surf
  inds = CartesianIndices((n_elem,))
  return Grid(n_nodes, n_elem, dz, dzi, dzi2, zc, zn, z, gw, zc_top, zn_top, zc_surf, zn_surf, zn_range, inds, Np)
end

@inline function over_nodes(grid::Grid)
  return 1:grid.n_nodes
end

@inline function over_nodes_real(grid::Grid)
  return 2:grid.n_nodes-1
end

@inline function over_centers(grid::Grid)
  return 1:grid.n_elem
end

@inline function over_elems(grid::Grid)
  return 1:grid.n_elem
end

@inline function over_centers_real(grid::Grid)
  return 2:grid.n_elem-1
end

@inline function over_centers_real_reversed(grid::Grid)
  return range(grid.n_elem-1, step = -1, stop = 1)
end

elems(grid::Grid) = grid.inds
function overelems(f::F, grid::Grid, args...) where F
  for I in elems(grid), n in 1:grid.Np
    f(I, n, grid, args...)
  end
end

ghostboundaries(grid::Grid) = first(elems(grid)), last(elems(grid))

@inline neighbor(elem::T, face, grid::Grid) where {T} = elem + face

end