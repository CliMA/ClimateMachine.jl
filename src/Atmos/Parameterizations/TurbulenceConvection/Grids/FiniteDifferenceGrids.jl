"""
    FiniteDifferenceGrids

A simple 1-dimensional uniform grid
for finite difference method.
"""
module FiniteDifferenceGrids

export Grid, over_elems, over_elems_real, over_elems_ghost
export ZBoundary, Zmin, Zmax, n_hat, binary, ghost_vec, ghost_dual
export first_interior, boundary, second_interior, first_ghost, boundary_points

"""
    Grid{T, I}

A simple 1-dimensional uniform grid of
type `T` for finite difference method.
"""
struct Grid{T, I}
  """
  Number of physical elements
  """
  n_elem :: I

  """
  Number of real elements
  """
  n_elem_real :: I

  """
  Number of ghost points per side
  """
  n_ghost :: I

  """
  Element size
  """
  Δz :: T

  """
  Inverse of element size squared
  """
  Δzi :: T

  """
  Inverse of element size squared
  """
  Δzi2 :: T

  """
  z-coordinate at cell edges
  """
  ze :: Vector{T}

  """
  z-coordinate at cell centers
  """
  zc :: Vector{T}

  """
  z-coordinate at bottom of boundary node
  """
  zn_min :: T

  """
  z-coordinate at top of boundary node
  """
  zn_max :: T
end

"""
    Grid(z_min::T, z_max::T, n_elem_real::Int, n_ghost::Int = 1)

A simple grid implementation that accepts the domain interval
`z_min` and `z_max`, the number of elements `n_elem_real` and
the number of ghost points per side `n_ghost`.
"""
function Grid(z_min::T, z_max::T, n_elem_real::I, n_ghost::I = 1) where {T, I}
  n_elem = n_elem_real+2*n_ghost
  Δz = (z_max-z_min)/n_elem_real
  Δzi = inv(Δz)
  Δzi2 = Δzi^2
  zc = [(i-n_ghost)*Δz-Δz/2 for i in 1:n_elem]
  ze = [(i-n_ghost)*Δz-Δz for i in 1:n_elem+1]
  zn_min = (zc[n_ghost]+zc[1+n_ghost])/2
  zn_max = (zc[end-n_ghost]+zc[end-n_ghost+1])/2
  return Grid{T, I}(n_elem, n_elem_real, n_ghost, Δz, Δzi, Δzi2, ze, zc, zn_min, zn_max)
end

function Base.show(io::IO, grid::Grid)
  println(io, "----------------- Grid")
  println(io, "zc      = ",grid.zc)
  println(io, "ze      = ",grid.ze)
  println(io, "n_elem  = ",grid.n_elem)
  println(io, "n_ghost = ",grid.n_ghost)
  println(io, "zn_min  = ",grid.zn_min)
  println(io, "zn_max  = ",grid.zn_max)
  println(io, "-----------------")
end

abstract type ZBoundary end

"""
    Zmax
Type to dispatch function to particular location
"""
struct Zmax <: ZBoundary end

"""
    Zmin
Type to dispatch function to particular location
"""
struct Zmin <: ZBoundary end

"""
    n_hat(::ZBoundary)

The outward normal vector to the boundary
"""
n_hat(::Zmin) = -1
n_hat(::Zmax) = 1

"""
    binary(::ZBoundary)

Returns 0 for Zmin and 1 for Zmax
"""
binary(::Zmin) = 0
binary(::Zmax) = 1

"""
    ghost_vec(::ZBoundary)

A 3-element vector near the boundary with 1's on ghost cells and 0's on interior cells
"""
ghost_vec(::Zmin) = [1, 0, 0]
ghost_vec(::Zmax) = [0, 0, 1]

"""
    ghost_dual(::ZBoundary)

A 2-element vector near the boundary with 1's on ghost cells and 0's on interior cells
"""
ghost_dual(::Zmin) = [1, 0]
ghost_dual(::Zmax) = [0, 1]

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
    first_interior(grid::Grid, ::ZBoundary)

Get the first element index in the physical domain
"""
@inline first_interior(grid::Grid, ::Zmax) = grid.n_elem-grid.n_ghost
@inline first_interior(grid::Grid, ::Zmin) = 1+grid.n_ghost

"""
    boundary(grid::Grid, ::ZBoundary)

Get the element index for edge data at the boundary
"""
@inline boundary(grid::Grid, ::Zmax) = grid.n_elem-grid.n_ghost
@inline boundary(grid::Grid, ::Zmin) = 1+grid.n_ghost

"""
    second_interior(grid::Grid, ::ZBoundary)

Get the second element index in the physical domain
"""
@inline second_interior(grid::Grid, ::Zmax) = first_interior(grid, Zmax())-1
@inline second_interior(grid::Grid, ::Zmin) = first_interior(grid, Zmin())+1

"""
    first_ghost(grid::Grid, ::ZBoundary)

Get the first element index in the ghost domain
"""
@inline first_ghost(grid::Grid, ::Zmax) = first_interior(grid, Zmax())+1
@inline first_ghost(grid::Grid, ::Zmin) = first_interior(grid, Zmin())-1

"""
    boundary_points(grid::Grid, ::ZBoundary)

Get the first element index in the ghost domain
"""
@inline boundary_points(grid::Grid, b::ZBoundary) = first_ghost(grid, b), first_interior(grid, b), second_interior(grid, b)

"""
    over_elems_ghost(grid::Grid)

Get the range of indexes to traverse only ghost grid elements
"""
@inline over_elems_ghost(grid::Grid) = setdiff(over_elems(grid), over_elems_real(grid))

"""
    over_elems_ghost(grid::Grid, ::ZBoundary)

Get the range of indexes to traverse only ghost grid elements below ``z_{min}`` and above ``z_{max}``.
"""
@inline over_elems_ghost(grid::Grid, ::Zmin) = 1 : grid.n_ghost
@inline over_elems_ghost(grid::Grid, ::Zmax) = grid.n_elem-grid.n_ghost+1 : grid.n_elem

Base.eltype(::Grid{T}) where T = T

include("GridOperators.jl")

end