"""
    BoundaryConditions

A set of functions that apply Dirichlet and Neumann boundary conditions
at the top and bottom of the domains of a grid, given a state vector.
"""
module BoundaryConditions

using ..Grids, ..StateVecs
export Dirichlet!, Neumann!, Top, Bottom

abstract type BoundaryLocation end

"""
    Top<:BoundaryLocation

Location to apply boundary condition
"""
struct Top<:BoundaryLocation end

"""
    Bottom<:BoundaryLocation

Location to apply boundary condition
"""
struct Bottom<:BoundaryLocation end

"""
    Dirichlet!(sv::StateVec, name::Symbol, val, grid, ::Bottom, i_sd=1)

Apply Dirichlet boundary conditions at the bottom of the domain
"""
function Dirichlet!(sv::StateVec, name::Symbol, val, grid, ::Bottom, i_sd=1)
  e = grid.n_ghost
  sv[name, e, i_sd] = 2*val - sv[name, e+1, i_sd]
end

"""
    Dirichlet!(sv::StateVec, name::Symbol, val, grid, ::Top, i_sd=1)

Apply Dirichlet boundary conditions at the bottom of the domain
"""
function Dirichlet!(sv::StateVec, name::Symbol, val, grid, ::Top, i_sd=1)
  e = grid.n_elem - (grid.n_ghost-1)
  sv[name, e, i_sd] = 2*val - sv[name, e-1, i_sd]
end

"""
    Neumann!(sv::StateVec, name::Symbol, val, grid, ::Bottom, i_sd=1)

Apply Neumann boundary conditions at the bottom of the domain.

    ∇u = g n̂

Where `n̂` is the outward facing normal and `u` is the variable `name` in `sv`.
"""
function Neumann!(sv::StateVec, name::Symbol, val, grid, ::Bottom, i_sd=1)
  e = grid.n_ghost
  sv[name, e, i_sd] = sv[name, e+1, i_sd] - grid.dz*val
end

"""
    Neumann!(sv::StateVec, name::Symbol, val, grid, ::Top, i_sd=1)

Apply Neumann boundary conditions at the bottom of the domain

    ∇u = g n̂

Where `n̂` is the outward facing normal and `u` is the variable `name` in `sv`.
"""
function Neumann!(sv::StateVec, name::Symbol, val, grid, ::Top, i_sd=1)
  e = grid.n_elem - (grid.n_ghost-1)
  sv[name, e, i_sd] = sv[name, e-1, i_sd] + grid.dz*val
end

end