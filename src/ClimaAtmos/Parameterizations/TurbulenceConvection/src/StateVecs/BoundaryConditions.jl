module BoundaryConditions

import Base
using Test, ..Grids, ..StateVecs
export Dirichlet!, Neumann!, Top, Bottom

abstract type BoundaryLocation end
struct Top<:BoundaryLocation end
struct Bottom<:BoundaryLocation end

function Dirichlet!(sv::StateVec, name::Symbol, val, grid, ::Bottom, i_sd=1)
  e = grid.n_ghost
  sv[name, e, i_sd] = 2*val - sv[name, e+1, i_sd]
end
function Dirichlet!(sv::StateVec, name::Symbol, val, grid, ::Top, i_sd=1)
  e = grid.n_elem - (grid.n_ghost-1)
  sv[name, e, i_sd] = 2*val - sv[name, e-1, i_sd]
end

function Neumann!(sv::StateVec, name::Symbol, val, grid, ::Top, i_sd=1)
  e = grid.n_elem - (grid.n_ghost-1)
  sv[name, e, i_sd] = sv[name, e-1, i_sd] + grid.dz*val
end
function Neumann!(sv::StateVec, name::Symbol, val, grid, ::Bottom, i_sd=1)
  e = grid.n_ghost
  sv[name, e, i_sd] = sv[name, e+1, i_sd] - grid.dz*val
end

end