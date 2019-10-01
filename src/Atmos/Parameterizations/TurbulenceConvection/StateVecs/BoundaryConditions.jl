#### BoundaryConditions

# A set of functions that apply Dirichlet or Neumann boundary conditions
# at the grid boundaries, given a state vector.

export Dirichlet, Neumann
export apply_Dirichlet!, apply_Neumann!
export DiffusionAbsorbed, AdvectionAbsorbed
export bc_source

abstract type BoundaryLocation end

"""
    AbsorbType

A boundary condition absorption type, used for dispatch.

See `bc_source` for more information
"""
abstract type AbsorbType end
struct DiffusionAbsorbed <: AbsorbType end
struct AdvectionAbsorbed <: AbsorbType end

"""
    BoundaryConditionType

A boundary condition type, used for dispatch. When
"""
abstract type BoundaryConditionType end
struct Dirichlet <: BoundaryConditionType end
struct Neumann <: BoundaryConditionType end

"""
    apply_Dirichlet!(q::StateVec, ϕ::Symbol, grid::Grid, val, B::Union{ZBoundary,Tuple{Zmin,Zmax}}, i=0)

Apply Dirichlet boundary conditions at the specified boundaries of the domain
"""
function apply_Dirichlet!(q::StateVec, ϕ::Symbol, grid::Grid{DT}, val::DT,
                          B::Union{ZBoundary,Tuple{Zmin,Zmax}}, i=0) where DT
  !(B isa Tuple) && (B = (B,))
  i==0 && (i = gridmean(DomainIdx(q)))
  for b in B
    kg = first_ghost(grid, b)
    ki = first_interior(grid, b)
    q[ϕ, kg, i] = 2*val - q[ϕ, ki, i]
  end
end

"""
    apply_Neumann!(q::StateVec, ϕ::Symbol, grid::Grid, val, B::Union{ZBoundary,Tuple{Zmin,Zmax}}, i=0)

Apply Neumann boundary conditions at the bottom of the domain.

    ∇u = g n̂

Where `n̂` is the outward facing normal and `u` is the variable `ϕ` in `q`.
"""
function apply_Neumann!(q::StateVec, ϕ::Symbol, grid::Grid{DT}, val::DT,
                        B::Union{ZBoundary,Tuple{Zmin,Zmax}}, i=0) where DT
  !(B isa Tuple) && (B = (B,))
  i==0 && (i = gridmean(DomainIdx(q)))
  for b in B
    kg = first_ghost(grid, b)
    ki = first_interior(grid, b)
    q[ϕ, kg, i] = q[ϕ, ki, i] + grid.Δz*val*n_hat(b)
  end
end

"""
    bc_source(q::StateVec,
              grid::Grid,
              tmp::StateVec,
              x::S,
              K::S,
              b::ZBoundary,
              val,
              ::BoundaryConditionType,
              ::AbsorbType,
              i=1) where {S<:Symbol}

Compute a boundary condition source term, absorbed by `::AbsorbType`, for the `BoundaryConditionType` case.

More concretely, consider the linear system:

``Ax = A(x_interior + x_bc) = b``

We may, instead, write this as

``A x_interior = b - A x_bc``

This allows a generic expression, with
respect to boundary conditions, of the
linear system. The `bc_source` functions
compute `A x_bc` for any combination of

 - Diffusion or Advection operators
 - Dirichlet or Neumann boundary conditions
"""
function bc_source(q::StateVec,
                   grid::Grid{DT},
                   tmp::StateVec,
                   x::S, ρ::S, a::S, K::S,
                   b::ZBoundary,
                   val,
                   ::BoundaryConditionType, ::AbsorbType, i=0) where {DT, S<:Symbol}
end

function bc_source(q::StateVec,
                   grid::Grid{DT},
                   tmp::StateVec,
                   x::S, ρ::S, a::S, K::S,
                   b::ZBoundary,
                   val::DT,
                   ::Dirichlet, ::DiffusionAbsorbed, i=0) where {DT, S<:Symbol}
  i==0 && (i = gridmean(DomainIdx(q)))
  ki = first_interior(grid, b)
  ghost_val = 2*val - q[x, ki, i]
  ghost_cut = ghost_vec(b)
  K_dual = tmp[K, Dual(ki), i]
  ρaK_dual = tmp[ρ, Dual(ki)] .* q[a, Dual(ki), i] .* K_dual
  bc_src = Δ_z_dual([ghost_val, ghost_val, ghost_val] .* ghost_cut, grid, ρaK_dual)
  return bc_src
end

function bc_source(q::StateVec,
                   grid::Grid{DT},
                   tmp::StateVec,
                   x::S, ρ::S, a::S, K::S,
                   b::ZBoundary,
                   val::DT,
                   ::Neumann, ::DiffusionAbsorbed, i=0) where {DT, S<:Symbol}
  i==0 && (i = gridmean(DomainIdx(q)))
  ki = first_interior(grid, b)
  kg = first_ghost(grid, b)
  ρ_dual = tmp[ρ, Dual(ki)]
  a_dual = q[a, Dual(ki), i]
  K_dual = tmp[K, Dual(ki), i]
  ρaK_dual = ρ_dual .* a_dual .* K_dual
  # K_boundary = ρaK_dual[binary(b)+1] # TODO: Check whether K should include/exclude ρa.
  K_boundary = K_dual[binary(b)+1]
  if abs(val)<eps(typeof(val)) # if val = 0, allow K = 0
    ghost_val = q[x, ki, i]
  else
    ghost_val = q[x, ki, i] + val*grid.Δz/K_boundary*n_hat(b)
  end
  ghost_cut = ghost_vec(b)
  bc_src = Δ_z_dual([ghost_val, ghost_val, ghost_val] .* ghost_cut, grid, ρaK_dual)
  return bc_src
end

function bc_source(q::StateVec,
                   grid::Grid{DT},
                   tmp::StateVec,
                   x::S, ρ::S, a::S, K::S,
                   b::ZBoundary,
                   val::DT,
                   ::Dirichlet, ::AdvectionAbsorbed, i=0) where {DT,S<:Symbol}
  i==0 && (i = gridmean(DomainIdx(q)))
  gd = ghost_dual(b)
  ki = first_interior(grid, b)
  ghost_val = 2*val - q[x, ki, i]
  bc_src = -∇_z_flux(tmp[ρ, Dual(ki)] .* q[a, Dual(ki), i] .* [ghost_val, ghost_val] .* gd, grid)
end

function bc_source(q::StateVec,
                   grid::Grid{DT},
                   tmp::StateVec,
                   x::S, ρ::S, a::S, K::S,
                   b::ZBoundary,
                   val::DT,
                   ::Neumann, ::AdvectionAbsorbed, i=0) where {DT, S<:Symbol}
  i==0 && (i = gridmean(DomainIdx(q)))
  gd = ghost_dual(b)
  ki = first_interior(grid, b)
  F = tmp[ρ, Dual(ki)] .* q[a, Dual(ki), i] .* [val, val] .* gd
  bc_src = -∇_z_flux(F, grid)
  return bc_src
end
