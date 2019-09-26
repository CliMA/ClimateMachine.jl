#### BoundaryConditions

# A set of functions that apply Dirichlet and Neumann boundary conditions
# at the grid boundaries, given a state vector.

export Dirichlet, Neumann
export apply_Dirichlet!, apply_Neumann!
export DiffusionAbsorbed, AdvectionAbsorbed
export bc_source

abstract type BoundaryLocation end

abstract type AbsorbType end
struct DiffusionAbsorbed <: AbsorbType end
struct AdvectionAbsorbed <: AbsorbType end

abstract type BoundaryConditionType end
struct Dirichlet <: BoundaryConditionType end
struct Neumann <: BoundaryConditionType end

"""
    apply_Dirichlet!(sv::StateVec, name::Symbol, grid::Grid, val, b::ZBoundary, i_sd=1)

Apply Dirichlet boundary conditions at the bottom of the domain
"""
function apply_Dirichlet!(sv::StateVec, name::Symbol, grid::Grid, val, b::ZBoundary, i_sd=0)
  i_sd==0 && (i_sd = gridmean(DomainIdx(sv)))
  kg = first_ghost(grid, b)
  ki = first_interior(grid, b)
  sv[name, kg, i_sd] = 2*val - sv[name, ki, i_sd]
end
function apply_Dirichlet!(sv::StateVec, name::Symbol, grid::Grid, val, B::Tuple{Zmin,Zmax}, i_sd=0)
  i_sd==0 && (i_sd = gridmean(DomainIdx(sv)))
  for b in B
    kg = first_ghost(grid, b)
    ki = first_interior(grid, b)
    sv[name, kg, i_sd] = 2*val - sv[name, ki, i_sd]
  end
end

"""
    apply_Neumann!(sv::StateVec, name::Symbol, grid::Grid, val, b::ZBoundary, i_sd=1)

Apply Neumann boundary conditions at the bottom of the domain.

    ∇u = g n̂

Where `n̂` is the outward facing normal and `u` is the variable `name` in `sv`.
"""
function apply_Neumann!(sv::StateVec, name::Symbol, grid::Grid, val, b::ZBoundary, i_sd=0)
  i_sd==0 && (i_sd = gridmean(DomainIdx(sv)))
  kg = first_ghost(grid, b)
  ki = first_interior(grid, b)
  sv[name, kg, i_sd] = sv[name, ki, i_sd] + grid.Δz*val*n_hat(b)
end
function apply_Neumann!(sv::StateVec, name::Symbol, grid::Grid, val, B::Tuple{Zmin,Zmax}, i_sd=0)
  i_sd==0 && (i_sd = gridmean(DomainIdx(sv)))
  for b in B
    kg = first_ghost(grid, b)
    ki = first_interior(grid, b)
    sv[name, kg, i_sd] = sv[name, ki, i_sd] + grid.Δz*val*n_hat(b)
  end
end

"""
    bc_source(q::StateVec, grid::Grid, tmp::StateVec, x_sym::Symbol, K_sym::Symbol, b::ZBoundary, val, ::Dirichlet, ::DiffusionAbsorbed, i_sd=1)

Compute a boundary condition source term, absorbed by Diffusion, for the Dirichlet case.
"""
function bc_source(q::StateVec, grid::Grid,
                   tmp::StateVec, x_sym::Symbol,
                   K_sym::Symbol, b::ZBoundary,
                   val, ::Dirichlet, ::DiffusionAbsorbed, i_sd=0)
  i_sd==0 && (i_sd = gridmean(DomainIdx(q)))
  ki = first_interior(grid, b)
  ghost_val = 2*val - q[x_sym, ki, i_sd]
  ghost_cut = ghost_vec(b)
  K_dual = tmp[K_sym, Dual(ki), i_sd]
  ρaK_dual = tmp[:ρ_0, Dual(ki)] .* q[:a, Dual(ki), i_sd] .* K_dual
  bc_src = Δ_z_dual([ghost_val, ghost_val, ghost_val] .* ghost_cut, grid, ρaK_dual)
  return bc_src
end

"""
    bc_source(q::StateVec, grid::Grid, tmp::StateVec, x_sym::Symbol, K_sym::Symbol, b::ZBoundary, val, ::Neumann, ::DiffusionAbsorbed, i_sd=1)

Compute a boundary condition source term, absorbed by Diffusion, for the Neumann case.
"""
function bc_source(q::StateVec, grid::Grid,
                   tmp::StateVec, x_sym::Symbol,
                   K_sym::Symbol, b::ZBoundary,
                   val, ::Neumann, ::DiffusionAbsorbed, i_sd=0)
  i_sd==0 && (i_sd = gridmean(DomainIdx(q)))
  ki = first_interior(grid, b)
  kg = first_ghost(grid, b)
  ρ_dual = tmp[:ρ_0, Dual(ki)]
  a_dual = q[:a, Dual(ki), i_sd]
  K_dual = tmp[K_sym, Dual(ki), i_sd]
  ρaK_dual = ρ_dual .* a_dual .* K_dual
  # K_boundary = ρaK_dual[binary(b)+1]
  K_boundary = K_dual[binary(b)+1]
  if abs(val)<eps(typeof(val)) # if val = 0, allow K = 0
    ghost_val = q[x_sym, ki, i_sd]
  else
    ghost_val = q[x_sym, ki, i_sd] + val*grid.Δz/K_boundary*n_hat(b)
  end
  ghost_cut = ghost_vec(b)
  bc_src = Δ_z_dual([ghost_val, ghost_val, ghost_val] .* ghost_cut, grid, ρaK_dual)
  return bc_src
end

"""
    bc_source(q::StateVec, grid::Grid, tmp::StateVec, x_sym::Symbol, b::ZBoundary, val, ::Dirichlet, ::AdvectionAbsorbed, i_sd=1)

Compute a boundary condition source term, absorbed by Advection, for the Dirichlet case.
"""
function bc_source(q::StateVec, grid::Grid,
                   tmp::StateVec, x_sym::Symbol,
                   b::ZBoundary, val, ::Dirichlet, ::AdvectionAbsorbed, i_sd=0)
  i_sd==0 && (i_sd = gridmean(DomainIdx(q)))
  gd = ghost_dual(b)
  ki = first_interior(grid, b)
  ghost_val = 2*val - q[x_sym, ki, i_sd]
  bc_src = -∇_z_flux(tmp[:ρ_0, Dual(ki)] .* q[:a, Dual(ki), i_sd] .* [ghost_val, ghost_val] .* gd, grid)
end

"""
    bc_source(q::StateVec, grid::Grid, tmp::StateVec, x_sym::Symbol, ρa_sym::Symbol, b::ZBoundary, val, ::Neumann, ::AdvectionAbsorbed, i_sd=1)

Compute a boundary condition source term, absorbed by Advection, for the Neumann case.
"""
function bc_source(q::StateVec, grid::Grid, tmp::StateVec,
                   x_sym::Symbol, ρa_sym::Symbol,
                   b::ZBoundary, val, ::Neumann, ::AdvectionAbsorbed, i_sd=0)
  i_sd==0 && (i_sd = gridmean(DomainIdx(q)))
  gd = ghost_dual(b)
  ki = first_interior(grid, b)
  F = tmp[:ρ_0, Dual(ki)] .* q[:a, Dual(ki), i_sd] .* [val, val] .* gd
  bc_src = -∇_z_flux(F, grid)
  return bc_src
end
