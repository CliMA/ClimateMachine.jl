module TriDiagSolvers

include("TriDiagSolverFuncs.jl")

export solve_tdma!
export solve_tridiag_wrapper!

using LinearAlgebra
using ..FiniteDifferenceGrids
using ..StateVecs

struct DifussionImplicit end
function TridiagonalMatrix(q::StateVec, tmp::StateVec, Δt::T, ρ::S, a::S, K::S, grid::Grid{T}, ::DifussionImplicit, i::Int) where {S<:Symbol, T}
  a_ = [   -Δt*grid.Δzi2*     (tmp[ρ, Dual(k)] .* q[a, Dual(k), i] .* tmp[K, Dual(k), i])[2] for k in over_elems_real(grid)[1:end-1]]
  b_ = [1 + Δt*grid.Δzi2*(sum((tmp[ρ, Dual(k)] .* q[a, Dual(k), i] .* tmp[K, Dual(k), i])))  for k in over_elems_real(grid)]
  c_ = [   -Δt*grid.Δzi2*     (tmp[ρ, Dual(k)] .* q[a, Dual(k), i] .* tmp[K, Dual(k), i])[1] for k in over_elems_real(grid)[2:end]]
  return Tridiagonal(a_,b_,c_)
end

"""
    solve_tdma!(q::StateVec, tendencies::StateVec, tmp::StateVec, x::S, ρ::S, a_τp1::S, a_τ::S, K::S, grid::Grid{T}, Δt::T, i::Int=1)

Solve tri-diagonal system using Diffusion-implicit time-marching.
"""
function solve_tdma!(q::StateVec, tendencies::StateVec, tmp::StateVec, x::S, ρ::S, a_τp1::S, a_τ::S, K::S, grid::Grid{T}, Δt::T, i::Int=1) where {S<:Symbol, T}
  B = [tmp[a_τ, k, i]/q[a_τp1, k, i]*q[x, k, i] +
  Δt*tendencies[x, k, i]/(tmp[ρ, k]*q[a_τp1, k, i])
  for k in over_elems_real(grid)]
  A = TridiagonalMatrix(q, tmp, Δt, ρ, a_τp1, K, grid, DifussionImplicit(), i)
  X = inv(A)*B
  assign_real!(q, x, grid, X, i)
end

"""
    solve_tridiag_wrapper!(grid::Grid, sv::StateVec, ϕ::Symbol, i::Int, tri_diag::StateVec)

Solve the tri-diagonal system, given by state vector `tri_diag` for variable `ϕ` in sub-domain `i`.
"""
function solve_tridiag_wrapper!(grid::Grid, sv::StateVec, ϕ::Symbol, i::Int, tri_diag::StateVec)
  f = [tri_diag[:f, k] for k in over_elems_real(grid)]
  a = [tri_diag[:a, k] for k in over_elems_real(grid)]
  b = [tri_diag[:b, k] for k in over_elems_real(grid)]
  c = [tri_diag[:c, k] for k in over_elems_real(grid)]
  solve_tridiag_old(grid.n_elem_real, f, a, b, c)
  assign_real!(sv, ϕ, grid, f, i)
end

# TODO: Replace this with Tridiagonal/inv
function solve_tridiag_old(nz::I, x::Vector{T}, a::Vector{T}, b::Vector{T}, c::Vector{T}) where {T<:AbstractFloat,I<:Int}
  scratch = deepcopy(x)
  scratch[1] = c[1]/b[1]
  x[1] = x[1]/b[1]
  @inbounds for i in 2:nz
    m = 1/(b[i] - a[i] * scratch[i-1])
    scratch[i] = c[i] * m
    x[i] = (x[i] - a[i] * x[i-1])*m
  end
  @inbounds for i in nz-1:-1:1
    x[i] = x[i] - scratch[i] * x[i+1]
  end
end

end
