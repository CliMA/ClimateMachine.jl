# Linear model for 1D IMEX
struct LinearHBModel{M} <: BalanceLaw
  ocean::M
  function LinearHBModel(ocean::M) where {M}
    return new{M}(ocean)
  end
end

function calculate_dt(grid, model::LinearHBModel, Courant_number)
    minΔx = min_node_distance(grid, HorizontalDirection())

    CFL_gravity = minΔx / model.ocean.cʰ
    CFL_diffusive = minΔx^2 / model.ocean.κʰ
    CFL_viscous = minΔx^2 / model.ocean.νʰ

    dt = 1//10 * minimum([CFL_gravity, CFL_diffusive, CFL_viscous])

    return dt
end

"""
    Copy over state, aux, and diff variables from HBModel
"""
vars_state(lm::LinearHBModel, FT) = vars_state(lm.ocean,FT)
vars_gradient(lm::LinearHBModel, FT) = vars_gradient(lm.ocean,FT)
vars_diffusive(lm::LinearHBModel, FT) = vars_diffusive(lm.ocean,FT)
vars_aux(lm::LinearHBModel, FT) = vars_aux(lm.ocean,FT)
vars_integrals(lm::LinearHBModel, FT) = @vars()

@inline integrate_aux!(::LinearHBModel, _...) = nothing
@inline flux_nondiffusive!(::LinearHBModel, _...) = nothing
@inline source!(::LinearHBModel, _...) = nothing

function wavespeed(lm::LinearHBModel, n⁻, _...)
  C = abs(SVector(lm.ocean.cʰ, lm.ocean.cʰ, lm.ocean.cᶻ)' * n⁻)
  return C
end

@inline function boundary_state!(nf, lm::LinearHBModel, Q⁺::Vars, A⁺::Vars,
                                 n⁻, Q⁻::Vars, A⁻::Vars, bctype, t, _...)
  return ocean_boundary_state!(lm.ocean, lm.ocean.problem, bctype, nf, Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
end

@inline function boundary_state!(nf, lm::LinearHBModel, Q⁺::Vars, D⁺::Vars, A⁺::Vars,
                                 n⁻, Q⁻::Vars, D⁻::Vars, A⁻::Vars, bctype, t, _...)
  return ocean_boundary_state!(lm.ocean, lm.ocean.problem, bctype, nf, Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
end

init_aux!(lm::LinearHBModel, A::Vars, geom::LocalGeometry) = nothing
init_state!(lm::LinearHBModel, Q::Vars, A::Vars, coords, t) = nothing

@inline function flux_diffusive!(lm::LinearHBModel, F::Grad, Q::Vars, D::Vars,
                                 A::Vars, t::Real)
  F.u -= Diagonal(A.ν) * D.∇u
  F.θ -= Diagonal(A.κ) * D.∇θ

  return nothing
end

@inline function gradvariables!(m::LinearHBModel, G::Vars, Q::Vars, A, t)
  G.u = Q.u
  G.θ = Q.θ

  return nothing
end

@inline function diffusive!(lm::LinearHBModel, D::Vars, G::Grad, Q::Vars,
                            A::Vars, t)
  D.∇u = G.u
  D.∇θ = G.θ

  return nothing
end
