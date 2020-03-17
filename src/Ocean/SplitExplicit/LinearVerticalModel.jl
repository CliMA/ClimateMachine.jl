# Linear model for 1D IMEX
struct LinearVerticalModel{M} <: AbstractOceanModel
  ocean::M
  function LinearVerticalModel(ocean::M) where {M}
    return new{M}(ocean)
  end
end

function calculate_dt(grid, model::LinearVerticalModel, Courant_number)
    minΔx = min_node_distance(grid, HorizontalDirection())

    CFL_gravity = minΔx / model.ocean.cʰ
    CFL_diffusive = minΔx^2 / model.ocean.κʰ
    CFL_viscous = minΔx^2 / model.ocean.νʰ

    dt = 1//10 * minimum([CFL_gravity, CFL_diffusive, CFL_viscous])

    return dt
end

vars_state(lm::LinearVerticalModel, FT) = vars_state(lm.ocean,FT)
vars_gradient(lm::LinearVerticalModel, FT) = vars_gradient(lm.ocean,FT)
vars_diffusive(lm::LinearVerticalModel, FT) = vars_diffusive(lm.ocean,FT)
vars_aux(lm::LinearVerticalModel, FT) = vars_aux(lm.ocean,FT)
vars_integrals(lm::LinearVerticalModel, FT) = @vars()

@inline integrate_aux!(::LinearVerticalModel, _...) = nothing
@inline flux_nondiffusive!(::LinearVerticalModel, _...) = nothing
@inline source!(::LinearVerticalModel, _...) = nothing

function wavespeed(lm::LinearVerticalModel, n⁻, _...)
  C = abs(SVector(lm.ocean.cʰ, lm.ocean.cʰ, lm.ocean.cᶻ)' * n⁻)
  return C
end

@inline function boundary_state!(nf, lm::LinearVerticalModel, Q⁺::Vars, A⁺::Vars,
                                 n⁻, Q⁻::Vars, A⁻::Vars, bctype, t, _...)
  return ocean_boundary_state!(lm.ocean, lm.ocean.problem, bctype, nf, Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
end

@inline function boundary_state!(nf, lm::LinearVerticalModel, Q⁺::Vars, D⁺::Vars, A⁺::Vars,
                                 n⁻, Q⁻::Vars, D⁻::Vars, A⁻::Vars, bctype, t, _...)
  return ocean_boundary_state!(lm.ocean, lm.ocean.problem, bctype, nf, Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
end

init_aux!(lm::LinearVerticalModel, A::Vars, geom::LocalGeometry) = nothing
init_state!(lm::LinearVerticalModel, Q::Vars, A::Vars, coords, t) = nothing

@inline function flux_diffusive!(lm::LinearVerticalModel, F::Grad, Q::Vars, D::Vars,
                                 A::Vars, t::Real)
  F.u -= Diagonal(A.ν) * D.∇u
  F.θ -= Diagonal(A.κ) * D.∇θ

  return nothing
end

@inline function gradvariables!(m::LinearVerticalModel, G::Vars, Q::Vars, A, t)
  G.u = Q.u
  G.θ = Q.θ

  return nothing
end

@inline function diffusive!(lm::LinearVerticalModel, D::Vars, G::Grad, Q::Vars,
                            A::Vars, t)
  D.∇u = G.u
  D.∇θ = G.θ

  return nothing
end
