# Linear model for 1D IMEX
"""
    LinearHBModel <: BalanceLaw

A `BalanceLaw` for modeling vertical diffusion implicitly.

write out the equations here

# Usage

    model = HydrostaticBoussinesqModel(problem)
    linear = LinearHBModel(model)

"""
struct LinearHBModel{M} <: BalanceLaw
  ocean::M
  function LinearHBModel(ocean::M) where {M}
    return new{M}(ocean)
  end
end

"""
    calculate_dt(grid, model::HBModel)

calculates the time step based on grid spacing and model parameters
takes minimum of gravity wave, diffusive, and viscous CFL

"""
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

"""
    No integration, hyperbolic flux, or source terms
"""
@inline integrate_aux!(::LinearHBModel, _...) = nothing
@inline flux_nondiffusive!(::LinearHBModel, _...) = nothing
@inline source!(::LinearHBModel, _...) = nothing

"""
    No need to init, initialize by full model
"""
init_aux!(lm::LinearHBModel, A::Vars, geom::LocalGeometry) = nothing
init_state!(lm::LinearHBModel, Q::Vars, A::Vars, coords, t) = nothing

"""
    gradvariables!(::LinearHBModel)
    
copy u and θ to var_gradient
this computation is done pointwise at each nodal point

# arguments:
- `m`: model in this case HBModel
- `G`: array of gradient variables
- `Q`: array of state variables
- `A`: array of aux variables
- `t`: time, not used
"""
@inline function gradvariables!(m::LinearHBModel, G::Vars, Q::Vars, A, t)
  G.u = Q.u
  G.θ = Q.θ

  return nothing
end

"""
    diffusive!(::LinearHBModel)

copy ∇u and ∇θ to var_diffusive
this computation is done pointwise at each nodal point

# arguments:
- `m`: model in this case HBModel
- `D`: array of diffusive variables
- `G`: array of gradient variables
- `Q`: array of state variables
- `A`: array of aux variables
- `t`: time, not used
"""
@inline function diffusive!(lm::LinearHBModel, D::Vars, G::Grad, Q::Vars,
                            A::Vars, t)
  D.∇u = G.u
  D.∇θ = G.θ

  return nothing
end

"""
    flux_diffusive!(::HBModel)

calculates the parabolic flux contribution to state variables
this computation is done pointwise at each nodal point

# arguments:
- `m`: model in this case HBModel
- `F`: array of fluxes for each state variable
- `Q`: array of state variables
- `D`: array of diff variables
- `A`: array of aux variables
- `t`: time, not used

# computations
∂ᵗu = -∇∘(ν∇u)
∂ᵗθ = -∇∘(κ∇θ)
"""
@inline function flux_diffusive!(lm::LinearHBModel, F::Grad, Q::Vars, D::Vars,
                                 HD::Vars, A::Vars, t::Real)
  F.u -= Diagonal(A.ν) * D.∇u
  F.θ -= Diagonal(A.κ) * D.∇θ

  return nothing
end

"""
    wavespeed(::LinaerHBModel)

calculates the wavespeed for rusanov flux
"""
function wavespeed(lm::LinearHBModel, n⁻, _...)
  C = abs(SVector(lm.ocean.cʰ, lm.ocean.cʰ, lm.ocean.cᶻ)' * n⁻)
  return C
end

"""
    boundary_state!(nf, ::LinearHBModel, Q⁺, A⁺, Q⁻, A⁻, bctype)

applies boundary conditions for the hyperbolic fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@inline function boundary_state!(nf, lm::LinearHBModel, Q⁺::Vars, A⁺::Vars,
                                 n⁻, Q⁻::Vars, A⁻::Vars, bctype, t, _...)
  return ocean_boundary_state!(lm.ocean, lm.ocean.problem, bctype, nf, Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
end

"""
    boundary_state!(nf, ::LinearHBModel, Q⁺, D⁺, A⁺, Q⁻, D⁻, A⁻, bctype)

applies boundary conditions for the parabolic fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@inline function boundary_state!(nf, lm::LinearHBModel, Q⁺::Vars, D⁺::Vars, A⁺::Vars,
                                 n⁻, Q⁻::Vars, D⁻::Vars, A⁻::Vars, bctype, t, _...)
  return ocean_boundary_state!(lm.ocean, lm.ocean.problem, bctype, nf, Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
end
