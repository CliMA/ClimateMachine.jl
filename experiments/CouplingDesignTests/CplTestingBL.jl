"""
    module CplTestingBL
 Defines an equation set for testing coupling

 Defines kernels to evaluates RHS of

  d ϕ / dt = - div κ( ϕ_init ) grad ϕ

 subject to specified prescribed gradient or flux bc's

 - [`CplTestBL`](@ref)
     balance law struct created by this module

"""
module CplTestingBL

export CplTestBL
export PenaltyNumFluxDiffusive
export ExteriorBoundaryCondition
export CoupledBoundaryCondition


using ClimateMachine.BalanceLaws:
      Auxiliary,
      BalanceLaw,
      Gradient,
      GradientFlux,
      Prognostic

import ClimateMachine.BalanceLaws:
       boundary_conditions,
       boundary_state!,
       compute_gradient_argument!,
       compute_gradient_flux!,
       flux_first_order!,
       flux_second_order!,
       init_state_prognostic!,
       nodal_init_state_auxiliary!,
       source!,
       vars_state,
       wavespeed

using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.MPIStateArrays

using ClimateMachine.DGMethods.NumericalFluxes:
      CentralNumericalFluxGradient,
      CentralNumericalFluxSecondOrder,
      NumericalFluxFirstOrder,
      NumericalFluxSecondOrder,
      RusanovNumericalFlux

import ClimateMachine.DGMethods.NumericalFluxes:
       numerical_boundary_flux_second_order!,
       numerical_flux_second_order!


using  ClimateMachine.VariableTemplates

using LinearAlgebra
using StaticArrays

"""
    CplTestBL
Type that holds specification for a diffusion equation balance law instance.
"""
struct CplTestBL{FT, BLP} <: BalanceLaw
        bl_prop :: BLP
        function CplTestBL{FT}(;
          bl_prop::BLP=nothing,
        ) where {FT, BLP}
          return new{FT,BLP}(
                 bl_prop,
                 )
        end
end

l_type=CplTestBL

"""
  Extend the NumericalFluxSecondOrder to include a penalty term numerical
  flux formulation.
"""
struct PenaltyNumFluxDiffusive <: NumericalFluxSecondOrder end

"""
  Set a default set of properties and their default values
  - init_aux_geom   :: function to initialize geometric terms stored in aux.
  - init_theta      :: function to set initial θ values.
  - source_theta    :: function to add a source term to θ.
  - calc_kappa_diff :: function to set diffusion coeffiecient(s).
  - get_wavespeed   :: function to return a wavespeed for Rusanov computations (there aren't any in this model)
  - get_penalty_tau :: function to set timescale on which to bring state+ and state- together
  - theta_shadow_boundary_flux :: function to set boundary flux into shadow variable for passing to coupler
"""
function prop_defaults()
  bl_prop=NamedTuple()

  function init_aux_geom(npt,elnum,x,y,z)
    return npt, elnum,x,y,z
  end
  # init_aux_geom(_...)=(return 0., 0., 0., 0., 0.)
  bl_prop=( bl_prop...,init_aux_geom=init_aux_geom )

  init_theta(_...)=(return 0.)
  bl_prop=( bl_prop...,   init_theta=init_theta )

  source_theta(_...)=(return 0.)
  bl_prop=( bl_prop..., source_theta=source_theta )

  calc_kappa_diff(_...)=(return 0., 0., 0.)
  bl_prop=( bl_prop..., calc_kappa_diff=calc_kappa_diff)

  get_wavespeed(_...)=(return 0.)
  bl_prop=( bl_prop..., get_wavespeed=get_wavespeed )

  get_penalty_tau(_...)=(return 1.)
  bl_prop=( bl_prop..., get_penalty_tau=get_penalty_tau )

  theta_shadow_boundary_flux(_...)=(return 0.)
  bl_prop=( bl_prop..., theta_shadow_boundary_flux=theta_shadow_boundary_flux )

  bl_prop=( bl_prop..., LAW=CplTestBL )
end

"""
  Declare single prognostic state variable, θ
  and a shadow variable, θ_boundary_export, for accumulating boundary flux over a single 
  timestep.

  "Shadow" variables XX_boundary_export will be used to capture boundary fluxes that we 
  want to export to coupling as time integrals. We use a shadow variable 
  because we want to integrate over whatever timestepper is being used. 
  Eventually we should have the ability to potentially use a 2d field here. 
  The shadow variable needs to be zeroed at the start of each
  coupling cycle for a component.
"""
function vars_state(bl::l_type, ::Prognostic, FT)
  @vars begin
     θ::FT
     θ_boundary_export::FT
  end
end

"""
  Auxiliary state variable symbols for array index and real world coordinates and for
  θ value at reference time used to compute κ.
"""
function vars_state(bl::l_type, ::Auxiliary, FT)
  @vars begin
       npt::Int
     elnum::Int
        xc::FT
        yc::FT
        zc::FT
     θⁱⁿⁱᵗ::FT
     boundary_in::FT
     boundary_out::FT
  end
end

"""
  Gradient computation stage input (and output) variable symbols
"""
function vars_state(bl::l_type, ::Gradient, FT)
  @vars begin
        ∇θ::FT
    ∇θⁱⁿⁱᵗ::FT
  end
end

"""
  Flux due to gradient accumulation stage variable symbols
"""
function vars_state(bl::l_type, ::GradientFlux, FT)
  @vars begin
   κ∇θ::SVector{3,FT}
  end
end

"""
  Initialize prognostic state variables
"""
function init_state_prognostic!(bl::l_type, Q::Vars, A::Vars, geom::LocalGeometry, FT)
  npt=getproperty(geom,:n)
  elnum=getproperty(geom,:e)
  x=geom.coord[1]
  y=geom.coord[2]
  z=geom.coord[3]
  Q.θ=bl.bl_prop.init_theta(x,y,z,npt,elnum)
  nothing
end

"""
  Initialize auxiliary state variables
"""
function nodal_init_state_auxiliary!(bl::l_type, A::Vars, tmp::Vars, geom::LocalGeometry, _...)
  npt=getproperty(geom,:n)
  elnum=getproperty(geom,:e)
  x=geom.coord[1]
  y=geom.coord[2]
  z=geom.coord[3]
  A.npt, A.elnum, A.xc, A.yc, A.zc = bl.bl_prop.init_aux_geom(npt,elnum,x,y,z)
  A.θⁱⁿⁱᵗ=0
  A.boundary_in = 0
  A.boundary_out = 0
  nothing
end

#====

Atmos

----

Land


====#




"""
  Set any source terms for prognostic state external sources.
  Also use to record boundary flux terms into shadow variables 
  for export to coupler.
"""
function source!(bl::l_type,S::Vars,Q::Vars,G::Vars,A::Vars,_...)
  S.θ=bl.bl_prop.source_theta(Q.θ,A.npt,A.elnum,A.xc,A.yc,A.zc,A.boundary_in)
  # Record boundary condition fluxes as needed by adding to shadow 
  # prognostic variable
  S.θ_boundary_export=
   bl.bl_prop.theta_shadow_boundary_flux(Q.θ,A.boundary_in,A.npt,A.elnum,A.xc,A.yc,A.zc)
  nothing
end

"""
  No flux first order for diffusion equation, but we must define a stub.
"""
function flux_first_order!( bl::l_type, _...)
  nothing
end

"""
  Set values to have gradients computed.
"""
function compute_gradient_argument!( bl::l_type, G::Vars, Q::Vars, A::Vars, t )
  G.∇θ = Q.θ
  G.∇θⁱⁿⁱᵗ=A.θⁱⁿⁱᵗ
  nothing
end

"""
  Compute diffusivity tensor times computed gradient to give net gradient flux.
"""
function compute_gradient_flux!( bl::l_type, GF::Vars, G::Grad, Q::Vars, A::Vars, t )
  # "Non-linear" form (for time stepped)
  ### κ¹,κ²,κ³=bl.bl_prop.calc_kappa_diff(G.∇θ,A.npt,A.elnum,A.xc,A.yc,A.zc)
  # "Linear" form (for implicit)
  κ¹,κ²,κ³=bl.bl_prop.calc_kappa_diff(G.∇θⁱⁿⁱᵗ,A.npt,A.elnum,A.xc,A.yc,A.zc)
  # Maybe I should pass both G.∇θ and G.∇θⁱⁿⁱᵗ?
  GF.κ∇θ = Diagonal(@SVector([κ¹,κ²,κ³]))*G.∇θ
  nothing
end

"""
  Pass flux components for second order term into update kernel.
"""
function flux_second_order!( bl::l_type, F::Grad, Q::Vars, GF::Vars, H::Vars, A::Vars, t )
  F.θ += GF.κ∇θ
  nothing
end

struct ExteriorBoundaryCondition
end

struct CoupledBoundaryCondition
end

"""
  Define boundary condition flags/types to iterate over, for now keep it simple.
"""
## function boundary_conditions( bl::l_type, _...)
##     (CoupledBoundaryCondition(), ExteriorBoundaryCondition())
## end
function boundary_conditions( bl::l_type, _...)
    (1, 2)
end

"""
  Zero normal gradient boundary condition.
"""
function boundary_state!(nF::Union{CentralNumericalFluxGradient}, bc, bl::l_type, Q⁺::Vars, A⁺::Vars,n,Q⁻::Vars,A⁻::Vars,t,_...)
 Q⁺.θ=Q⁻.θ
 nothing
end

"""
  Set any first order numerical flux to null
  NumericalFluxFirstOrder is an abstract type that currently generalizes
  RusanovNumericalFlux, CentralNumericalFluxFirstOrder, RoeNumericalFlux, HLLCNumericalFlux.
"""
# No first order fluxes so numerical flux needed. NumericalFluxFirstOrder
function boundary_state!(nF::NumericalFluxFirstOrder, bc, bl::l_type, Q⁺::Vars, A⁺::Vars,n,Q⁻::Vars,A⁻::Vars,t,_...)
 nothing
end
# Zero gradient at exterior boundaries
function boundary_state!(nF::Union{NumericalFluxSecondOrder}, bc::ExteriorBoundaryCondition, bl::l_type, Q⁺::Vars, GF⁺::Vars, A⁺::Vars,n⁻,Q⁻::Vars,GF⁻::Vars,A⁻::Vars,t,_...)
 Q⁺.θ=Q⁻.θ
 GF⁺.κ∇θ= n⁻ * -0
 if A⁻.npt == 0
  println("Exterior!!")
 end
 nothing
end
# Use boundary flux
function boundary_state!(nF::Union{NumericalFluxSecondOrder}, bc::CoupledBoundaryCondition, bl::l_type, Q⁺::Vars, GF⁺::Vars, A⁺::Vars,n⁻,Q⁻::Vars,GF⁻::Vars,A⁻::Vars,t,_...)
  ## Need to try this
  ## GF⁺.κ∇θ = n⁻ * aux.boundary_in
  GF⁺.κ∇θ= n⁻ * -0
  if A⁻.npt == 0
   println("Coupled!!")
  end
  nothing
end

# Pending types
function boundary_state!(nF::Union{NumericalFluxSecondOrder}, bc, bl::l_type, Q⁺::Vars, GF⁺::Vars, A⁺::Vars,n⁻,Q⁻::Vars,GF⁻::Vars,A⁻::Vars,t,_...)
  Q⁺.θ=Q⁻.θ
  println("bc=",bc)
  
  if bc == 1
   println("Exterior")
   GF⁺.κ∇θ= n⁻ * -0
   if A⁻.npt == 0
    println("Exterior!!")
   end
  end

  if bc == 2
   println("Coupled")
   ## Need to try this
   ## GF⁺.κ∇θ = n⁻ * aux.boundary_in
   GF⁺.κ∇θ= n⁻ * -0
   if A⁻.npt == 0
    println("Coupled!!")
   end
  end

  nothing
end


function update_auxiliary_state_gradient!(
      dg,
      balance_law::l_type,
      state_prognostic,
      t,
      elems,
    )

    A = dg.state_auxiliary
    D = dg.state_gradient_flux
    A.boundary_out .= D.κ∇θ
end

function wavespeed(bl::l_type, _...)
 # Used in Rusanov term.
 # Only active if there is a flux first order term?
 bl.bl_prop.get_wavespeed()
end

"""
  Penalty flux formulation of second order numerical flux. This formulation
  computes the CentralNumericalFluxSecondOrder term first (which is just the average
  of the + and - fluxes and an edge), and then adds a "penalty" flux that relaxes
  the edge state + and - toward each other.
"""
function numerical_flux_second_order!(
    ::PenaltyNumFluxDiffusive,
    bl::l_type,
    fluxᵀn::Vars{S},
    n::SVector,
    state⁻::Vars{S},
    diff⁻::Vars{D},
    hyperdiff⁻::Vars{HD},
    aux⁻::Vars{A},
    state⁺::Vars{S},
    diff⁺::Vars{D},
    hyperdiff⁺::Vars{HD},
    aux⁺::Vars{A},
    t,
) where {S, HD, D, A}

    numerical_flux_second_order!(
        CentralNumericalFluxSecondOrder(),
        bl,
        fluxᵀn,
        n,
        state⁻,
        diff⁻,
        hyperdiff⁻,
        aux⁻,
        state⁺,
        diff⁺,
        hyperdiff⁺,
        aux⁺,
        t,
    )

    Fᵀn = parent(fluxᵀn)
    FT = eltype(Fᵀn)
    tau = bl.bl_prop.get_penalty_tau()
    Fᵀn .-= tau * (parent(state⁻) - parent(state⁺))
end

# We are assuming zero gradient bc for now - so there is no numerical second order
# flux from boundary
numerical_boundary_flux_second_order!(nf::PenaltyNumFluxDiffusive, _...) = nothing

end
