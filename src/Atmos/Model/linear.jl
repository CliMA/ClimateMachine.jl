"""
    linearized_air_pressure(ρ, ρe_tot, ρe_pot, ρq_tot=0, ρq_liq=0, ρq_ice=0)

The air pressure, linearized around a dry rest state, from the equation of state
(ideal gas law) where:

 - `ρ` (moist-)air density
 - `ρe_tot` total energy density
 - `ρe_pot` potential energy density
and, optionally,
 - `ρq_tot` total water density
 - `ρq_liq` liquid water density
 - `ρq_ice` ice density
"""
function linearized_air_pressure(ρ::FT, ρe_tot::FT, ρe_pot::FT,
                                 ρq_tot::FT=FT(0), ρq_liq::FT=FT(0), ρq_ice::FT=FT(0)) where {FT<:Real}
  ρ*FT(R_d)*FT(T_0) + FT(R_d)/FT(cv_d)*(ρe_tot - ρe_pot - (ρq_tot - ρq_liq)*FT(e_int_v0) + ρq_ice*(FT(e_int_i0) + FT(e_int_v0)))
end

@inline function linearized_pressure(::DryModel, orientation::Orientation, state::Vars, aux::Vars)
  ρe_pot = state.ρ * gravitational_potential(orientation, aux)
  linearized_air_pressure(state.ρ, state.ρe, ρe_pot)
end
@inline function linearized_pressure(::EquilMoist, orientation::Orientation, state::Vars, aux::Vars)
  ρe_pot = state.ρ * gravitational_potential(orientation, aux)
  linearized_air_pressure(state.ρ, state.ρe, ρe_pot, state.moisture.ρq_tot)
end

abstract type AtmosLinearModel <: BalanceLaw
end

function calculate_dt(grid, model::AtmosLinearModel, Courant_number)
    T = 290.0
    return Courant_number * min_node_distance(grid, HorizontalDirection()) / soundspeed_air(T)
end

vars_state(lm::AtmosLinearModel, FT) = vars_state(lm.atmos,FT)
vars_gradient(lm::AtmosLinearModel, FT) = @vars()
vars_diffusive(lm::AtmosLinearModel, FT) = @vars()
vars_aux(lm::AtmosLinearModel, FT) = vars_aux(lm.atmos,FT)
vars_integrals(lm::AtmosLinearModel, FT) = @vars()
vars_reverse_integrals(lm::AtmosLinearModel, FT) = @vars()


function update_aux!(dg::DGModel, lm::AtmosLinearModel, Q::MPIStateArray, t::Real)
  return false
end
function flux_diffusive!(lm::AtmosLinearModel, flux::Grad, state::Vars,
                         diffusive::Vars, hyperdiffusive::Vars, aux::Vars, t::Real)
  nothing
end
integral_load_aux!(lm::AtmosLinearModel, integ::Vars, state::Vars, aux::Vars) = nothing
integral_set_aux!(lm::AtmosLinearModel, aux::Vars, integ::Vars) = nothing
reverse_integral_load_aux!(lm::AtmosLinearModel, integ::Vars, state::Vars, aux::Vars) = nothing
reverse_integral_set_aux!(lm::AtmosLinearModel, aux::Vars, integ::Vars) = nothing
flux_diffusive!(lm::AtmosLinearModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) = nothing
function wavespeed(lm::AtmosLinearModel, nM, state::Vars, aux::Vars, t::Real)
  ref = aux.ref_state
  return soundspeed_air(ref.T)
end

function boundary_state!(nf::NumericalFluxNonDiffusive, lm::AtmosLinearModel, x...)
  atmos_boundary_state!(nf, NoFluxBC(), lm.atmos, x...)
end
function boundary_state!(nf::NumericalFluxDiffusive, lm::AtmosLinearModel, x...)
  nothing
end
init_aux!(lm::AtmosLinearModel, aux::Vars, geom::LocalGeometry) = nothing
init_state!(lm::AtmosLinearModel, state::Vars, aux::Vars, coords, t) = nothing


struct AtmosAcousticLinearModel{M} <: AtmosLinearModel
  atmos::M
  function AtmosAcousticLinearModel(atmos::M) where {M}
    if atmos.ref_state === NoReferenceState()
      error("AtmosAcousticLinearModel needs a model with a reference state")
    end
    new{M}(atmos)
  end
end

function flux_nondiffusive!(lm::AtmosAcousticLinearModel, flux::Grad, state::Vars, aux::Vars, t::Real)
  FT = eltype(state)
  ref = aux.ref_state
  e_pot = gravitational_potential(lm.atmos.orientation, aux)

  flux.ρ = state.ρu
  pL = linearized_pressure(lm.atmos.moisture, lm.atmos.orientation, state, aux)
  flux.ρu += pL*I
  flux.ρe = ((ref.ρe + ref.p)/ref.ρ - e_pot)*state.ρu
  nothing
end
function source!(lm::AtmosAcousticLinearModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  nothing
end

struct AtmosAcousticGravityLinearModel{M} <: AtmosLinearModel
  atmos::M
  function AtmosAcousticGravityLinearModel(atmos::M) where {M}
    if atmos.ref_state === NoReferenceState()
      error("AtmosAcousticGravityLinearModel needs a model with a reference state")
    end
    new{M}(atmos)
  end
end
function flux_nondiffusive!(lm::AtmosAcousticGravityLinearModel, flux::Grad, state::Vars, aux::Vars, t::Real)
  FT = eltype(state)
  ref = aux.ref_state
  e_pot = gravitational_potential(lm.atmos.orientation, aux)

  flux.ρ = state.ρu
  pL = linearized_pressure(lm.atmos.moisture, lm.atmos.orientation, state, aux)
  flux.ρu += pL*I
  flux.ρe = ((ref.ρe + ref.p)/ref.ρ)*state.ρu
  nothing
end
function source!(lm::AtmosAcousticGravityLinearModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  ∇Φ = ∇gravitational_potential(lm.atmos.orientation, aux)
  source.ρu -= state.ρ * ∇Φ
  nothing
end
