T(x) = (dimension(x), typeof(upreferred(x)))
E_dim, E_units    = T(u"J")
D_dim, D_units    = T(u"kg/m^3")
T_dim, T_units    = T(u"K")
P_dim, P_units    = T(u"Pa")
EV_dim, EV_units  = T(u"J/m^3")
MF_dim, MF_units  = T(u"kg/m^2/s")
PE_dim, PE_units    = T(u"J/kg")
SHC_dim, SHC_units  = T(u"J/kg/K")

EQ{FT} = Quantity{FT, E_dim, E_units}
DQ{FT} = Quantity{FT, D_dim, D_units}
TQ{FT} = Quantity{FT, T_dim, T_units}
PQ{FT} = Quantity{FT, P_dim, P_units}
EVQ{FT} = Quantity{FT, EV_dim, EV_units}
MFQ{FT} = Quantity{FT, MF_dim, MF_units}
PEQ{FT} = Quantity{FT, PE_dim, PE_units}
SHCQ{FT} = Quantity{FT, SHC_dim, SHC_units}

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
function linearized_air_pressure(ρ::DQ{FT}, ρe_tot::EVQ{FT}, ρe_pot::EVQ{FT},
                                 ρq_tot::DQ{FT}=FT(0)*u"kg/m^3", ρq_liq::DQ{FT}=FT(0)*u"kg/m^3",
                                 ρq_ice::DQ{FT}=FT(0)*u"kg/m^3") where {FT<:Real}
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

vars_state(lm::AtmosLinearModel, FT) = vars_state(lm.atmos,FT)
vars_gradient(lm::AtmosLinearModel, FT) = @vars()
vars_diffusive(lm::AtmosLinearModel, FT) = @vars()
vars_aux(lm::AtmosLinearModel, FT) = vars_aux(lm.atmos,FT)
vars_integrals(lm::AtmosLinearModel, FT) = @vars()

space_unit(::AtmosLinearModel) = u"m"
time_unit(::AtmosLinearModel) = u"s"

update_aux!(dg::DGModel, lm::AtmosLinearModel, Q::MPIStateArray, t::Real) = nothing
integrate_aux!(lm::AtmosLinearModel, integ::Vars, state::Vars, aux::Vars) = nothing
flux_diffusive!(lm::AtmosLinearModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) = nothing
function wavespeed(lm::AtmosLinearModel, nM, state::Vars, aux::Vars, t::Real)
  ref = aux.ref_state
  return soundspeed_air(ref.T)
end

function boundary_state!(nf::Rusanov, lm::AtmosLinearModel, x...)
  atmos_boundary_state!(nf, NoFluxBC(), lm.atmos, x...)
end
function boundary_state!(nf::CentralNumericalFluxDiffusive, lm::AtmosLinearModel, x...)
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
function source!(lm::AtmosAcousticLinearModel, source::Vars, state::Vars, aux::Vars, t::Real)
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
function source!(lm::AtmosAcousticGravityLinearModel, source::Vars, state::Vars, aux::Vars, t::Real)
  ∇Φ = ∇gravitational_potential(lm.atmos.orientation, aux)
  source.ρu -= state.ρ * ∇Φ * time_unit(lm)
  nothing
end
