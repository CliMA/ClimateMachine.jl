abstract type AtmosNonlinearModel <: BalanceLaw
end


vars_state(nlm::AtmosNonlinearModel, T) = vars_state(nlm.atmos,T)
vars_gradient(nlm::AtmosNonlinearModel, T) = vars_gradient(nlm.atmos,T)
vars_diffusive(nlm::AtmosNonlinearModel, T) = vars_diffusive(nlm.atmos,T)
vars_aux(nlm::AtmosNonlinearModel, T) = vars_aux(nlm.atmos,T)
vars_integrals(nlm::AtmosNonlinearModel,T) = vars_integrals(nlm.atmos,T)

update_aux!(dg::DGModel, nlm::AtmosNonlinearModel, Q::MPIStateArray, auxstate::MPIStateArray, t::Real) =
  update_aux!(dg, nlm.atmos, Q, auxstate, t)
  
integrate_aux!(nlm::AtmosNonlinearModel, integ::Vars, state::Vars, aux::Vars) =
  integrate_aux!(nlm.atmos, integ, state, aux)

flux_diffusive!(nlm::AtmosNonlinearModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) =
  flux_diffusive!(nlm.atmos, flux, state, diffusive, aux, t)

gradvariables!(nlm::AtmosNonlinearModel, transform::Vars, state::Vars, aux::Vars, t::Real) =
  gradvariables!(nlm.atmos, transform, state, aux, t)

diffusive!(nlm::AtmosNonlinearModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real) =
  diffusive!(nlm.atmos, diffusive, ∇transform, state, aux, t)

function wavespeed(nlm::AtmosNonlinearModel, nM, state::Vars, aux::Vars, t::Real)
  ref = aux.ref_state
  return wavespeed(nlm.atmos, nM, state, aux, t) - soundspeed_air(ref.T)
end

boundary_state!(nf, nlm::AtmosNonlinearModel, x...) = boundary_state!(nf, nlm.atmos, x...)

init_aux!(nlm::AtmosNonlinearModel, aux::Vars, geom::LocalGeometry) = nothing
init_state!(nlm::AtmosNonlinearModel, state::Vars, aux::Vars, coords, t) = nothing

struct AtmosAcousticNonlinearModel{M} <: AtmosNonlinearModel
  atmos::M
end
function flux_nondiffusive!(nlm::AtmosAcousticNonlinearModel, flux::Grad, state::Vars, aux::Vars, t::Real)
  FT = eltype(state)
  ρ = state.ρ
  ρu = state.ρu
  ρe = state.ρe
  ref = aux.ref_state
  u = ρu / ρ
  e_pot = gravitational_potential(nlm.atmos.orientation, aux)
  p = pressure(nlm.atmos.moisture, nlm.atmos.orientation, state, aux)
  # TODO: use MoistThermodynamics.linearized_air_pressure 
  # need to avoid dividing then multiplying by ρ
  pL = ρ * FT(R_d) * FT(T_0) + FT(R_d) / FT(cv_d) * (ρe - ρ * e_pot)

  flux.ρ = -zero(FT)
  flux.ρu = ρu .* u' + (p - pL) * I
  flux.ρe = ((ρe + p) / ρ - (ref.ρe + ref.p) / ref.ρ + e_pot) * ρu
end
function source!(nlm::AtmosAcousticNonlinearModel, source::Vars, state::Vars, aux::Vars, t::Real)
  source!(nlm.atmos, source, state, aux, t)
end

#TODO: AtmosAcousticGravityNonlinearModel

struct RemainderModel{M,S} <: BalanceLaw
  main::M
  subs::S
end

vars_state(rem::RemainderModel, T) = vars_state(rem.main,T)
vars_gradient(rem::RemainderModel, T) = vars_gradient(rem.main,T)
vars_diffusive(rem::RemainderModel, T) = vars_diffusive(rem.main,T)
vars_aux(rem::RemainderModel, T) = vars_aux(rem.main,T)
vars_integrals(rem::RemainderModel,T) = vars_integrals(rem.main,T)

update_aux!(dg::DGModel, rem::RemainderModel, Q::MPIStateArray, auxstate::MPIStateArray, t::Real) =
  update_aux!(dg, rem.main, Q, auxstate, t)

integrate_aux!(rem::RemainderModel, integ::Vars, state::Vars, aux::Vars) =
  integrate_aux!(rem.main, integ, state, aux)

flux_diffusive!(rem::RemainderModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) =
  flux_diffusive!(rem.main, flux, state, diffusive, aux, t)

gradvariables!(rem::RemainderModel, transform::Vars, state::Vars, aux::Vars, t::Real) =
  gradvariables!(rem.main, transform, state, aux, t)

diffusive!(rem::RemainderModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real) =
  diffusive!(rem.main, diffusive, ∇transform, state, aux, t)

function wavespeed(rem::RemainderModel, nM, state::Vars, aux::Vars, t::Real)
  ref = aux.ref_state
  return wavespeed(rem.main, nM, state, aux, t) - sum(sub -> wavespeed(sub, nM, state, aux, t), rem.subs)
end

boundary_state!(nf, rem::RemainderModel, x...) = boundary_state!(nf, rem.main, x...)

init_aux!(rem::RemainderModel, aux::Vars, geom::LocalGeometry) = nothing
init_state!(rem::RemainderModel, state::Vars, aux::Vars, coords, t) = nothing


function flux_nondiffusive!(rem::RemainderModel, flux::Grad, state::Vars, aux::Vars, t::Real)
  m = getfield(flux, :array)
  flux_nondiffusive!(rem.main, flux, state, aux, t)

  flux_s = similar(flux)
  m_s = getfield(flux_s, :array)

  for sub in rem.subs
    fill!(m_s, 0)
    flux_nondiffusive!(sub, flux_s, state, aux, t)
    m .-= m_s
  end
  nothing
end

function source!(rem::RemainderModel, source::Vars, state::Vars, aux::Vars, t::Real)
  m = getfield(source, :array)
  source!(rem.main, source, state, aux, t)

  source_s = similar(source)
  m_s = getfield(source_s, :array)

  for sub in rem.subs
    fill!(m_s, 0)
    source!(sub, source_s, state, aux, t)
    m .-= m_s
  end
  nothing
end
