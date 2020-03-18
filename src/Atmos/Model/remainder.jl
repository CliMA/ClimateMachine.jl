"""
    RemainderModel(main::BalanceLaw, subcomponents::Tuple)

Compute the "remainder" contribution of the `main` model, after subtracting `subcomponents`.
"""
struct RemainderModel{M,S} <: BalanceLaw
  main::M
  subs::S
end

vars_state(rem::RemainderModel, FT) = vars_state(rem.main,FT)
vars_gradient(rem::RemainderModel, FT) = vars_gradient(rem.main,FT)
vars_diffusive(rem::RemainderModel, FT) = vars_diffusive(rem.main,FT)
vars_aux(rem::RemainderModel, FT) = vars_aux(rem.main,FT)
vars_integrals(rem::RemainderModel,FT) = vars_integrals(rem.main,FT)
vars_reverse_integrals(rem::RemainderModel,FT) = vars_integrals(rem.main,FT)

update_aux!(dg::DGModel, rem::RemainderModel, Q::MPIStateArray, t::Real) =
  update_aux!(dg, rem.main, Q, t)

integral_load_aux!(rem::RemainderModel, integ::Vars, state::Vars, aux::Vars) =
  integral_load_aux!(rem.main, integ, state, aux)

integral_set_aux!(rem::RemainderModel, aux::Vars, integ::Vars) =
  integral_set_aux!(rem.main, aux, integ)

reverse_integral_load_aux!(rem::RemainderModel, integ::Vars, state::Vars, aux::Vars) =
  reverse_integral_load_aux!(rem.main, integ, state, aux)

reverse_integral_set_aux!(rem::RemainderModel, aux::Vars, integ::Vars) =
  reverse_integral_set_aux!(rem.main, aux, integ)

function flux_diffusive!(rem::RemainderModel, flux::Grad, state::Vars,
                         diffusive::Vars, hyperdiffusive::Vars, aux::Vars, t::Real)
  flux_diffusive!(rem.main, flux, state, diffusive, hyperdiffusive, aux, t)
end

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

function source!(rem::RemainderModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  m = getfield(source, :array)
  source!(rem.main, source, state, diffusive, aux, t)

  source_s = similar(source)
  m_s = getfield(source_s, :array)

  for sub in rem.subs
    fill!(m_s, 0)
    source!(sub, source_s, state, diffusive, aux, t)
    m .-= m_s
  end
  nothing
end
