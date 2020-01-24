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

update_aux!(dg::DGModel, rem::RemainderModel, Q::MPIStateArray, t::Real) =
  update_aux!(dg, rem.main, Q, t)

integrate_aux!(rem::RemainderModel, integ::Vars, state::Vars, aux::Vars) =
  integrate_aux!(rem.main, integ, state, aux)

flux_diffusive!(rem::RemainderModel, state::Vars, aux::Vars, t::Real, flux::Grad, diffusive::Vars) =
  flux_diffusive!(rem.main, state, aux, t, flux, diffusive)

gradvariables!(rem::RemainderModel, state::Vars, aux::Vars, t::Real, transform::Vars) =
  gradvariables!(rem.main, state, aux, t, transform)

diffusive!(rem::RemainderModel, state::Vars, aux::Vars, t::Real, diffusive::Vars, ∇transform::Grad) =
  diffusive!(rem.main, state, aux, t, diffusive, ∇transform)

function wavespeed(rem::RemainderModel, nM, state::Vars, aux::Vars, t::Real)
  ref = aux.ref_state
  return wavespeed(rem.main, nM, state, aux, t) - sum(sub -> wavespeed(sub, nM, state, aux, t), rem.subs)
end

boundary_state!(nf, rem::RemainderModel, x...) = boundary_state!(nf, rem.main, x...)

init_aux!(rem::RemainderModel, aux::Vars, geom::LocalGeometry) = nothing
init_state!(rem::RemainderModel, state::Vars, aux::Vars, coords, t) = nothing


function flux_nondiffusive!(rem::RemainderModel, state::Vars, aux::Vars, t::Real, flux::Grad)
  m = getfield(flux, :array)
  flux_nondiffusive!(rem.main, state, aux, t, flux)

  flux_s = similar(flux)
  m_s = getfield(flux_s, :array)

  for sub in rem.subs
    fill!(m_s, 0)
    flux_nondiffusive!(sub, state, aux, t, flux_s)
    m .-= m_s
  end
  nothing
end

function source!(rem::RemainderModel, state::Vars, aux::Vars, t::Real, source::Vars)
  m = getfield(source, :array)
  source!(rem.main, state, aux, t, source)

  source_s = similar(source)
  m_s = getfield(source_s, :array)

  for sub in rem.subs
    fill!(m_s, 0)
    source!(sub, state, aux, t, source_s)
    m .-= m_s
  end
  nothing
end
