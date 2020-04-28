"""
    RemainderModel(main::BalanceLaw, subcomponents::Tuple)

Compute the "remainder" contribution of the `main` model, after subtracting `subcomponents`.
"""
struct RemainderModel{M, S} <: BalanceLaw
    main::M
    subs::S
end

vars_state_conservative(rem::RemainderModel, FT) =
    vars_state_conservative(rem.main, FT)
vars_state_gradient(rem::RemainderModel, FT) = vars_state_gradient(rem.main, FT)
vars_state_gradient_flux(rem::RemainderModel, FT) =
    vars_state_gradient_flux(rem.main, FT)
vars_state_auxiliary(rem::RemainderModel, FT) =
    vars_state_auxiliary(rem.main, FT)
vars_integrals(rem::RemainderModel, FT) = vars_integrals(rem.main, FT)
vars_reverse_integrals(rem::RemainderModel, FT) = vars_integrals(rem.main, FT)
vars_gradient_laplacian(rem::RemainderModel, FT) =
    vars_gradient_laplacian(rem.main, FT)
vars_hyperdiffusive(rem::RemainderModel, FT) = vars_hyperdiffusive(rem.main, FT)

update_auxiliary_state!(
    dg::DGModel,
    rem::RemainderModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
) = update_auxiliary_state!(dg, rem.main, Q, t, elems)

integral_load_auxiliary_state!(
    rem::RemainderModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) = integral_load_auxiliary_state!(rem.main, integ, state, aux)

integral_set_auxiliary_state!(rem::RemainderModel, aux::Vars, integ::Vars) =
    integral_set_auxiliary_state!(rem.main, aux, integ)

reverse_integral_load_auxiliary_state!(
    rem::RemainderModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) = reverse_integral_load_auxiliary_state!(rem.main, integ, state, aux)

reverse_integral_set_auxiliary_state!(
    rem::RemainderModel,
    aux::Vars,
    integ::Vars,
) = reverse_integral_set_auxiliary_state!(rem.main, aux, integ)

function transform_post_gradient_laplacian!(
    rem::RemainderModel,
    hyperdiffusive::Vars,
    hypertransform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform_post_gradient_laplacian!(
        rem.main,
        hyperdiffusive,
        hypertransform,
        state,
        aux,
        t,
    )
end
function flux_second_order!(
    rem::RemainderModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    flux_second_order!(rem.main, flux, state, diffusive, hyperdiffusive, aux, t)
end

compute_gradient_argument!(
    rem::RemainderModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
) = compute_gradient_argument!(rem.main, transform, state, aux, t)

compute_gradient_flux!(
    rem::RemainderModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) = compute_gradient_flux!(rem.main, diffusive, ∇transform, state, aux, t)

function wavespeed(rem::RemainderModel, nM, state::Vars, aux::Vars, t::Real)
    ref = aux.ref_state
    return wavespeed(rem.main, nM, state, aux, t) -
           sum(sub -> wavespeed(sub, nM, state, aux, t), rem.subs)
end

import .NumericalFluxes: normal_boundary_flux_second_order!
boundary_state!(nf, rem::RemainderModel, x...) =
    boundary_state!(nf, rem.main, x...)
function normal_boundary_flux_second_order!(
    nf,
    rem::RemainderModel,
    fluxᵀn::Vars{S},
    n⁻,
    state⁻,
    diff⁻,
    hyperdiff⁻,
    aux⁻,
    state⁺,
    diff⁺,
    hyperdiff⁺,
    aux⁺,
    bctype::Integer,
    t,
    args...,
) where {S}
    normal_boundary_flux_second_order!(
        nf,
        rem.main,
        fluxᵀn,
        n⁻,
        state⁻,
        diff⁻,
        hyperdiff⁻,
        aux⁻,
        state⁺,
        diff⁺,
        hyperdiff⁺,
        aux⁺,
        bctype,
        t,
        args...,
    )
end

init_state_auxiliary!(rem::RemainderModel, _...) = nothing
init_state_conservative!(rem::RemainderModel, _...) = nothing

function flux_first_order!(
    rem::RemainderModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    m = getfield(flux, :array)
    flux_first_order!(rem.main, flux, state, aux, t)

    flux_s = similar(flux)
    m_s = getfield(flux_s, :array)

    for sub in rem.subs
        fill!(m_s, 0)
        flux_first_order!(sub, flux_s, state, aux, t)
        m .-= m_s
    end
    nothing
end

function source!(
    rem::RemainderModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    m = getfield(source, :array)
    source!(rem.main, source, state, diffusive, aux, t, direction)

    source_s = similar(source)
    m_s = getfield(source_s, :array)

    for sub in rem.subs
        fill!(m_s, 0)
        source!(sub, source_s, state, diffusive, aux, t, direction)
        m .-= m_s
    end
    nothing
end
