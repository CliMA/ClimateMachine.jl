export remainder_DGModel
"""
    RemainderModel(main::BalanceLaw, subcomponents::Tuple)

Compute the "remainder" contribution of the `main` model, after subtracting
`subcomponents`.

Currently only the `flux_nondiffusive!` and `source!` are handled by the
remainder model
"""
struct RemainderModel{M, S} <: BalanceLaw
    main::M
    subs::S
end

function remainder_DGModel(
    balance_law,
    grid,
    numerical_flux_first_order,
    numerical_flux_second_order,
    numerical_flux_gradient;
    state_auxiliary = create_auxiliary_state(balance_law, grid),
    state_gradient_flux = create_gradient_state(balance_law, grid),
    states_higher_order = create_higher_order_states(balance_law, grid),
    direction = EveryDirection(),
    diffusion_direction = direction,
    modeldata = nothing,
)
    DGModel(
        balance_law,
        grid,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        state_auxiliary,
        state_gradient_flux,
        states_higher_order,
        direction,
        diffusion_direction,
        modeldata,
    )
end

# Inherit most of the functionality from the main model
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

update_auxiliary_state!(dg::DGModel, rem::RemainderModel, args...) =
    update_auxiliary_state!(dg, rem.main, args...)

update_auxiliary_state_gradient!(dg::DGModel, rem::RemainderModel, args...) =
    update_auxiliary_state_gradient!(dg, rem.main, args...)

integral_load_auxiliary_state!(rem::RemainderModel, args...) =
    integral_load_auxiliary_state!(rem.main, args...)

integral_set_auxiliary_state!(rem::RemainderModel, args...) =
    integral_set_auxiliary_state!(rem.main, args...)

reverse_integral_load_auxiliary_state!(rem::RemainderModel, args...) =
    reverse_integral_load_auxiliary_state!(rem.main, args...)

reverse_integral_set_auxiliary_state!(rem::RemainderModel, args...) =
    reverse_integral_set_auxiliary_state!(rem.main, args...)

transform_post_gradient_laplacian!(rem::RemainderModel, args...) =
    transform_post_gradient_laplacian!(rem.main, args...)

flux_second_order!(rem::RemainderModel, args...) =
    flux_second_order!(rem.main, args...)

compute_gradient_argument!(rem::RemainderModel, args...) =
    compute_gradient_argument!(rem.main, args...)

compute_gradient_flux!(rem::RemainderModel, args...) =
    compute_gradient_flux!(rem.main, args...)

boundary_state!(nf, rem::RemainderModel, args...) =
    boundary_state!(nf, rem.main, args...)

init_state_auxiliary!(rem::RemainderModel, args...) =
    init_state_auxiliary!(rem.main, args...)

init_state_conservative!(rem::RemainderModel, args...) =
    init_state_conservative!(rem.main, args...)

function wavespeed(rem::RemainderModel, nM, state::Vars, aux::Vars, t::Real)
    ref = aux.ref_state
    return wavespeed(rem.main, nM, state, aux, t) -
           sum(sub -> wavespeed(sub, nM, state, aux, t), rem.subs)
end

import .NumericalFluxes: normal_boundary_flux_second_order!
boundary_state!(nf, rem::RemainderModel, x...) =
    boundary_state!(nf, rem.main, x...)

normal_boundary_flux_second_order!(
    nf,
    rem::RemainderModel,
    fluxᵀn::Vars{S},
    args...,
) where {S} = normal_boundary_flux_second_order!(nf, rem.main, fluxᵀn, args...)

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
