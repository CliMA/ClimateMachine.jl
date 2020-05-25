using StaticNumbers
export remainder_DGModel

"""
    RemBL(main::BalanceLaw, subcomponents::Tuple)

Compute the "remainder" contribution of the `main` model, after subtracting
`subcomponents`.

Currently only the `flux_nondiffusive!` and `source!` are handled by the
remainder model
"""
struct RemBL{M, S} <: BalanceLaw
    main::M
    subs::S
end


"""
    remainder_DGModel(
        maindg::DGModel,
        subsdg::NTuple{NumModels, DGModel};
        direction = EveryDirection(),
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        state_auxiliary,
        state_gradient_flux,
        states_higher_order,
        diffusion_direction,
        modeldata,
    )


Constructs a `DGModel` from the `maindg` model and the tuple of
`subsdg` models. The concept of a remainder model is that it computes the
contribution of the  model after subtracting all of the subcomponents.

By default the numerical fluxes are set to be a tuple of the main models
numerical flux and the splitting is done at the PDE level (e.g., the remainder
model is calculated prior to discretization). If instead a tuple of numerical
fluxes is passed in the main numerical flux is evaluated first and then the
subcomponent numerical fluxes are subtracted off. This is discretely different
(for the Rusanov / local Lax-Friedrichs flux) than defining a numerical flux for
the remainder of the physics model.

The other parameters are set to the value in the `maindg` component, mainly the
data and arrays are aliased to the `maindg` values.
"""
function remainder_DGModel(
    maindg::DGModel,
    subsdg::NTuple{NumModels, DGModel};
    direction = EveryDirection(),
    numerical_flux_first_order = maindg.numerical_flux_first_order,
    numerical_flux_second_order = maindg.numerical_flux_second_order,
    numerical_flux_gradient = maindg.numerical_flux_gradient,
    state_auxiliary = maindg.state_auxiliary,
    state_gradient_flux = maindg.state_gradient_flux,
    states_higher_order = maindg.states_higher_order,
    diffusion_direction = maindg.diffusion_direction,
    modeldata = maindg.modeldata,
) where {NumModels}
    balance_law = RemBL(
        maindg.balance_law,
        ntuple(i -> subsdg[i].balance_law, length(subsdg)),
    )
    FT = eltype(state_auxiliary)

    # If any of these asserts fail, the remainder model will need to be extended
    # to allow for it; see `flux_first_order!` and `source!` below.
    for subdg in subsdg
        @assert number_state_conservative(subdg.balance_law, FT) <=
                number_state_conservative(maindg.balance_law, FT)

        @assert number_state_auxiliary(subdg.balance_law, FT) ==
                number_state_auxiliary(maindg.balance_law, FT)

        @assert number_state_gradient(subdg.balance_law, FT) == 0
        @assert number_state_gradient_flux(subdg.balance_law, FT) == 0

        @assert num_gradient_laplacian(subdg.balance_law, FT) == 0
        @assert num_hyperdiffusive(subdg.balance_law, FT) == 0

        @assert num_integrals(subdg.balance_law, FT) == 0
        @assert num_reverse_integrals(subdg.balance_law, FT) == 0
    end


    DGModel(
        balance_law,
        maindg.grid,
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
vars_state_conservative(rem_balance_law::RemBL, FT) =
    vars_state_conservative(rem_balance_law.main, FT)

vars_state_gradient(rem_balance_law::RemBL, FT) =
    vars_state_gradient(rem_balance_law.main, FT)

vars_state_gradient_flux(rem_balance_law::RemBL, FT) =
    vars_state_gradient_flux(rem_balance_law.main, FT)

vars_state_auxiliary(rem_balance_law::RemBL, FT) =
    vars_state_auxiliary(rem_balance_law.main, FT)

vars_integrals(rem_balance_law::RemBL, FT) =
    vars_integrals(rem_balance_law.main, FT)

vars_reverse_integrals(rem_balance_law::RemBL, FT) =
    vars_integrals(rem_balance_law.main, FT)

vars_gradient_laplacian(rem_balance_law::RemBL, FT) =
    vars_gradient_laplacian(rem_balance_law.main, FT)

vars_hyperdiffusive(rem_balance_law::RemBL, FT) =
    vars_hyperdiffusive(rem_balance_law.main, FT)

update_auxiliary_state!(dg::DGModel, rem_balance_law::RemBL, args...) =
    update_auxiliary_state!(dg, rem_balance_law.main, args...)

update_auxiliary_state_gradient!(dg::DGModel, rem_balance_law::RemBL, args...) =
    update_auxiliary_state_gradient!(dg, rem_balance_law.main, args...)

integral_load_auxiliary_state!(rem_balance_law::RemBL, args...) =
    integral_load_auxiliary_state!(rem_balance_law.main, args...)

integral_set_auxiliary_state!(rem_balance_law::RemBL, args...) =
    integral_set_auxiliary_state!(rem_balance_law.main, args...)

reverse_integral_load_auxiliary_state!(rem_balance_law::RemBL, args...) =
    reverse_integral_load_auxiliary_state!(rem_balance_law.main, args...)

reverse_integral_set_auxiliary_state!(rem_balance_law::RemBL, args...) =
    reverse_integral_set_auxiliary_state!(rem_balance_law.main, args...)

transform_post_gradient_laplacian!(rem_balance_law::RemBL, args...) =
    transform_post_gradient_laplacian!(rem_balance_law.main, args...)

flux_second_order!(rem_balance_law::RemBL, args...) =
    flux_second_order!(rem_balance_law.main, args...)

compute_gradient_argument!(rem_balance_law::RemBL, args...) =
    compute_gradient_argument!(rem_balance_law.main, args...)

compute_gradient_flux!(rem_balance_law::RemBL, args...) =
    compute_gradient_flux!(rem_balance_law.main, args...)

boundary_state!(nf, rem_balance_law::RemBL, args...) =
    boundary_state!(nf, rem_balance_law.main, args...)

init_state_auxiliary!(rem_balance_law::RemBL, args...) =
    init_state_auxiliary!(rem_balance_law.main, args...)

init_state_conservative!(rem_balance_law::RemBL, args...) =
    init_state_conservative!(rem_balance_law.main, args...)

"""
    function flux_first_order!(
        rem_balance_law::RemBL,
        flux::Grad,
        state::Vars,
        aux::Vars,
        t::Real,
        directions,
    )

Evaluate the remainder flux by first evaluating the main flux and subtracting
the subcomponent fluxes.

Only models which have directions that are included in the `directions` tuple
are evaluated. When these models are evaluated the models underlying `direction`
is passed (not the original `directions` argument).
"""
function flux_first_order!(
    rem_balance_law::RemBL,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    m = parent(flux)
    flux_first_order!(rem_balance_law.main, flux, state, aux, t)

    flux_s = similar(flux)
    m_s = parent(flux_s)

    for sub in rem_balance_law.subs
        fill!(m_s, 0)
        flux_first_order!(sub, flux_s, state, aux, t)
        m .-= m_s
    end
    nothing
end


"""
    function source!(
        rem_balance_law::RemBL,
        source::Vars,
        state::Vars,
        diffusive::Vars,
        aux::Vars,
        t::Real,
        directions,
    )

Evaluate the remainder source by first evaluating the main source and subtracting
the subcomponent sources.

Only models which have directions that are included in the `directions` tuple
are evaluated. When these models are evaluated the models underlying `direction`
is passed (not the original `directions` argument).
"""
function source!(
    rem_balance_law::RemBL,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    m = parent(source)
    source!(rem_balance_law.main, source, state, diffusive, aux, t, direction)

    source_s = similar(source)
    m_s = parent(source_s)

    for sub in rem_balance_law.subs
        fill!(m_s, 0)
        source!(sub, source_s, state, diffusive, aux, t, direction)
        m .-= m_s
    end
    nothing
end

"""
    function wavespeed(
        rem_balance_law::RemBL,
        args...,
    )

The wavespeed for a remainder model is defined to be the difference of the wavespeed
of the main model and the sum of the subcomponents.

Note: Defining the wavespeed in this manner can result in a smaller value than
the actually wavespeed of the remainder physics model depending on the
composition of the models.
"""
function wavespeed(rem::RemBL, nM, state::Vars, aux::Vars, t::Real)
    FT = eltype(state)

    ws = fill(0, MVector{number_state_conservative(rem.main, FT), FT})
    rs = fill(0, MVector{number_state_conservative(rem.main, FT), FT})

    ws .= wavespeed(rem.main, nM, state, aux, t)

    for sub in rem.subs
        num_state = static(number_state_conservative(sub, Float32))
        @inbounds rs[static(1):num_state] .+= wavespeed(sub, nM, state, aux, t)
    end

    ws .-= rs

    return ws
end

# Here the fluxes are pirated to handle the case of tuples of fluxes
import ..DGMethods.NumericalFluxes:
    NumericalFluxFirstOrder,
    numerical_flux_first_order!,
    numerical_boundary_flux_first_order!,
    normal_boundary_flux_second_order!

"""
    function numerical_flux_first_order!(
        numerical_fluxes::Tuple{
            NumericalFluxFirstOrder,
            NTuple{NumSubFluxes, NumericalFluxFirstOrder},
        },
        rem_balance_law::RemBL,
        fluxᵀn::Vars{S},
        normal_vector::SVector,
        state_conservative⁻::Vars{S},
        state_auxiliary⁻::Vars{A},
        state_conservative⁺::Vars{S},
        state_auxiliary⁺::Vars{A},
        t,
        directions,
    )

When the `numerical_fluxes` are a tuple and the balance law is a remainder
balance law the main components numerical flux is evaluated then all the
subcomponent numerical fluxes are evaluated and subtracted.

Only models which have directions that are included in the `directions` tuple
are evaluated. When these models are evaluated the models underlying `direction`
is passed (not the original `directions` argument).
"""
function numerical_flux_first_order!(
    numerical_fluxes::Tuple{
        NumericalFluxFirstOrder,
        NTuple{NumSubFluxes, NumericalFluxFirstOrder},
    },
    rem_balance_law::RemBL,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_conservative⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_conservative⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    x...,
) where {NumSubFluxes, S, A}
    # Call the numerical flux for the main model
    @inbounds numerical_flux_first_order!(
        numerical_fluxes[1],
        rem_balance_law.main,
        fluxᵀn,
        normal_vector,
        state_conservative⁻,
        state_auxiliary⁻,
        state_conservative⁺,
        state_auxiliary⁺,
        x...,
    )

    # Create put the sub model fluxes
    a_fluxᵀn = parent(fluxᵀn)
    sub_fluxᵀn = similar(fluxᵀn)
    a_sub_fluxᵀn = parent(sub_fluxᵀn)

    FT = eltype(a_sub_fluxᵀn)
    @unroll for k in 1:NumSubFluxes
        @inbounds sub = rem_balance_law.subs[k]
        @inbounds nf = numerical_fluxes[2][k]
        # compute this submodels flux
        fill!(a_sub_fluxᵀn, -zero(FT))
        numerical_flux_first_order!(
            nf,
            sub,
            sub_fluxᵀn,
            normal_vector,
            state_conservative⁻,
            state_auxiliary⁻,
            state_conservative⁺,
            state_auxiliary⁺,
            x...,
        )

        # Subtract off this sub models flux
        a_fluxᵀn .-= a_sub_fluxᵀn
    end
end

"""
    function numerical_boundary_flux_first_order!(
        numerical_fluxes::Tuple{
            NumericalFluxFirstOrder,
            NTuple{NumSubFluxes, NumericalFluxFirstOrder},
        },
        rem_balance_law::RemBL,
        fluxᵀn::Vars{S},
        normal_vector::SVector,
        state_conservative⁻::Vars{S},
        state_auxiliary⁻::Vars{A},
        state_conservative⁺::Vars{S},
        state_auxiliary⁺::Vars{A},
        bctype,
        t,
        directions,
        args...,
    )

When the `numerical_fluxes` are a tuple and the balance law is a remainder
balance law the main components numerical flux is evaluated then all the
subcomponent numerical fluxes are evaluated and subtracted.

Only models which have directions that are included in the `directions` tuple
are evaluated. When these models are evaluated the models underlying `direction`
is passed (not the original `directions` argument).
"""
function numerical_boundary_flux_first_order!(
    numerical_fluxes::Tuple{
        NumericalFluxFirstOrder,
        NTuple{NumSubFluxes, NumericalFluxFirstOrder},
    },
    rem_balance_law::RemBL,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_conservative⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_conservative⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    x...,
) where {NumSubFluxes, S, A}
    # Since the fluxes are allowed to modified these we need backups so they can
    # be reset as we go
    a_state_conservative⁺ = parent(state_conservative⁺)
    a_state_auxiliary⁺ = parent(state_auxiliary⁺)

    a_back_state_conservative⁺ = copy(a_state_conservative⁺)
    a_back_state_auxiliary⁺ = copy(a_state_auxiliary⁺)


    # Call the numerical flux for the main model
    @inbounds numerical_boundary_flux_first_order!(
        numerical_fluxes[1],
        rem_balance_law.main,
        fluxᵀn,
        normal_vector,
        state_conservative⁻,
        state_auxiliary⁻,
        state_conservative⁺,
        state_auxiliary⁺,
        x...,
    )

    # Create put the sub model fluxes
    a_fluxᵀn = parent(fluxᵀn)
    sub_fluxᵀn = similar(fluxᵀn)
    a_sub_fluxᵀn = parent(sub_fluxᵀn)

    FT = eltype(a_sub_fluxᵀn)
    @unroll for k in 1:NumSubFluxes
        @inbounds sub = rem_balance_law.subs[k]
        @inbounds nf = numerical_fluxes[2][k]

        # reset the plus-side data
        a_state_conservative⁺ .= a_back_state_conservative⁺
        a_state_auxiliary⁺ .= a_back_state_auxiliary⁺

        # compute this submodels flux
        fill!(a_sub_fluxᵀn, -zero(FT))
        numerical_boundary_flux_first_order!(
            nf,
            sub,
            sub_fluxᵀn,
            normal_vector,
            state_conservative⁻,
            state_auxiliary⁻,
            state_conservative⁺,
            state_auxiliary⁺,
            x...,
        )

        # Subtract off this sub models flux
        a_fluxᵀn .-= a_sub_fluxᵀn
    end
end

"""
    normal_boundary_flux_second_order!(nf, rem_balance_law::RemBL, args...)

Currently the main models `normal_boundary_flux_second_order!` is called. If the
subcomponents models have second order terms this would need to be updated.
"""
normal_boundary_flux_second_order!(
    nf,
    rem_balance_law::RemBL,
    fluxᵀn::Vars{S},
    args...,
) where {S} = normal_boundary_flux_second_order!(
    nf,
    rem_balance_law.main,
    fluxᵀn,
    args...,
)
