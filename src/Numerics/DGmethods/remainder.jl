export DGRemainderModel
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

struct DGRemainderModel{BL, G, NFND, NFD, GNF, AS, DS, HDS, D, DD, MD} <:
       AbstractDGModel
    balancelaw::BL
    grid::G
    numerical_flux_first_order::NFND
    numerical_flux_second_order::NFD
    numerical_flux_gradient::GNF
    state_auxiliary::AS
    state_gradient_flux::DS
    states_higher_order::HDS
    direction::D
    diffusion_direction::DD
    modeldata::MD
end

"""
    DGRemainderModel(
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


Constructs a `DGRemainderModel` from the `maindg` model and the tuple of
`subsdg` models. The concept of a remainder model is that it computes the
contribution of the main model after subtracting all of the subcomponents.

By default the numerical fluxes are set to be a tuple of the main and
subcomponent numerical fluxes. The main numerical flux is evaluated first and
then the subcomponent numerical fluxes are subtracted off. This is discretely
different (for the Rusanov / local Lax-Friedrichs flux) than defining a
numerical flux for the remainder of the physics model.

If the user sets the `numerical_flux_first_order` to a since
`NumericalFluxFirstOrder` type then the flux is used directly on the remainder
of the physics model.

The other parameters are set to the value in the `maindg` component, mainly the
data and arrays are aliased to the `maindg` values.
"""
function DGRemainderModel(
    maindg::DGModel,
    subsdg::NTuple{NumModels, DGModel};
    direction = EveryDirection(),
    numerical_flux_first_order = (
        maindg.numerical_flux_first_order,
        ntuple(i -> subsdg[i].numerical_flux_first_order, length(subsdg))...,
    ),
    numerical_flux_second_order = (
        maindg.numerical_flux_second_order,
        ntuple(i -> subsdg[i].numerical_flux_second_order, length(subsdg))...,
    ),
    numerical_flux_gradient = (
        maindg.numerical_flux_gradient,
        ntuple(i -> subsdg[i].numerical_flux_gradient, length(subsdg))...,
    ),
    state_auxiliary = maindg.state_auxiliary,
    state_gradient_flux = maindg.state_gradient_flux,
    states_higher_order = maindg.states_higher_order,
    diffusion_direction = maindg.diffusion_direction,
    modeldata = maindg.modeldata,
) where {NumModels}
    balancelaw = RemBL(
        maindg.balancelaw,
        ntuple(i -> subsdg[i].balancelaw, length(subsdg)),
    )
    DGRemainderModel(
        balancelaw,
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
vars_state_conservative(rem::RemBL, FT) = vars_state_conservative(rem.main, FT)

vars_state_gradient(rem::RemBL, FT) = vars_state_gradient(rem.main, FT)

vars_state_gradient_flux(rem::RemBL, FT) =
    vars_state_gradient_flux(rem.main, FT)

vars_state_auxiliary(rem::RemBL, FT) = vars_state_auxiliary(rem.main, FT)

vars_integrals(rem::RemBL, FT) = vars_integrals(rem.main, FT)

vars_reverse_integrals(rem::RemBL, FT) = vars_integrals(rem.main, FT)

vars_gradient_laplacian(rem::RemBL, FT) = vars_gradient_laplacian(rem.main, FT)

vars_hyperdiffusive(rem::RemBL, FT) = vars_hyperdiffusive(rem.main, FT)

update_auxiliary_state!(dg::DGRemainderModel, rem::RemBL, args...) =
    update_auxiliary_state!(dg, rem.main, args...)

update_auxiliary_state_gradient!(dg::DGRemainderModel, rem::RemBL, args...) =
    update_auxiliary_state_gradient!(dg, rem.main, args...)

integral_load_auxiliary_state!(rem::RemBL, args...) =
    integral_load_auxiliary_state!(rem.main, args...)

integral_set_auxiliary_state!(rem::RemBL, args...) =
    integral_set_auxiliary_state!(rem.main, args...)

reverse_integral_load_auxiliary_state!(rem::RemBL, args...) =
    reverse_integral_load_auxiliary_state!(rem.main, args...)

reverse_integral_set_auxiliary_state!(rem::RemBL, args...) =
    reverse_integral_set_auxiliary_state!(rem.main, args...)

transform_post_gradient_laplacian!(rem::RemBL, args...) =
    transform_post_gradient_laplacian!(rem.main, args...)

flux_second_order!(rem::RemBL, args...) = flux_second_order!(rem.main, args...)

compute_gradient_argument!(rem::RemBL, args...) =
    compute_gradient_argument!(rem.main, args...)

compute_gradient_flux!(rem::RemBL, args...) =
    compute_gradient_flux!(rem.main, args...)

boundary_state!(nf, rem::RemBL, args...) =
    boundary_state!(nf, rem.main, args...)

init_state_auxiliary!(rem::RemBL, args...) =
    init_state_auxiliary!(rem.main, args...)

init_state_conservative!(rem::RemBL, args...) =
    init_state_conservative!(rem.main, args...)

function wavespeed(rem::RemBL, nM, state::Vars, aux::Vars, t::Real)
    ref = aux.ref_state
    return wavespeed(rem.main, nM, state, aux, t) -
           sum(sub -> wavespeed(sub, nM, state, aux, t), rem.subs)
end

function flux_first_order!(
    rem::RemBL,
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
    rem::RemBL,
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

# Here the fluxes are pirated to handle the case of tuples of fluxes
import ..DGmethods.NumericalFluxes:
    NumericalFluxFirstOrder,
    numerical_flux_first_order!,
    numerical_boundary_flux_first_order!,
    CentralNumericalFluxSecondOrder,
    numerical_flux_second_order!,
    numerical_boundary_flux_second_order!,
    CentralNumericalFluxGradient,
    numerical_flux_gradient!,
    numerical_boundary_flux_gradient!,
    normal_boundary_flux_second_order!

"""
    function numerical_flux_first_order!(
        numerical_fluxes::NTuple{NumOfFluxes, NumericalFluxFirstOrder},
        rem_balance_law::RemBL,
        x...
    )

When the `numerical_fluxes` are a tuple and the balance law is a remainder
balance law the main components numerical flux is evaluated then all the
subcomponent numerical fluxes are evaluated and subtracted.
"""
function numerical_flux_first_order!(
    numerical_fluxes::NTuple{NumOfFluxes, NumericalFluxFirstOrder},
    rem_balance_law::RemBL,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_conservative⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_conservative⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    x...,
) where {NumOfFluxes, S, A}
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
    a_fluxᵀn = getfield(fluxᵀn, :array)
    sub_fluxᵀn = similar(fluxᵀn)
    a_sub_fluxᵀn = getfield(sub_fluxᵀn, :array)

    FT = eltype(a_sub_fluxᵀn)
    @unroll for k in 2:NumOfFluxes
        @inbounds sub = rem_balance_law.subs[k - 1]
        @inbounds nf = numerical_fluxes[k]
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
        numerical_fluxes::NTuple{NumOfFluxes, NumericalFluxFirstOrder},
        rem_balance_law::RemBL,
        x...
    )

When the `numerical_fluxes` are a tuple and the balance law is a remainder
balance law the main components numerical flux is evaluated then all the
subcomponent numerical fluxes are evaluated and subtracted.
"""
function numerical_boundary_flux_first_order!(
    numerical_fluxes::NTuple{NumOfFluxes, NumericalFluxFirstOrder},
    rem_balance_law::RemBL,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_conservative⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_conservative⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    x...,
) where {NumOfFluxes, S, A}
    # Since the fluxes are allowed to modified these we need backups so they can
    # be reset as we go
    a_state_conservative⁺ = getfield(state_conservative⁺, :array)
    a_state_auxiliary⁺ = getfield(state_auxiliary⁺, :array)

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
    a_fluxᵀn = getfield(fluxᵀn, :array)
    sub_fluxᵀn = similar(fluxᵀn)
    a_sub_fluxᵀn = getfield(sub_fluxᵀn, :array)

    FT = eltype(a_sub_fluxᵀn)
    @unroll for k in 2:NumOfFluxes
        @inbounds sub = rem_balance_law.subs[k - 1]
        @inbounds nf = numerical_fluxes[k]

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
    function numerical_flux_second_order!(
        numerical_fluxes::NTuple{NumOfFluxes, CentralNumericalFluxSecondOrder},
        rem_balance_law::RemBL,
        x...
    )

With the `CentralNumericalFluxSecondOrder` subtraction of the flux then
evaluation of the numerical flux or subtraction of the numerical fluxes is
discretely equivalent, thus in this case the former is done.
"""
function numerical_flux_second_order!(
    numerical_fluxes::NTuple{NumOfFluxes, CentralNumericalFluxSecondOrder},
    rem_balance_law::RemBL,
    x...,
) where {NumOfFluxes}
    numerical_flux_second_order!(
        CentralNumericalFluxSecondOrder(),
        rem_balance_law,
        x...,
    )
end

"""
    function numerical_boundary_flux_second_order!(
        numerical_fluxes::NTuple{NumOfFluxes, CentralNumericalFluxSecondOrder},
        rem_balance_law::RemBL,
        x...
    )

With the `CentralNumericalFluxSecondOrder` subtraction of the flux then
evaluation of the numerical flux or subtraction of the numerical fluxes is
discretely equivalent, thus in this case the former is done.
"""
function numerical_boundary_flux_second_order!(
    numerical_fluxes::NTuple{NumOfFluxes, CentralNumericalFluxSecondOrder},
    rem_balance_law::RemBL,
    x...,
) where {NumOfFluxes}
    numerical_boundary_flux_second_order!(
        CentralNumericalFluxSecondOrder(),
        rem_balance_law,
        x...,
    )
end


"""
    function numerical_flux_gradient!(
        numerical_fluxes::NTuple{NumOfFluxes, CentralNumericalFluxGradient},
        rem_balance_law::RemBL,
        x...
    )

With the `CentralNumericalFluxGradient` subtraction of the flux then evaluation
of the numerical flux or subtraction of the numerical fluxes is discretely
equivalent, thus in this case the former is done.
"""
function numerical_flux_gradient!(
    numerical_fluxes::NTuple{NumOfFluxes, CentralNumericalFluxGradient},
    rem_balance_law::RemBL,
    x...,
) where {NumOfFluxes}
    numerical_flux_gradient!(
        CentralNumericalFluxGradient(),
        rem_balance_law,
        x...,
    )
end

"""
    function numerical_boundary_flux_gradient!(
        numerical_fluxes::NTuple{NumOfFluxes, CentralNumericalFluxGradient},
        rem_balance_law::RemBL,
        x...
    )

With the `CentralNumericalFluxGradient` subtraction of the flux then evaluation
of the numerical flux or subtraction of the numerical fluxes is discretely
equivalent, thus in this case the former is done.
"""
function numerical_boundary_flux_gradient!(
    numerical_fluxes::NTuple{NumOfFluxes, CentralNumericalFluxGradient},
    rem_balance_law::RemBL,
    x...,
) where {NumOfFluxes}
    numerical_boundary_flux_gradient!(
        CentralNumericalFluxGradient(),
        rem_balance_law,
        x...,
    )
end

"""
    normal_boundary_flux_second_order!(nf, rem::RemBL, args...)

Currently the main models `normal_boundary_flux_second_order!` is called. If the
subcomponents models have second order terms this would need to be updated.
"""
normal_boundary_flux_second_order!(
    nf,
    rem::RemBL,
    fluxᵀn::Vars{S},
    args...,
) where {S} = normal_boundary_flux_second_order!(nf, rem.main, fluxᵀn, args...)
