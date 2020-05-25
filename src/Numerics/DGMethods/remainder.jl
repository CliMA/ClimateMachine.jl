using StaticNumbers
export remainder_DGModel

import ..BalanceLaws:
    vars_state_conservative,
    vars_state_auxiliary,
    vars_state_gradient,
    vars_state_gradient_flux,
    vars_integrals,
    vars_reverse_integrals,
    vars_gradient_laplacian,
    vars_hyperdiffusive,
    init_state_conservative!,
    init_state_auxiliary!,
    flux_first_order!,
    flux_second_order!,
    compute_gradient_flux!,
    compute_gradient_argument!,
    source!,
    transform_post_gradient_laplacian!,
    wavespeed,
    boundary_state!,
    update_auxiliary_state!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!

"""
    RemBL(
        main::BalanceLaw,
        subcomponents::Tuple,
        maindir::Direction,
        subsdir::Tuple,
    )

Balance law for holding remainder model information. Direction is put here since
direction is so intertwined with the DGModel_kernels, that it is easier to hande
this in this container.
"""
struct RemBL{M, S, MD, SD} <: BalanceLaw
    main::M
    subs::S
    maindir::MD
    subsdir::SD
end

"""
    rembl_has_subs_direction(
        direction::Direction,
        rem_balance_law::RemBL,
    )

Query whether the `rem_balance_law` has any subcomponent balance laws operating
in the direction `direction`
"""
@generated function rembl_has_subs_direction(
    ::Dir,
    ::RemBL{MainBL, SubsBL, MainDir, SubsDir},
) where {Dir <: Direction, MainBL, SubsBL, MainDir, SubsDir <: Tuple}
    if Dir in SubsDir.types
        return :(true)
    else
        return :(false)
    end
end


"""
    remainder_DGModel(
        maindg::DGModel,
        subsdg::NTuple{NumModels, DGModel};
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        state_auxiliary,
        state_gradient_flux,
        states_higher_order,
        diffusion_direction,
        modeldata,
    )


Constructs a `DGModel` from the `maindg` model and the tuple of `subsdg` models.
The concept of a remainder model is that it computes the contribution of the
model after subtracting all of the subcomponents.

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
    numerical_flux_first_order = maindg.numerical_flux_first_order,
    numerical_flux_second_order = maindg.numerical_flux_second_order,
    numerical_flux_gradient = maindg.numerical_flux_gradient,
    state_auxiliary = maindg.state_auxiliary,
    state_gradient_flux = maindg.state_gradient_flux,
    states_higher_order = maindg.states_higher_order,
    diffusion_direction = maindg.diffusion_direction,
    modeldata = maindg.modeldata,
) where {NumModels}
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

        # Do not currenlty support nested remainder models
        # For this to work the way directions and numerical fluxes are handled
        # would need to be updated.
        @assert !(subdg.balance_law isa RemBL)

        @assert num_integrals(subdg.balance_law, FT) == 0
        @assert num_reverse_integrals(subdg.balance_law, FT) == 0

        # The remainder model requires that the subcomponent direction be
        # included in the main model directions
        @assert (
            maindg.direction isa EveryDirection ||
            maindg.direction === subdg.direction
        )
    end
    balance_law = RemBL(
        maindg.balance_law,
        ntuple(i -> subsdg[i].balance_law, length(subsdg)),
        maindg.direction,
        ntuple(i -> subsdg[i].direction, length(subsdg)),
    )


    DGModel(
        balance_law,
        maindg.grid,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        state_auxiliary,
        state_gradient_flux,
        states_higher_order,
        maindg.direction,
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
    ::Dirs,
) where {NumDirs, Dirs <: NTuple{NumDirs, Direction}}
    m = parent(flux)
    if rem_balance_law.maindir isa Union{Dirs.types...}
        flux_first_order!(
            rem_balance_law.main,
            flux,
            state,
            aux,
            t,
            (rem_balance_law.maindir,),
        )
    end

    flux_s = similar(flux)
    m_s = parent(flux_s)

    # Force the loop to unroll to get type stability on the GPU
    @inbounds ntuple(Val(length(rem_balance_law.subs))) do k
        Base.@_inline_meta
        if rem_balance_law.subsdir[k] isa Union{Dirs.types...}
            sub = rem_balance_law.subs[k]
            fill!(m_s, -zero(eltype(m_s)))
            flux_first_order!(
                sub,
                flux_s,
                state,
                aux,
                t,
                (rem_balance_law.subsdir[k],),
            )
            m .-= m_s
        end
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
    ::Dirs,
) where {NumDirs, Dirs <: NTuple{NumDirs, Direction}}
    m = parent(source)
    if EveryDirection() isa Union{Dirs.types...} ||
       rem_balance_law.maindir isa EveryDirection ||
       rem_balance_law.maindir isa Union{Dirs.types...}
        source!(
            rem_balance_law.main,
            source,
            state,
            diffusive,
            aux,
            t,
            (rem_balance_law.maindir,),
        )
    end

    source_s = similar(source)
    m_s = parent(source_s)

    # Force the loop to unroll to get type stability on the GPU
    ntuple(Val(length(rem_balance_law.subs))) do k
        Base.@_inline_meta
        @inbounds if EveryDirection() isa Union{Dirs.types...} ||
                     rem_balance_law.subsdir[k] isa EveryDirection ||
                     rem_balance_law.subsdir[k] isa Union{Dirs.types...}
            sub = rem_balance_law.subs[k]
            fill!(m_s, -zero(eltype(m_s)))
            source!(
                sub,
                source_s,
                state,
                diffusive,
                aux,
                t,
                (rem_balance_law.subsdir[k],),
            )
            m .-= m_s
        end
    end
    nothing
end

"""
    function wavespeed(
        rem_balance_law::RemBL,
        args...,
    )

The wavespeed for a remainder model is defined to be the difference of the
wavespeed of the main model and the sum of the subcomponents.

Note: Defining the wavespeed in this manner can result in a smaller value than
the actually wavespeed of the remainder physics model depending on the
composition of the models.
"""
function wavespeed(
    rem_balance_law::RemBL,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
    dir::Dirs,
) where {ND, Dirs <: NTuple{ND, Direction}}
    FT = eltype(state)

    ws = fill(
        -zero(FT),
        MVector{number_state_conservative(rem_balance_law.main, FT), FT},
    )
    rs = fill(
        -zero(FT),
        MVector{number_state_conservative(rem_balance_law.main, FT), FT},
    )

    # Compute the main components wavespeed
    if rem_balance_law.maindir isa Union{Dirs.types...}
        ws .= wavespeed(
            rem_balance_law.main,
            nM,
            state,
            aux,
            t,
            (rem_balance_law.maindir,),
        )
    end

    # Compute the sub components wavespeed
    for (sub, subdir) in zip(rem_balance_law.subs, rem_balance_law.subsdir)
        @inbounds if subdir isa Union{Dirs.types...}
            num_state = static(number_state_conservative(sub, Float32))
            rs[static(1):num_state] .+=
                wavespeed(sub, nM, state, aux, t, (subdir,))
        end
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
    t,
    ::Dirs,
) where {NumSubFluxes, S, A, Dirs <: NTuple{2, Direction}}
    # Call the numerical flux for the main model
    if rem_balance_law.maindir isa Union{Dirs.types...}
        @inbounds numerical_flux_first_order!(
            numerical_fluxes[1],
            rem_balance_law.main,
            fluxᵀn,
            normal_vector,
            state_conservative⁻,
            state_auxiliary⁻,
            state_conservative⁺,
            state_auxiliary⁺,
            t,
            (rem_balance_law.maindir,),
        )
    end

    # Create put the sub model fluxes
    a_fluxᵀn = parent(fluxᵀn)

    # Force the loop to unroll to get type stability on the GPU
    ntuple(Val(length(rem_balance_law.subs))) do k
        Base.@_inline_meta
        @inbounds if rem_balance_law.subsdir[k] isa Union{Dirs.types...}
            sub = rem_balance_law.subs[k]
            nf = numerical_fluxes[2][k]

            FT = eltype(a_fluxᵀn)
            num_state_conservative = number_state_conservative(sub, FT)

            a_sub_fluxᵀn = MVector{num_state_conservative, FT}(undef)
            a_sub_state_conservative⁻ =
                MVector{num_state_conservative, FT}(undef)
            a_sub_state_conservative⁺ =
                MVector{num_state_conservative, FT}(undef)

            state_rng = static(1):static(number_state_conservative(sub, FT))
            a_sub_fluxᵀn .= a_fluxᵀn[state_rng]
            a_sub_state_conservative⁻ .= parent(state_conservative⁻)[state_rng]
            a_sub_state_conservative⁺ .= parent(state_conservative⁺)[state_rng]

            # compute this submodels flux
            fill!(a_sub_fluxᵀn, -zero(eltype(a_sub_fluxᵀn)))
            numerical_flux_first_order!(
                nf,
                sub,
                Vars{vars_state_conservative(sub, FT)}(a_sub_fluxᵀn),
                normal_vector,
                Vars{vars_state_conservative(sub, FT)}(
                    a_sub_state_conservative⁻,
                ),
                state_auxiliary⁻,
                Vars{vars_state_conservative(sub, FT)}(
                    a_sub_state_conservative⁺,
                ),
                state_auxiliary⁺,
                t,
                (rem_balance_law.subsdir[k],),
            )

            # Subtract off this sub models flux
            a_fluxᵀn[state_rng] .-= a_sub_fluxᵀn
        end
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
    bctype,
    t,
    ::Dirs,
    state_conservative1⁻::Vars{S},
    state_auxiliary1⁻::Vars{A},
) where {NumSubFluxes, S, A, Dirs <: NTuple{2, Direction}}
    # Since the fluxes are allowed to modified these we need backups so they can
    # be reset as we go
    a_state_conservative⁺ = parent(state_conservative⁺)
    a_state_auxiliary⁺ = parent(state_auxiliary⁺)

    a_back_state_conservative⁺ = copy(a_state_conservative⁺)
    a_back_state_auxiliary⁺ = copy(a_state_auxiliary⁺)


    # Call the numerical flux for the main model
    if rem_balance_law.maindir isa Union{Dirs.types...}
        @inbounds numerical_boundary_flux_first_order!(
            numerical_fluxes[1],
            rem_balance_law.main,
            fluxᵀn,
            normal_vector,
            state_conservative⁻,
            state_auxiliary⁻,
            state_conservative⁺,
            state_auxiliary⁺,
            bctype,
            t,
            (rem_balance_law.maindir,),
            state_conservative1⁻,
            state_auxiliary1⁻,
        )
    end

    # Create put the sub model fluxes
    a_fluxᵀn = parent(fluxᵀn)

    # Force the loop to unroll to get type stability on the GPU
    ntuple(Val(length(rem_balance_law.subs))) do k
        Base.@_inline_meta
        @inbounds if rem_balance_law.subsdir[k] isa Union{Dirs.types...}
            sub = rem_balance_law.subs[k]
            nf = numerical_fluxes[2][k]

            # reset the plus-side data
            a_state_conservative⁺ .= a_back_state_conservative⁺
            a_state_auxiliary⁺ .= a_back_state_auxiliary⁺

            FT = eltype(a_fluxᵀn)
            num_state_conservative = number_state_conservative(sub, FT)

            a_sub_fluxᵀn = MVector{num_state_conservative, FT}(undef)
            a_sub_state_conservative⁻ =
                MVector{num_state_conservative, FT}(undef)
            a_sub_state_conservative⁺ =
                MVector{num_state_conservative, FT}(undef)
            a_sub_state_conservative1⁻ =
                MVector{num_state_conservative, FT}(undef)

            state_rng = static(1):static(number_state_conservative(sub, FT))
            a_sub_fluxᵀn .= a_fluxᵀn[state_rng]
            a_sub_state_conservative⁻ .= parent(state_conservative⁻)[state_rng]
            a_sub_state_conservative⁺ .= parent(state_conservative⁺)[state_rng]
            a_sub_state_conservative1⁻ .=
                parent(state_conservative1⁻)[state_rng]


            # compute this submodels flux
            fill!(a_sub_fluxᵀn, -zero(eltype(a_sub_fluxᵀn)))
            numerical_boundary_flux_first_order!(
                nf,
                sub,
                Vars{vars_state_conservative(sub, FT)}(a_sub_fluxᵀn),
                normal_vector,
                Vars{vars_state_conservative(sub, FT)}(
                    a_sub_state_conservative⁻,
                ),
                state_auxiliary⁻,
                Vars{vars_state_conservative(sub, FT)}(
                    a_sub_state_conservative⁺,
                ),
                state_auxiliary⁺,
                bctype,
                t,
                (rem_balance_law.subsdir[k],),
                Vars{vars_state_conservative(sub, FT)}(
                    a_sub_state_conservative1⁻,
                ),
                state_auxiliary1⁻,
            )

            # Subtract off this sub models flux
            a_fluxᵀn[state_rng] .-= a_sub_fluxᵀn
        end
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
