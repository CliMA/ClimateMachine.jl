
"""
    WrapVars

A singleton used to indicate that the called
method will wrap arguments in `Vars`/`Grad`
types.
"""
struct WrapVars end

function init_state_prognostic!(
    ::WrapVars,
    balance_law,
    state::AbstractArray,
    aux::AbstractArray,
    local_geom,
    args...,
)
    FT = eltype(state)
    init_state_prognostic!(
        balance_law,
        Vars{vars_state(balance_law, Prognostic(), FT)}(state),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(aux),
        local_geom,
        args...,
    )

end

function source!(
    ::WrapVars,
    balance_law,
    source::AbstractArray,
    state::AbstractArray,
    diffusive::AbstractArray,
    aux::AbstractArray,
    t::Real,
    direction,
)
    FT = eltype(state)
    source!(
        balance_law,
        Vars{vars_state(balance_law, Prognostic(), FT)}(source),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state),
        Vars{vars_state(balance_law, GradientFlux(), FT)}(diffusive),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(aux),
        t,
        direction,
    )
end

function flux_first_order!(
    ::WrapVars,
    balance_law,
    flux::AbstractArray,
    state::AbstractArray,
    aux::AbstractArray,
    t::Real,
    direction,
)
    FT = eltype(state)
    flux_first_order!(
        balance_law,
        Grad{vars_state(balance_law, Prognostic(), FT)}(flux),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(aux),
        t,
        direction,
    )
end

function flux_second_order!(
    ::WrapVars,
    balance_law,
    flux::AbstractArray,
    state::AbstractArray,
    diffusive::AbstractArray,
    hyperdiffusive::AbstractArray,
    aux::AbstractArray,
    t::Real,
)
    FT = eltype(state)
    flux_second_order!(
        balance_law,
        Grad{vars_state(balance_law, Prognostic(), FT)}(flux),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state),
        Vars{vars_state(balance_law, GradientFlux(), FT)}(diffusive),
        Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(hyperdiffusive),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(aux),
        t,
    )
end

function compute_gradient_argument!(
    ::WrapVars,
    balance_law,
    transform::AbstractArray,
    state::AbstractArray,
    aux::AbstractArray,
    t::Real,
)
    FT = eltype(state)
    compute_gradient_argument!(
        balance_law,
        Vars{vars_state(balance_law, Gradient(), FT)}(transform),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(aux),
        t,
    )
end

function compute_gradient_flux!(
    ::WrapVars,
    balance_law,
    diffusive::AbstractArray,
    ∇transform::AbstractArray,
    state::AbstractArray,
    aux::AbstractArray,
    t::Real,
)

    FT = eltype(state)
    compute_gradient_flux!(
        balance_law,
        Vars{vars_state(balance_law, GradientFlux(), FT)}(diffusive),
        Grad{vars_state(balance_law, Gradient(), FT)}(∇transform),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(aux),
        t,
    )
end

function integral_load_auxiliary_state!(
    ::WrapVars,
    balance_law,
    local_kernel::AbstractArray,
    state_prognostic::AbstractArray,
    state_auxiliary::AbstractArray,
)
    FT = eltype(state_auxiliary)
    integral_load_auxiliary_state!(
        balance_law,
        Vars{vars_state(balance_law, UpwardIntegrals(), FT)}(local_kernel),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary),
    )
end

function integral_set_auxiliary_state!(
    ::WrapVars,
    balance_law,
    state_auxiliary::AbstractArray,
    local_kernel::AbstractArray,
)

    FT = eltype(state_auxiliary)
    integral_set_auxiliary_state!(
        balance_law,
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary),
        Vars{vars_state(balance_law, UpwardIntegrals(), FT)}(local_kernel),
    )
end

function reverse_integral_load_auxiliary_state!(
    ::WrapVars,
    balance_law,
    l_T::AbstractArray,
    state::AbstractArray,
    state_auxiliary::AbstractArray,
)

    FT = eltype(state_auxiliary)
    reverse_integral_load_auxiliary_state!(
        balance_law,
        Vars{vars_state(balance_law, DownwardIntegrals(), FT)}(l_T),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary),
    )

end

function reverse_integral_set_auxiliary_state!(
    ::WrapVars,
    balance_law,
    state_auxiliary::AbstractArray,
    l_V::AbstractArray,
)

    FT = eltype(state_auxiliary)
    reverse_integral_set_auxiliary_state!(
        balance_law,
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary),
        Vars{vars_state(balance_law, DownwardIntegrals(), FT)}(l_V),
    )
end

function transform_post_gradient_laplacian!(
    ::WrapVars,
    balance_law,
    hyperdiffusion::AbstractArray,
    l_grad_lap::AbstractArray,
    prognostic::AbstractArray,
    auxiliary::AbstractArray,
    t,
)

    FT = eltype(prognostic)
    transform_post_gradient_laplacian!(
        balance_law,
        Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(hyperdiffusion),
        Grad{vars_state(balance_law, GradientLaplacian(), FT)}(l_grad_lap),
        Vars{vars_state(balance_law, Prognostic(), FT)}(prognostic),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(auxiliary),
        t,
    )
end
