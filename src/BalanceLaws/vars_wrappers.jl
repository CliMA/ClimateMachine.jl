
function source_arr!(
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

function flux_first_order_arr!(
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

function flux_second_order_arr!(
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

function compute_gradient_argument_arr!(
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

function compute_gradient_flux_arr!(
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
