#### Interface kernels

function wrapper_flux_first_order!(
    balance_law::BalanceLaw,
    local_flux::AbstractArray{FT, 2},
    local_state_prognostic::AbstractArray{FT, 1},
    local_state_auxiliary::AbstractArray{FT, 1},
    t::AbstractFloat,
    direction,
) where {FT}
    vs_prognostic = vars_state(balance_law, Prognostic(), FT)
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    flux_first_order!(
        balance_law,
        Grad{vs_prognostic}(local_flux),
        Vars{vs_prognostic}(local_state_prognostic),
        Vars{vs_auxiliary}(local_state_auxiliary),
        t,
        direction,
    )
end

function wrapper_flux_second_order!(
    balance_law::BalanceLaw,
    local_flux::AbstractArray{FT, 2},
    local_state_prognostic::AbstractArray{FT, 1},
    local_state_gradient_flux::AbstractArray{FT, 1},
    local_state_hyperdiffusion::AbstractArray{FT, 1},
    local_state_auxiliary::AbstractArray{FT, 1},
    t::AbstractFloat,
) where {FT}
    vs_prognostic = vars_state(balance_law, Prognostic(), FT)
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    vs_grad_flux = vars_state(balance_law, GradientFlux(), FT)
    vs_hyperdiff = vars_state(balance_law, Hyperdiffusive(), FT)
    flux_second_order!(
        balance_law,
        Grad{vs_prognostic}(local_flux),
        Vars{vs_prognostic}(local_state_prognostic),
        Vars{vs_grad_flux}(local_state_gradient_flux),
        Vars{vs_hyperdiff}(local_state_hyperdiffusion),
        Vars{vs_auxiliary}(local_state_auxiliary),
        t,
    )

end

function wrapper_source!(
    balance_law::BalanceLaw,
    local_source::AbstractArray{FT, 1},
    local_state_prognostic::AbstractArray{FT, 1},
    local_state_gradient_flux::AbstractArray{FT, 1},
    local_state_auxiliary::AbstractArray{FT, 1},
    t::AbstractFloat,
    direction,
) where {FT}
    vs_prognostic = vars_state(balance_law, Prognostic(), FT)
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    vs_grad_flux = vars_state(balance_law, GradientFlux(), FT)

    source!(
        balance_law,
        Vars{vs_prognostic}(local_source),
        Vars{vs_prognostic}(local_state_prognostic),
        Vars{vs_grad_flux}(local_state_gradient_flux),
        Vars{vs_auxiliary}(local_state_auxiliary),
        t,
        direction,
    )

end

function wrapper_compute_gradient_argument!(
    balance_law::BalanceLaw,
    local_transform::AbstractArray{FT, 1},
    local_state_prognostic::AbstractArray{FT, 1},
    local_state_auxiliary::AbstractArray{FT, 1},
    t::AbstractFloat,
) where {FT}
    vs_prognostic = vars_state(balance_law, Prognostic(), FT)
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    vs_gradient = vars_state(balance_law, Gradient(), FT)
    compute_gradient_argument!(
        balance_law,
        Vars{vs_gradient}(local_transform),
        Vars{vs_prognostic}(local_state_prognostic),
        Vars{vs_auxiliary}(local_state_auxiliary),
        t,
    )

end

function wrapper_compute_gradient_flux!(
    balance_law::BalanceLaw,
    local_state_gradient_flux::AbstractArray{FT, 1},
    local_transform_gradient::AbstractArray{FT, 2},
    local_state_prognostic::AbstractArray{FT, 1},
    local_state_auxiliary::AbstractArray{FT, 1},
    t::AbstractFloat,
) where {FT}
    vs_prognostic = vars_state(balance_law, Prognostic(), FT)
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    vs_gradient = vars_state(balance_law, Gradient(), FT)
    vs_grad_flux = vars_state(balance_law, GradientFlux(), FT)
    compute_gradient_flux!(
        balance_law,
        Vars{vs_grad_flux}(local_state_gradient_flux),
        Grad{vs_gradient}(local_transform_gradient),
        Vars{vs_prognostic}(local_state_prognostic),
        Vars{vs_auxiliary}(local_state_auxiliary),
        t,
    )

end

function wrapper_transform_post_gradient_laplacian!(
    balance_law::BalanceLaw,
    local_state_hyperdiffusion::AbstractArray{FT, 1},
    l_grad_lap::AbstractArray{FT, 2},
    local_state_prognostic,
    local_state_auxiliary,
    t,
) where {FT}
    vs_prognostic = vars_state(balance_law, Prognostic(), FT)
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    vs_grad_lap = vars_state(balance_law, GradientLaplacian(), FT)
    vs_hyperdiff = vars_state(balance_law, Hyperdiffusive(), FT)
    transform_post_gradient_laplacian!(
        balance_law,
        Vars{vs_hyperdiff}(local_state_hyperdiffusion),
        Grad{vs_grad_lap}(l_grad_lap),
        Vars{vs_prognostic}(local_state_prognostic),
        Vars{vs_auxiliary}(local_state_auxiliary),
        t,
    )
end

function wrapper_init_state_prognostic!(
    balance_law::BalanceLaw,
    l_state::AbstractArray{FT, 1},
    local_state_auxiliary::AbstractArray{FT, 1},
    coords,
    args...,
) where {FT}
    vs_prognostic = vars_state(balance_law, Prognostic(), FT)
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    init_state_prognostic!(
        balance_law,
        Vars{vs_prognostic}(l_state),
        Vars{vs_auxiliary}(local_state_auxiliary),
        coords,
        args...,
    )
end

function wrapper_integral_load_auxiliary_state!(
    balance_law::BalanceLaw,
    local_kernel::AbstractArray{FT, 1},
    local_state_prognostic::AbstractArray{FT, 1},
    local_state_auxiliary::AbstractArray{FT, 1},
) where {FT}

    vs_up_integ = vars_state(balance_law, UpwardIntegrals(), FT)
    vs_prognostic = vars_state(balance_law, Prognostic(), FT)
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    integral_load_auxiliary_state!(
        balance_law,
        Vars{vs_up_integ}(local_kernel),
        Vars{vs_prognostic}(local_state_prognostic),
        Vars{vs_auxiliary}(local_state_auxiliary),
    )

end

function wrapper_integral_set_auxiliary_state!(
    balance_law::BalanceLaw,
    state_auxiliary::AbstractArray{FT, 1},
    local_kernel::AbstractArray{FT, 1},
) where {FT}
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    vs_up_integ = vars_state(balance_law, UpwardIntegrals(), FT)
    integral_set_auxiliary_state!(
        balance_law,
        Vars{vs_auxiliary}(state_auxiliary),
        Vars{vs_up_integ}(local_kernel),
    )

end

function wrapper_reverse_integral_load_auxiliary_state!(
    balance_law::BalanceLaw,
    l_V::AbstractArray{FT, 1},
    state::AbstractArray{FT, 1},
    state_auxiliary::AbstractArray{FT, 1},
) where {FT}

    vs_prognostic = vars_state(balance_law, Prognostic(), FT)
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    vs_dn_integ = vars_state(balance_law, DownwardIntegrals(), FT)
    reverse_integral_load_auxiliary_state!(
        balance_law,
        Vars{vs_dn_integ}(l_V),
        Vars{vs_prognostic}(state),
        Vars{vs_auxiliary}(state_auxiliary),
    )

end

function wrapper_reverse_integral_set_auxiliary_state!(
    balance_law::BalanceLaw,
    state_auxiliary::AbstractArray{FT, 1},
    l_V::AbstractArray{FT, 1},
) where {FT}
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    vs_dn_integ = vars_state(balance_law, DownwardIntegrals(), FT)
    reverse_integral_set_auxiliary_state!(
        balance_law,
        Vars{vs_auxiliary}(state_auxiliary),
        Vars{vs_dn_integ}(l_V),
    )

end
