######
###### Numerical fluxes
######

function wrapper_numerical_flux_gradient!(
    numerical_flux_gradient::NumericalFluxGradient,
    balance_law::BalanceLaw,
    local_transform_gradient::AbstractArray{FT, 2},
    normal_vector::AbstractArray{FT, 1},
    local_transform⁻::AbstractArray{FT, 1},
    local_state_prognostic⁻::AbstractArray{FT, 1},
    local_state_auxiliary⁻::AbstractArray{FT, 1},
    local_transform⁺::AbstractArray{FT, 1},
    local_state_prognostic⁺::AbstractArray{FT, 1},
    local_state_auxiliary⁺::AbstractArray{FT, 1},
    t,
) where {FT}

    vs_prognostic = vars_state(balance_law, Prognostic(), FT)
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    vs_gradient = vars_state(balance_law, Gradient(), FT)
    numerical_flux_gradient!(
        numerical_flux_gradient,
        balance_law,
        local_transform_gradient,
        normal_vector,
        Vars{vs_gradient}(local_transform⁻),
        Vars{vs_prognostic}(local_state_prognostic⁻),
        Vars{vs_auxiliary}(local_state_auxiliary⁻),
        Vars{vs_gradient}(local_transform⁺),
        Vars{vs_prognostic}(local_state_prognostic⁺),
        Vars{vs_auxiliary}(local_state_auxiliary⁺),
        t,
    )
end

function wrapper_numerical_flux_first_order!(
    numerical_flux_first_order::NumericalFluxFirstOrder,
    balance_law::BalanceLaw,
    local_flux::AbstractArray{FT, 1},
    normal_vector::AbstractArray{FT, 1},
    local_state_prognostic⁻::AbstractArray{FT, 1},
    local_state_auxiliary⁻::AbstractArray{FT, 1},
    local_state_prognostic⁺nondiff::AbstractArray{FT, 1},
    local_state_auxiliary⁺nondiff::AbstractArray{FT, 1},
    t,
    face_direction,
) where {FT}
    vs_prognostic = vars_state(balance_law, Prognostic(), FT)
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    vs_grad_flux = vars_state(balance_law, GradientFlux(), FT)

    numerical_flux_first_order!(
        numerical_flux_first_order,
        balance_law,
        Vars{vs_prognostic}(local_flux),
        normal_vector,
        Vars{vs_prognostic}(local_state_prognostic⁻),
        Vars{vs_auxiliary}(local_state_auxiliary⁻),
        Vars{vs_prognostic}(local_state_prognostic⁺nondiff),
        Vars{vs_auxiliary}(local_state_auxiliary⁺nondiff),
        t,
        face_direction,
    )

end

function wrapper_numerical_flux_second_order!(
    numerical_flux_second_order::NumericalFluxSecondOrder,
    balance_law::BalanceLaw,
    local_flux::AbstractArray{FT, 1},
    normal_vector::AbstractArray{FT, 1},
    local_state_prognostic⁻::AbstractArray{FT, 1},
    local_state_gradient_flux⁻::AbstractArray{FT, 1},
    local_state_hyperdiffusion⁻::AbstractArray{FT, 1},
    local_state_auxiliary⁻::AbstractArray{FT, 1},
    local_state_prognostic⁺diff::AbstractArray{FT, 1},
    local_state_gradient_flux⁺::AbstractArray{FT, 1},
    local_state_hyperdiffusion⁺::AbstractArray{FT, 1},
    local_state_auxiliary⁺diff::AbstractArray{FT, 1},
    t,
) where {FT}

    vs_prognostic = vars_state(balance_law, Prognostic(), FT)
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    vs_grad_flux = vars_state(balance_law, GradientFlux(), FT)
    vs_hyperdiff = vars_state(balance_law, Hyperdiffusive(), FT)

    numerical_flux_second_order!(
        numerical_flux_second_order,
        balance_law,
        Vars{vs_prognostic}(local_flux),
        normal_vector,
        Vars{vs_prognostic}(local_state_prognostic⁻),
        Vars{vs_grad_flux}(local_state_gradient_flux⁻),
        Vars{vs_hyperdiff}(local_state_hyperdiffusion⁻),
        Vars{vs_auxiliary}(local_state_auxiliary⁻),
        Vars{vs_prognostic}(local_state_prognostic⁺diff),
        Vars{vs_grad_flux}(local_state_gradient_flux⁺),
        Vars{vs_hyperdiff}(local_state_hyperdiffusion⁺),
        Vars{vs_auxiliary}(local_state_auxiliary⁺diff),
        t,
    )
end

function wrapper_numerical_flux_divergence!(
    divgradnumpenalty::DivNumericalPenalty,
    balance_law::BalanceLaw,
    l_div::AbstractArray{FT, 1},
    normal_vector::AbstractArray{FT, 1},
    l_grad⁻::AbstractArray{FT, 2},
    l_grad⁺::AbstractArray{FT, 2},
) where {FT}
    vs_grad_lap = vars_state(balance_law, GradientLaplacian(), FT)
    numerical_flux_divergence!(
        divgradnumpenalty,
        balance_law,
        Vars{vs_grad_lap}(l_div),
        normal_vector,
        Grad{vs_grad_lap}(l_grad⁻),
        Grad{vs_grad_lap}(l_grad⁺),
    )

end

function wrapper_numerical_flux_higher_order!(
    hyperviscnumflux::GradNumericalFlux,
    balance_law::BalanceLaw,
    local_state_hyperdiffusion::AbstractArray{FT, 1},
    normal_vector::AbstractArray{FT, 1},
    l_lap⁻::AbstractArray{FT, 1},
    local_state_prognostic⁻::AbstractArray{FT, 1},
    local_state_auxiliary⁻::AbstractArray{FT, 1},
    l_lap⁺::AbstractArray{FT, 1},
    local_state_prognostic⁺::AbstractArray{FT, 1},
    local_state_auxiliary⁺::AbstractArray{FT, 1},
    t,
) where {FT}
    vs_prognostic = vars_state(balance_law, Prognostic(), FT)
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    vs_grad_lap = vars_state(balance_law, GradientLaplacian(), FT)
    vs_hyperdiff = vars_state(balance_law, Hyperdiffusive(), FT)
    numerical_flux_higher_order!(
        hyperviscnumflux,
        balance_law,
        Vars{vs_hyperdiff}(local_state_hyperdiffusion),
        normal_vector,
        Vars{vs_grad_lap}(l_lap⁻),
        Vars{vs_prognostic}(local_state_prognostic⁻),
        Vars{vs_auxiliary}(local_state_auxiliary⁻),
        Vars{vs_grad_lap}(l_lap⁺),
        Vars{vs_prognostic}(local_state_prognostic⁺),
        Vars{vs_auxiliary}(local_state_auxiliary⁺),
        t,
    )
end

######
###### Numerical boundary fluxes
######

function wrapper_numerical_boundary_flux_gradient!(
    numerical_flux_gradient::NumericalFluxGradient,
    balance_law::BalanceLaw,
    local_transform_gradient,
    normal_vector::AbstractArray{FT, 1},
    local_transform⁻::AbstractArray{FT, 1},
    local_state_prognostic⁻::AbstractArray{FT, 1},
    local_state_auxiliary⁻::AbstractArray{FT, 1},
    local_transform⁺::AbstractArray{FT, 1},
    local_state_prognostic⁺::AbstractArray{FT, 1},
    local_state_auxiliary⁺::AbstractArray{FT, 1},
    bctype::Int,
    t,
    local_state_prognostic_bottom1::AbstractArray{FT, 1},
    local_state_auxiliary_bottom1::AbstractArray{FT, 1},
) where {FT}

    vs_prognostic = vars_state(balance_law, Prognostic(), FT)
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    vs_gradient = vars_state(balance_law, Gradient(), FT)

    numerical_boundary_flux_gradient!(
        numerical_flux_gradient,
        balance_law,
        local_transform_gradient,
        normal_vector,
        Vars{vs_gradient}(local_transform⁻),
        Vars{vs_prognostic}(local_state_prognostic⁻),
        Vars{vs_auxiliary}(local_state_auxiliary⁻),
        Vars{vs_gradient}(local_transform⁺),
        Vars{vs_prognostic}(local_state_prognostic⁺),
        Vars{vs_auxiliary}(local_state_auxiliary⁺),
        bctype,
        t,
        Vars{vs_prognostic}(local_state_prognostic_bottom1),
        Vars{vs_auxiliary}(local_state_auxiliary_bottom1),
    )

end

function wrapper_numerical_boundary_flux_first_order!(
    numerical_flux_first_order::NumericalFluxFirstOrder,
    balance_law::BalanceLaw,
    local_flux::AbstractArray{FT, 1},
    normal_vector::AbstractArray{FT, 1},
    local_state_prognostic⁻::AbstractArray{FT, 1},
    local_state_auxiliary⁻::AbstractArray{FT, 1},
    local_state_prognostic⁺nondiff::AbstractArray{FT, 1},
    local_state_auxiliary⁺nondiff::AbstractArray{FT, 1},
    bctype::Int,
    t,
    face_direction,
    local_state_prognostic_bottom1::AbstractArray{FT, 1},
    local_state_auxiliary_bottom1::AbstractArray{FT, 1},
) where {FT}
    vs_prognostic = vars_state(balance_law, Prognostic(), FT)
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)

    numerical_boundary_flux_first_order!(
        numerical_flux_first_order,
        balance_law,
        Vars{vs_prognostic}(local_flux),
        normal_vector,
        Vars{vs_prognostic}(local_state_prognostic⁻),
        Vars{vs_auxiliary}(local_state_auxiliary⁻),
        Vars{vs_prognostic}(local_state_prognostic⁺nondiff),
        Vars{vs_auxiliary}(local_state_auxiliary⁺nondiff),
        bctype,
        t,
        face_direction,
        Vars{vs_prognostic}(local_state_prognostic_bottom1),
        Vars{vs_auxiliary}(local_state_auxiliary_bottom1),
    )
end

function wrapper_numerical_boundary_flux_second_order!(
    numerical_flux_second_order::NumericalFluxSecondOrder,
    balance_law::BalanceLaw,
    local_flux::AbstractArray{FT, 1},
    normal_vector::AbstractArray{FT, 1},
    local_state_prognostic⁻::AbstractArray{FT, 1},
    local_state_gradient_flux⁻::AbstractArray{FT, 1},
    local_state_hyperdiffusion⁻::AbstractArray{FT, 1},
    local_state_auxiliary⁻::AbstractArray{FT, 1},
    local_state_prognostic⁺diff::AbstractArray{FT, 1},
    local_state_gradient_flux⁺::AbstractArray{FT, 1},
    local_state_hyperdiffusion⁺::AbstractArray{FT, 1},
    local_state_auxiliary⁺diff::AbstractArray{FT, 1},
    bctype::Int,
    t,
    local_state_prognostic_bottom1::AbstractArray{FT, 1},
    local_state_gradient_flux_bottom1::AbstractArray{FT, 1},
    local_state_auxiliary_bottom1::AbstractArray{FT, 1},
) where {FT}

    vs_prognostic = vars_state(balance_law, Prognostic(), FT)
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    vs_grad_flux = vars_state(balance_law, GradientFlux(), FT)
    vs_hyperdiff = vars_state(balance_law, Hyperdiffusive(), FT)
    numerical_boundary_flux_second_order!(
        numerical_flux_second_order,
        balance_law,
        Vars{vs_prognostic}(local_flux),
        normal_vector,
        Vars{vs_prognostic}(local_state_prognostic⁻),
        Vars{vs_grad_flux}(local_state_gradient_flux⁻),
        Vars{vs_hyperdiff}(local_state_hyperdiffusion⁻),
        Vars{vs_auxiliary}(local_state_auxiliary⁻),
        Vars{vs_prognostic}(local_state_prognostic⁺diff),
        Vars{vs_grad_flux}(local_state_gradient_flux⁺),
        Vars{vs_hyperdiff}(local_state_hyperdiffusion⁺),
        Vars{vs_auxiliary}(local_state_auxiliary⁺diff),
        bctype,
        t,
        Vars{vs_prognostic}(local_state_prognostic_bottom1),
        Vars{vs_grad_flux}(local_state_gradient_flux_bottom1),
        Vars{vs_auxiliary}(local_state_auxiliary_bottom1),
    )
end

function wrapper_numerical_boundary_flux_divergence!(
    divgradnumpenalty::DivNumericalPenalty,
    balance_law::BalanceLaw,
    l_div::AbstractArray{FT, 1},
    normal_vector::AbstractArray{FT, 1},
    l_grad⁻::AbstractArray{FT, 2},
    l_grad⁺::AbstractArray{FT, 2},
    bctype::Int,
) where {FT}
    vs_grad_lap = vars_state(balance_law, GradientLaplacian(), FT)
    numerical_boundary_flux_divergence!(
        divgradnumpenalty,
        balance_law,
        Vars{vs_grad_lap}(l_div),
        normal_vector,
        Grad{vs_grad_lap}(l_grad⁻),
        Grad{vs_grad_lap}(l_grad⁺),
        bctype,
    )
end

function wrapper_numerical_boundary_flux_higher_order!(
    hyperviscnumflux::GradNumericalFlux,
    balance_law::BalanceLaw,
    local_state_hyperdiffusion::AbstractArray{FT, 1},
    normal_vector::AbstractArray{FT, 1},
    l_lap⁻::AbstractArray{FT, 1},
    local_state_prognostic⁻::AbstractArray{FT, 1},
    local_state_auxiliary⁻::AbstractArray{FT, 1},
    l_lap⁺::AbstractArray{FT, 1},
    local_state_prognostic⁺::AbstractArray{FT, 1},
    local_state_auxiliary⁺::AbstractArray{FT, 1},
    bctype::Int,
    t,
) where {FT}
    vs_prognostic = vars_state(balance_law, Prognostic(), FT)
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    vs_grad_lap = vars_state(balance_law, GradientLaplacian(), FT)
    vs_hyperdiff = vars_state(balance_law, Hyperdiffusive(), FT)
    numerical_boundary_flux_higher_order!(
        hyperviscnumflux,
        balance_law,
        Vars{vs_hyperdiff}(local_state_hyperdiffusion),
        normal_vector,
        Vars{vs_grad_lap}(l_lap⁻),
        Vars{vs_prognostic}(local_state_prognostic⁻),
        Vars{vs_auxiliary}(local_state_auxiliary⁻),
        Vars{vs_grad_lap}(l_lap⁺),
        Vars{vs_prognostic}(local_state_prognostic⁺),
        Vars{vs_auxiliary}(local_state_auxiliary⁺),
        bctype,
        t,
    )

end

function wrapper_update_penalty!(
    numerical_flux::NumericalFluxFirstOrder,
    balance_law::BalanceLaw,
    normal_vector::AbstractArray{FT, 1},
    max_wavespeed,
    penalty::AbstractArray{FT},
    state_prognostic⁻::AbstractArray{FT, 1},
    state_auxiliary⁻::AbstractArray{FT, 1},
    state_prognostic⁺::AbstractArray{FT, 1},
    state_auxiliary⁺::AbstractArray{FT, 1},
    t,
) where {FT}
    vs_prognostic = vars_state(balance_law, Prognostic(), FT)
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    update_penalty!(
        numerical_flux,
        balance_law,
        normal_vector,
        max_wavespeed,
        Vars{vs_prognostic}(penalty),
        Vars{vs_prognostic}(state_prognostic⁻),
        Vars{vs_auxiliary}(state_auxiliary⁻),
        Vars{vs_prognostic}(state_prognostic⁺),
        Vars{vs_auxiliary}(state_auxiliary⁺),
        t,
    )

end

function wrapper_boundary_state!(
    numerical_flux,
    balance_law::BalanceLaw,
    state_prognostic⁺::AbstractArray{FT, 1},
    state_auxiliary⁺::AbstractArray{FT, 1},
    normal_vector::AbstractArray{FT, 1},
    state_prognostic⁻::AbstractArray{FT, 1},
    state_auxiliary⁻::AbstractArray{FT, 1},
    bctype::Int,
    t,
    state1⁻::AbstractArray{FT, 1},
    aux1⁻::AbstractArray{FT, 1},
) where {FT}
    vs_prognostic = vars_state(balance_law, Prognostic(), FT)
    vs_auxiliary = vars_state(balance_law, Auxiliary(), FT)
    boundary_state!(
        numerical_flux,
        balance_law,
        Vars{vs_prognostic}(state_prognostic⁺),
        Vars{vs_auxiliary}(state_auxiliary⁺),
        normal_vector,
        Vars{vs_prognostic}(state_prognostic⁻),
        Vars{vs_auxiliary}(state_auxiliary⁻),
        bctype,
        t,
        Vars{vs_prognostic}(state1⁻),
        Vars{vs_auxiliary}(aux1⁻),
    )
end
