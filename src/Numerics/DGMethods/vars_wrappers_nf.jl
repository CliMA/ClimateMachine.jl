#####
##### Vars wrappers
#####

# For the generic balance laws:
function numerical_flux_first_order!(
    ::WrapVars,
    numerical_flux::NumericalFluxFirstOrder,
    balance_law::BalanceLaw,
    fluxᵀn::AbstractArray,
    normal_vector::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
    direction,
)
    FT = eltype(state_prognostic⁺)
    vs_prog = Vars{vars_state(balance_law, Prognostic(), FT)}
    vs_aux = Vars{vars_state(balance_law, Auxiliary(), FT)}
    numerical_flux_first_order!(
        numerical_flux,
        balance_law,
        vs_prog(fluxᵀn),
        SVector(normal_vector),
        vs_prog(state_prognostic⁻),
        vs_aux(state_auxiliary⁻),
        vs_prog(state_prognostic⁺),
        vs_aux(state_auxiliary⁺),
        t,
        direction,
    )
end

# For the remainder model:
function numerical_flux_first_order!(
    ::WrapVars,
    numerical_fluxes::Tuple{
        NumericalFluxFirstOrder,
        NTuple{NumSubFluxes, NumericalFluxFirstOrder},
    },
    rem_balance_law,
    fluxᵀn::AbstractArray,
    normal_vector::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
    directions::Dirs,
) where {NumSubFluxes, S, A, Dirs <: NTuple{2, Direction}}

    FT = eltype(state_prognostic⁻)
    vs_prog = Vars{vars_state(rem_balance_law, Prognostic(), FT)}
    vs_aux = Vars{vars_state(rem_balance_law, Auxiliary(), FT)}

    numerical_flux_first_order!(
        numerical_fluxes,
        rem_balance_law,
        vs_prog(fluxᵀn),
        SVector(normal_vector),
        vs_prog(state_prognostic⁻),
        vs_aux(state_auxiliary⁻),
        vs_prog(state_prognostic⁺),
        vs_aux(state_auxiliary⁺),
        t,
        directions,
    )
end


function numerical_flux_second_order!(
    ::WrapVars,
    numerical_flux,
    balance_law::BalanceLaw,
    fluxᵀn::AbstractArray,
    normal_vector⁻::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_gradient_flux⁻::AbstractArray,
    state_hyperdiffusive⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_gradient_flux⁺::AbstractArray,
    state_hyperdiffusive⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
)

    FT = eltype(state_prognostic⁻)
    vs_prog = Vars{vars_state(balance_law, Prognostic(), FT)}
    vs_diff = Vars{vars_state(balance_law, GradientFlux(), FT)}
    vs_hyperdiff = Vars{vars_state(balance_law, Hyperdiffusive(), FT)}
    vs_aux = Vars{vars_state(balance_law, Auxiliary(), FT)}

    numerical_flux_second_order!(
        numerical_flux,
        balance_law,
        vs_prog(fluxᵀn),
        SVector(normal_vector⁻),
        vs_prog(state_prognostic⁻),
        vs_diff(state_gradient_flux⁻),
        vs_hyperdiff(state_hyperdiffusive⁻),
        vs_aux(state_auxiliary⁻),
        vs_prog(state_prognostic⁺),
        vs_diff(state_gradient_flux⁺),
        vs_hyperdiff(state_hyperdiffusive⁺),
        vs_aux(state_auxiliary⁺),
        t,
    )
end

# function numerical_flux_second_order!(
#     numerical_flux_second_order,
#     balance_law,
#     flux::AbstractArray,
#     normal_vector::AbstractArray,
#     state_prognostic⁻::AbstractArray,
#     state_gradient_flux⁻::AbstractArray,
#     state_hyperdiffusive⁻::AbstractArray,
#     state_auxiliary⁻::AbstractArray,
#     state_prognostic⁺::AbstractArray,
#     state_gradient_flux⁺::AbstractArray,
#     state_hyperdiffusive⁺::AbstractArray,
#     state_auxiliary⁺::AbstractArray,
#     t,
# )
#     FT = eltype(flux)
#     numerical_flux_second_order!(
#         numerical_flux_second_order,
#         balance_law,
#         Vars{vars_state(balance_law, Prognostic(), FT)}(flux),
#         SVector(normal_vector),
#         Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁻),
#         Vars{vars_state(balance_law, GradientFlux(), FT)}(state_gradient_flux⁻),
#         Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
#             state_hyperdiffusive⁻,
#         ),
#         Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁻),
#         Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁺),
#         Vars{vars_state(balance_law, GradientFlux(), FT)}(state_gradient_flux⁺),
#         Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
#             state_hyperdiffusive⁺,
#         ),
#         Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁺),
#         t,
#     )
# end

function update_penalty!(
    ::WrapVars,
    numerical_flux,
    balance_law,
    normal_vector,
    max_wavespeed,
    penalty::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
)
    FT = eltype(state_prognostic⁻)
    vs_prog = Vars{vars_state(balance_law, Prognostic(), FT)}
    vs_aux = Vars{vars_state(balance_law, Auxiliary(), FT)}

    update_penalty!(
        numerical_flux,
        balance_law,
        SVector(normal_vector),
        max_wavespeed,
        vs_prog(penalty),
        vs_prog(state_prognostic⁻),
        vs_aux(state_auxiliary⁻),
        vs_prog(state_prognostic⁺),
        vs_aux(state_auxiliary⁺),
        t,
    )
end

function numerical_flux_divergence!(
    ::WrapVars,
    numerical_flux,
    balance_law::BalanceLaw,
    div_penalty::AbstractArray,
    normal_vector::AbstractArray,
    grad⁻::AbstractArray,
    grad⁺::AbstractArray,
)
    FT = eltype(div_penalty)
    vs_div_pen = Vars{vars_state(balance_law, GradientLaplacian(), FT)}
    vs_grad_lap = Grad{vars_state(balance_law, GradientLaplacian(), FT)}

    numerical_flux_divergence!(
        numerical_flux,
        balance_law,
        vs_div_pen(div_penalty),
        SVector(normal_vector),
        vs_grad_lap(grad⁻),
        vs_grad_lap(grad⁺),
    )
end

function numerical_boundary_flux_first_order!(
    ::WrapVars,
    numerical_flux_first_order,
    bc,
    balance_law,
    local_flux::AbstractArray,
    normal_vector::AbstractArray,
    local_state_prognostic⁻::AbstractArray,
    local_state_auxiliary⁻::AbstractArray,
    local_state_prognostic⁺nondiff::AbstractArray,
    local_state_auxiliary⁺nondiff::AbstractArray,
    t,
    face_direction,
    local_state_prognostic_bottom1::AbstractArray,
    local_state_auxiliary_bottom1::AbstractArray,
)

    FT = eltype(local_flux)
    vs_prog = Vars{vars_state(balance_law, Prognostic(), FT)}
    vs_aux = Vars{vars_state(balance_law, Auxiliary(), FT)}

    numerical_boundary_flux_first_order!(
        numerical_flux_first_order,
        bc,
        balance_law,
        vs_prog(local_flux),
        SVector(normal_vector),
        vs_prog(local_state_prognostic⁻),
        vs_aux(local_state_auxiliary⁻),
        vs_prog(local_state_prognostic⁺nondiff),
        vs_aux(local_state_auxiliary⁺nondiff),
        t,
        face_direction,
        vs_prog(local_state_prognostic_bottom1),
        vs_aux(local_state_auxiliary_bottom1),
    )

end

function numerical_boundary_flux_second_order!(
    ::WrapVars,
    numerical_flux_second_order,
    bc,
    balance_law,
    local_flux::AbstractArray,
    normal_vector::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_gradient_flux⁻::AbstractArray,
    state_hyperdiffusion⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_prognostic⁺diff::AbstractArray,
    state_gradient_flux⁺::AbstractArray,
    state_hyperdiffusion⁺::AbstractArray,
    state_auxiliary⁺diff::AbstractArray,
    t,
    state_prognostic_bottom1::AbstractArray,
    state_gradient_flux_bottom1::AbstractArray,
    state_auxiliary_bottom1::AbstractArray,
)
    FT = eltype(local_flux)
    vs_prog = Vars{vars_state(balance_law, Prognostic(), FT)}
    vs_diff = Vars{vars_state(balance_law, GradientFlux(), FT)}
    vs_hyperdiff = Vars{vars_state(balance_law, Hyperdiffusive(), FT)}
    vs_aux = Vars{vars_state(balance_law, Auxiliary(), FT)}
    numerical_boundary_flux_second_order!(
        numerical_flux_second_order,
        bc,
        balance_law,
        vs_prog(local_flux),
        normal_vector,
        vs_prog(state_prognostic⁻),
        vs_diff(state_gradient_flux⁻),
        vs_hyperdiff(state_hyperdiffusion⁻),
        vs_aux(state_auxiliary⁻),
        vs_prog(state_prognostic⁺diff),
        vs_diff(state_gradient_flux⁺),
        vs_hyperdiff(state_hyperdiffusion⁺),
        vs_aux(state_auxiliary⁺diff),
        t,
        vs_prog(state_prognostic_bottom1),
        vs_diff(state_gradient_flux_bottom1),
        vs_aux(state_auxiliary_bottom1),
    )
end

function numerical_boundary_flux_gradient!(
    ::WrapVars,
    numerical_flux_gradient,
    bc,
    balance_law,
    transform_gradient::AbstractArray,
    normal_vector::AbstractArray,
    transform⁻::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    transform⁺::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
    state_prognostic_bottom1::AbstractArray,
    state_auxiliary_bottom1::AbstractArray,
)

    FT = eltype(transform_gradient)
    vs_prog = Vars{vars_state(balance_law, Prognostic(), FT)}
    vs_aux = Vars{vars_state(balance_law, Auxiliary(), FT)}
    vs_grad = Vars{vars_state(balance_law, Gradient(), FT)}

    numerical_boundary_flux_gradient!(
        numerical_flux_gradient,
        bc,
        balance_law,
        transform_gradient,
        SVector(normal_vector),
        vs_grad(transform⁻),
        vs_prog(state_prognostic⁻),
        vs_aux(state_auxiliary⁻),
        vs_grad(transform⁺),
        vs_prog(state_prognostic⁺),
        vs_aux(state_auxiliary⁺),
        t,
        vs_prog(state_prognostic_bottom1),
        vs_aux(state_auxiliary_bottom1),
    )

end

function numerical_boundary_flux_divergence!(
    ::WrapVars,
    numerical_flux,
    bctype,
    balance_law::BalanceLaw,
    div_penalty::AbstractArray,
    normal_vector::AbstractArray,
    grad⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    grad⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
)
    FT = eltype(div_penalty)
    vs_grad_lap = Vars{vars_state(balance_law, GradientLaplacian(), FT)}
    gs_grad_lap = Grad{vars_state(balance_law, GradientLaplacian(), FT)}
    vs_aux = Vars{vars_state(balance_law, Auxiliary(), FT)}
    numerical_boundary_flux_divergence!(
        numerical_flux,
        bctype,
        balance_law,
        vs_grad_lap(div_penalty),
        SVector(normal_vector),
        gs_grad_lap(grad⁻),
        vs_aux(state_auxiliary⁻),
        gs_grad_lap(grad⁺),
        vs_aux(state_auxiliary⁺),
        t,
    )
end

function numerical_flux_higher_order!(
    ::WrapVars,
    numerical_flux::GradNumericalFlux,
    balance_law::BalanceLaw,
    hyperdiff::AbstractArray,
    normal_vector::AbstractArray,
    lap⁻::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    lap⁺::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
)

    FT = eltype(state_prognostic⁻)
    vs_prog = Vars{vars_state(balance_law, Prognostic(), FT)}
    vs_aux = Vars{vars_state(balance_law, Auxiliary(), FT)}
    vs_hyperdiff = Vars{vars_state(balance_law, Hyperdiffusive(), FT)}
    vs_grad_lap = Vars{vars_state(balance_law, GradientLaplacian(), FT)}

    numerical_flux_higher_order!(
        numerical_flux,
        balance_law,
        vs_hyperdiff(hyperdiff),
        SVector(normal_vector),
        vs_grad_lap(lap⁻),
        vs_prog(state_prognostic⁻),
        vs_aux(state_auxiliary⁻),
        vs_grad_lap(lap⁺),
        vs_prog(state_prognostic⁺),
        vs_aux(state_auxiliary⁺),
        t,
    )
end

function numerical_boundary_flux_higher_order!(
    ::WrapVars,
    numerical_flux::GradNumericalFlux,
    bctype,
    balance_law::BalanceLaw,
    hyperdiff::AbstractArray,
    normal_vector::AbstractArray,
    lap⁻::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    lap⁺::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
)

    FT = eltype(hyperdiff)
    vs_prog = Vars{vars_state(balance_law, Prognostic(), FT)}
    vs_aux = Vars{vars_state(balance_law, Auxiliary(), FT)}
    vs_hyperdiff = Vars{vars_state(balance_law, Hyperdiffusive(), FT)}
    vs_grad_lap = Vars{vars_state(balance_law, GradientLaplacian(), FT)}

    numerical_boundary_flux_higher_order!(
        numerical_flux,
        bctype,
        balance_law,
        vs_hyperdiff(hyperdiff),
        SVector(normal_vector),
        vs_grad_lap(lap⁻),
        vs_prog(state_prognostic⁻),
        vs_aux(state_auxiliary⁻),
        vs_grad_lap(lap⁺),
        vs_prog(state_prognostic⁺),
        vs_aux(state_auxiliary⁺),
        t,
    )
end

#####
##### boundary_state!
#####

# TODO: develop a DG-unaware boundary_state! interface

function boundary_state!(
    ::WrapVars,
    numerical_flux,
    bctype,
    balance_law,
    state_prognostic⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    normal_vector::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    t,
    state1⁻::AbstractArray,
    aux1⁻::AbstractArray,
)
    FT = eltype(state_prognostic⁺)
    vs_prog = Vars{vars_state(balance_law, Prognostic(), FT)}
    vs_aux = Vars{vars_state(balance_law, Auxiliary(), FT)}
    boundary_state!(
        numerical_flux,
        bctype,
        balance_law,
        vs_prog(state_prognostic⁺),
        vs_aux(state_auxiliary⁺),
        SVector(normal_vector),
        vs_prog(state_prognostic⁻),
        vs_aux(state_auxiliary⁻),
        t,
        vs_prog(state1⁻),
        vs_aux(aux1⁻),
    )

end

function boundary_state!(
    ::WrapVars,
    numerical_flux::NumericalFluxSecondOrder,
    bctype,
    balance_law,
    state_prognostic⁺::AbstractArray,
    state_gradient_flux⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    normal_vector::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_gradient_flux⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    t,
    state1⁻::AbstractArray,
    diff1⁻::AbstractArray,
    aux1⁻::AbstractArray,
)

    FT = eltype(state_prognostic⁺)
    vs_prog = Vars{vars_state(balance_law, Prognostic(), FT)}
    vs_aux = Vars{vars_state(balance_law, Auxiliary(), FT)}
    vs_diff = Vars{vars_state(balance_law, GradientFlux(), FT)}

    boundary_state!(
        numerical_flux,
        bctype,
        balance_law,
        vs_prog(state_prognostic⁺),
        vs_diff(state_gradient_flux⁺),
        vs_aux(state_auxiliary⁺),
        SVector(normal_vector),
        vs_prog(state_prognostic⁻),
        vs_diff(state_gradient_flux⁻),
        vs_aux(state_auxiliary⁻),
        t,
        vs_prog(state1⁻),
        vs_diff(diff1⁻),
        vs_aux(aux1⁻),
    )
end

function boundary_state!(
    ::WrapVars,
    numerical_flux::Union{DivNumericalPenalty},
    bctype,
    balance_law,
    grad⁺,
    normal_vector,
    grad⁻,
)
    boundary_state!(
        numerical_flux,
        bctype,
        balance_law,
        grad⁺,
        normal_vector,
        grad⁻,
    )
end
