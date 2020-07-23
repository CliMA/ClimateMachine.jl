@inline function initialize_states!(
    baroclinic::HBModel{C},
    barotropic::SWModel{C},
    model_bc,
    model_bt,
    state_bc,
    state_bt,
) where {C <: Coupled}
    model_bc.state_auxiliary.ΔGᵘ .= -0

    return nothing
end

@inline function tendency_from_slow_to_fast!(
    baroclinic::HBModel{C},
    barotropic::SWModel{C},
    model_bc,
    model_bt,
    state_bc,
    state_bt,
    forcing_tendency,
) where {C <: Coupled}
    FT = eltype(state_bc)
    Nq, Nqk, _, _, nelemv, nelemh, nrealelemh, _ = basic_grid_info(model_bc)

    #### integrate the tendency
    model_int = model_bc.modeldata.integral_model
    integral = model_int.balance_law
    update_auxiliary_state!(model_int, integral, forcing_tendency, 0)

    ### properly shape MPIStateArrays
    num_aux_int = number_states(integral, Auxiliary(), FT)
    data_int = model_int.state_auxiliary.data
    data_int = reshape(data_int, Nq^2, Nqk, num_aux_int, nelemv, nelemh)

    num_aux_bt = number_states(barotropic, Auxiliary(), FT)
    data_bt = model_bt.state_auxiliary.data
    data_bt = reshape(data_bt, Nq^2, num_aux_bt, nelemh)

    num_aux_bc = number_states(baroclinic, Auxiliary(), FT)
    data_bc = model_bc.state_auxiliary.data
    data_bc = reshape(data_bc, Nq^2, Nqk, num_aux_bc, nelemv, nelemh)

    ### get vars indices
    index_∫du = varsindex(vars_state(integral, Auxiliary(), FT), :(∫x))
    index_Gᵁ = varsindex(vars_state(barotropic, Auxiliary(), FT), :Gᵁ)
    index_ΔGᵘ = varsindex(vars_state(baroclinic, Auxiliary(), FT), :ΔGᵘ)

    ### get top value (=integral over full depth) of ∫du
    ∫du = @view data_int[:, end, index_∫du, end, 1:nrealelemh]

    ### copy into Gᵁ of barotropic model
    Gᵁ = @view data_bt[:, index_Gᵁ, 1:nrealelemh]
    Gᵁ .= ∫du

    ### get top value (=integral over full depth) of ∫du
    ∫du = @view data_int[:, end:end, index_∫du, end:end, 1:nrealelemh]

    ### save vertically averaged tendency to remove from 3D tendency
    ### need to reshape for the broadcast
    ΔGᵘ = @view data_bc[:, :, index_ΔGᵘ, :, 1:nrealelemh]
    ΔGᵘ .-= ∫du / baroclinic.problem.H

    return nothing
end

@inline function cummulate_fast_solution!(
    baroclinic::HBModel{C},
    barotropic::SWModel{C},
    model_bt,
    state_bt,
    fast_time,
    fast_dt,
    substep,
) where {C <: Coupled}
    return nothing
end

@inline function reconcile_from_fast_to_slow!(
    baroclinic::HBModel{C},
    barotropic::SWModel{C},
    model_bc,
    model_bt,
    state_bc,
    state_bt,
) where {C <: Coupled}
    FT = eltype(state_bc)
    Nq, Nqk, _, _, nelemv, nelemh, nrealelemh, _ = basic_grid_info(model_bc)

    ### integrate the horizontal velocity
    model_int = model_bc.modeldata.integral_model
    integral = model_int.balance_law
    update_auxiliary_state!(model_int, integral, state_bc, 0)

    ### properly shape MPIStateArrays
    num_aux_int = number_states(integral, Auxiliary(), FT)
    data_int = model_int.state_auxiliary.data
    data_int = reshape(data_int, Nq^2, Nqk, num_aux_int, nelemv, nelemh)

    num_aux_bt = number_states(barotropic, Auxiliary(), FT)
    data_bt_aux = model_bt.state_auxiliary.data
    data_bt_aux = reshape(data_bt_aux, Nq^2, num_aux_bt, nelemh)

    num_state_bt = number_states(barotropic, Prognostic(), FT)
    data_bt_state = state_bt.data
    data_bt_state = reshape(data_bt_state, Nq^2, num_state_bt, nelemh)

    num_state_bc = number_states(baroclinic, Prognostic(), FT)
    data_bc_state = state_bc.data
    data_bc_state =
        reshape(data_bc_state, Nq^2, Nqk, num_state_bc, nelemv, nelemh)

    ### get vars indices
    index_∫u = varsindex(vars_state(integral, Auxiliary(), FT), :(∫x))
    index_Δu = varsindex(vars_state(barotropic, Auxiliary(), FT), :Δu)
    index_U = varsindex(vars_state(barotropic, Prognostic(), FT), :U)
    index_u = varsindex(vars_state(baroclinic, Prognostic(), FT), :u)
    index_η_3D = varsindex(vars_state(baroclinic, Prognostic(), FT), :η)
    index_η_2D = varsindex(vars_state(barotropic, Prognostic(), FT), :η)

    ### get top value (=integral over full depth)
    ∫u = @view data_int[:, end, index_∫u, end, 1:nrealelemh]

    ### Δu is a place holder for 1/H * (Ū - ∫u)
    Δu = @view data_bt_aux[:, index_Δu, 1:nrealelemh]
    U = @view data_bt_state[:, index_U, 1:nrealelemh]
    Δu .= 1 / baroclinic.problem.H * (U - ∫u)

    ### copy the 2D contribution down the 3D solution
    ### need to reshape for the broadcast
    data_bt_aux = reshape(data_bt_aux, Nq^2, 1, num_aux_bt, 1, nelemh)
    Δu = @view data_bt_aux[:, :, index_Δu, :, 1:nrealelemh]
    u = @view data_bc_state[:, :, index_u, :, 1:nrealelemh]
    u .+= Δu

    ### copy η from barotropic mode to baroclinic mode
    ### need to reshape for the broadcast
    data_bt_state = reshape(data_bt_state, Nq^2, 1, num_state_bt, 1, nelemh)
    η_2D = @view data_bt_state[:, :, index_η_2D, :, 1:nrealelemh]
    η_3D = @view data_bc_state[:, :, index_η_3D, :, 1:nrealelemh]
    η_3D .= η_2D

    return nothing
end
