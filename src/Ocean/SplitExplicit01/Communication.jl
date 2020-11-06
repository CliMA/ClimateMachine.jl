import ...BalanceLaws:
    tendency_from_slow_to_fast!,
    cummulate_fast_solution!,
    reconcile_from_fast_to_slow!

using Printf

@inline function set_fast_for_stepping!(
    slow::OceanModel,
    fast::BarotropicModel,
    dgFast,
    Qfast,
    S_fast,
    slow_dt,
    rkC,
    rkW,
    s,
    nStages,
    fast_time_rec,
    fast_steps,
)
    FT = typeof(slow_dt)

    #- inverse ratio of additional fast time steps (for weighted average)
    #  --> do 1/add more time-steps and average from: 1 - 1/add up to: 1 + 1/add
    add = slow.add_fast_substeps

    #- set time-step
    #  Warning: only make sense for LS3NRK33Heuns
    #  where 12 is lowest common mutiple (LCM) of all RK-Coeff inverse
    fast_dt = fast_time_rec[1]
    steps = fast_dt > 0 ? ceil(Int, slow_dt / fast_dt / FT(12)) : 1
    ntsFull = 12 * steps
    fast_dt = slow_dt / ntsFull
    add = add > 0 ? floor(Int, ntsFull / add) : 0

    #- time to start fast time-stepping (fast_time_rec[3]) for this stage:
    if s == nStages
        #  Warning: only works with few RK-scheme such as LS3NRK33Heuns
        fast_time_rec[3] = rkW[1]
        fract_dt = (1 - fast_time_rec[3])
        fast_time_rec[3] *= slow_dt
        fast_steps[2] = 1
    else
        fast_time_rec[3] = 0.0
        fract_dt = rkC[s + 1] - fast_time_rec[3]
        fast_steps[2] = 0
    end

    #- set number of sub-steps we need
    #  will time-average fast over: fast_steps[1] , fast_steps[3]
    #  centered on fract_dt*slow_dt which corresponds to advance in time of slow
    steps = ceil(Int, fract_dt * slow_dt / fast_dt)
    add = min(add, steps - 1)
    fast_steps[1] = steps - add
    fast_steps[3] = steps + add
    fast_time_rec[1] = fract_dt * slow_dt / steps

    #- select which fast time-step (fast_steps[2]) solution to save for next time-step
    #  Warning: only works with few RK-scheme such as LS3NRK33Heuns
    fast_steps[2] *= steps
    if s == 1
        fast_steps[2] = round(Int, ntsFull * rkW[1])
    end
    # @printf("Update @ s= %i : frac_dt = %.6f , dt_fast = %.1f , steps= %i , add= %i\n",
    #          s, fract_dt, fast_time_rec[1], steps, add)
    # println(" fast_time_rec = ",fast_time_rec)
    # println(" fast_steps = ",fast_steps)

    # set starting point for fast-state solution
    #  Warning: only works with few RK-scheme such as LS3NRK33Heuns
    if s == 1
        S_fast.η .= Qfast.η
        S_fast.U .= Qfast.U
    elseif s == nStages
        Qfast.η .= dgFast.state_auxiliary.η_s
        Qfast.U .= dgFast.state_auxiliary.U_s
    else
        Qfast.η .= S_fast.η
        Qfast.U .= S_fast.U
    end

    # initialise cumulative arrays
    fast_time_rec[2] = 0.0
    dgFast.state_auxiliary.η_c .= -0
    dgFast.state_auxiliary.U_c .= (@SVector [-0, -0])'

    return nothing
end

@inline function initialize_fast_state!(
    slow::OceanModel,
    fast::BarotropicModel,
    dgSlow,
    dgFast,
    Qslow,
    Qfast,
    slow_dt,
    fast_time_rec,
    fast_steps;
    firstStage = false,
)

    #- inverse ratio of additional fast time steps (for weighted average)
    #  --> do 1/add more time-steps and average from: 1 - 1/add up to: 1 + 1/add
    add = slow.add_fast_substeps

    #- set time-step and number of sub-steps we need
    #  will time-average fast over: fast_steps[1] , fast_steps[3]
    #  centered on fast_steps[2] which corresponds to advance in time of slow
    fast_dt = fast_time_rec[1]
    if add == 0
        steps = fast_dt > 0 ? ceil(Int, slow_dt / fast_dt) : 1
        fast_steps[1:3] = [1 1 1] * steps
    else
        steps = fast_dt > 0 ? ceil(Int, slow_dt / fast_dt / add) : 1
        fast_steps[2] = add * steps
        fast_steps[1] = (add - 1) * steps
        fast_steps[3] = (add + 1) * steps
    end
    fast_time_rec[1] = slow_dt / fast_steps[2]
    fast_time_rec[2] = 0.0
    # @printf("Update: frac_dt = %.1f , dt_fast = %.1f , nsubsteps= %i\n",
    #          slow_dt,fast_time_rec[1],fast_steps[3])
    # println(" fast_steps = ",fast_steps)

    dgFast.state_auxiliary.η_c .= -0
    dgFast.state_auxiliary.U_c .= (@SVector [-0, -0])'

    # set fast-state to previously stored value
    if !firstStage
        Qfast.η .= dgFast.state_auxiliary.η_s
        Qfast.U .= dgFast.state_auxiliary.U_s
    end

    return nothing
end

@inline function initialize_adjustment!(
    slow::OceanModel,
    fast::BarotropicModel,
    dgSlow,
    dgFast,
    Qslow,
    Qfast,
)
    ## reset tendency adjustment before calling Baroclinic Model
    dgSlow.state_auxiliary.ΔGu .= 0

    return nothing
end

@inline function tendency_from_slow_to_fast!(
    slow::OceanModel,
    fast::BarotropicModel,
    dgSlow,
    dgFast,
    Qslow,
    Qfast,
    dQslow2fast,
)
    FT = eltype(Qslow)

    # integrate the tendency
    tendency_dg = dgSlow.modeldata.tendency_dg
    tend = tendency_dg.balance_law
    grid = dgSlow.grid
    elems = grid.topology.elems
    update_auxiliary_state!(tendency_dg, tend, dQslow2fast, 0, elems)

    Nq, Nqk, _, _, nelemv, nelemh, nrealelemh, _ = basic_grid_info(dgSlow)

    ## get top value (=integral over full depth) of ∫du
    nb_aux_tnd = number_states(tend, Auxiliary())
    data_tnd = reshape(
        tendency_dg.state_auxiliary.data,
        Nq^2,
        Nqk,
        nb_aux_tnd,
        nelemv,
        nelemh,
    )
    index_∫du = varsindex(vars_state(tend, Auxiliary(), FT), :∫du)
    flat_∫du = @view data_tnd[:, end, index_∫du, end, 1:nrealelemh]

    ## copy into Gᵁ of dgFast
    nb_aux_fst = number_states(fast, Auxiliary())
    data_fst = reshape(dgFast.state_auxiliary.data, Nq^2, nb_aux_fst, nelemh)
    index_Gᵁ = varsindex(vars_state(fast, Auxiliary(), FT), :Gᵁ)
    boxy_Gᵁ = @view data_fst[:, index_Gᵁ, 1:nrealelemh]
    boxy_Gᵁ .= flat_∫du

    ## scale by -1/H and copy back to ΔGu
    # note: since tendency_dg.state_auxiliary.∫du is not used after this, could be
    #   re-used to store a 3-D copy of "-Gu"
    nb_aux_slw = number_states(slow, Auxiliary())
    data_slw = reshape(
        dgSlow.state_auxiliary.data,
        Nq^2,
        Nqk,
        nb_aux_slw,
        nelemv,
        nelemh,
    )
    index_ΔGu = varsindex(vars_state(slow, Auxiliary(), FT), :ΔGu)
    boxy_ΔGu = @view data_slw[:, :, index_ΔGu, :, 1:nrealelemh]
    boxy_∫du = @view data_tnd[:, end:end, index_∫du, end:end, 1:nrealelemh]
    boxy_ΔGu .= -boxy_∫du / slow.problem.H

    return nothing
end

@inline function cummulate_fast_solution!(
    fast::BarotropicModel,
    dgFast,
    Qfast,
    fast_time,
    fast_dt,
    substep,
    fast_steps,
    fast_time_rec,
)
    #- might want to use some of the weighting factors: weights_η & weights_U
    #- should account for case where fast_dt < fast.param.dt

    # cumulate Fast solution:
    if substep >= fast_steps[1]
        dgFast.state_auxiliary.U_c .+= Qfast.U
        dgFast.state_auxiliary.η_c .+= Qfast.η
        fast_time_rec[2] += 1.0
    end

    # save mid-point solution to start from the next time-step
    if substep == fast_steps[2]
        dgFast.state_auxiliary.U_s .= Qfast.U
        dgFast.state_auxiliary.η_s .= Qfast.η
    end

    return nothing
end

@inline function reconcile_from_fast_to_slow!(
    slow::OceanModel,
    fast::BarotropicModel,
    dgSlow,
    dgFast,
    Qslow,
    Qfast,
    fast_time_rec;
    lastStage = false,
)
    FT = eltype(Qslow)
    Nq, Nqk, _, _, nelemv, nelemh, nrealelemh, _ = basic_grid_info(dgSlow)
    grid = dgSlow.grid
    elems = grid.topology.elems

    # need to calculate int_u using integral kernels
    # u_slow := u_slow + (1/H) * (u_fast - \int_{-H}^{0} u_slow)

    ## get time weighted averaged out of cumulative arrays
    dgFast.state_auxiliary.U_c .*= 1 / fast_time_rec[2]
    dgFast.state_auxiliary.η_c .*= 1 / fast_time_rec[2]
    # @printf(" reconcile_from_fast_to_slow! @ s= %i : time_Count = %6.3f\n",
    #          0, fast_time_rec[2])
    #   #      s, fast_time_rec[2])

    ## Compute: \int_{-H}^{0} u_slow

    # integrate vertically horizontal velocity
    flowintegral_dg = dgSlow.modeldata.flowintegral_dg
    flowint = flowintegral_dg.balance_law
    update_auxiliary_state!(flowintegral_dg, flowint, Qslow, 0, elems)

    # get top value (=integral over full depth)
    nb_aux_flw = number_states(flowint, Auxiliary())
    data_flw = reshape(
        flowintegral_dg.state_auxiliary.data,
        Nq^2,
        Nqk,
        nb_aux_flw,
        nelemv,
        nelemh,
    )
    index_∫u = varsindex(vars_state(flowint, Auxiliary(), FT), :∫u)
    flat_∫u = @view data_flw[:, end, index_∫u, end, 1:nrealelemh]

    ## substract ∫u from U and divide by H

    # Δu is a place holder for 1/H * (Ū - ∫u)
    Δu = dgFast.state_auxiliary.Δu
    Δu .= dgFast.state_auxiliary.U_c

    nb_aux_fst = number_states(fast, Auxiliary())
    data_fst = reshape(dgFast.state_auxiliary.data, Nq^2, nb_aux_fst, nelemh)
    index_Δu = varsindex(vars_state(fast, Auxiliary(), FT), :Δu)
    boxy_Δu = @view data_fst[:, index_Δu, 1:nrealelemh]
    boxy_Δu .-= flat_∫u
    boxy_Δu ./= slow.problem.H

    ## apply the 2D correction to the 3D solution
    nb_cons_slw = number_states(slow, Prognostic())
    data_slw = reshape(Qslow.data, Nq^2, Nqk, nb_cons_slw, nelemv, nelemh)
    index_u = varsindex(vars_state(slow, Prognostic(), FT), :u)
    boxy_u = @view data_slw[:, :, index_u, :, 1:nrealelemh]
    boxy_u .+= reshape(boxy_Δu, Nq^2, 1, 2, 1, nrealelemh)

    ## save Eta from 3D model into η_diag (aux var of 2D model)
    ## and store difference between η from Barotropic Model and η_diag
    ## Note: since 3D η is not used (just in output), only do this at last stage
    ##       (save computation and get Δη diagnose over full time-step)
    if lastStage
        index_η = varsindex(vars_state(slow, Prognostic(), FT), :η)
        boxy_η_3D = @view data_slw[:, :, index_η, :, 1:nrealelemh]
        flat_η = @view data_slw[:, end, index_η, end, 1:nrealelemh]
        index_η_diag = varsindex(vars_state(fast, Auxiliary(), FT), :η_diag)
        boxy_η_diag = @view data_fst[:, index_η_diag, 1:nrealelemh]
        boxy_η_diag .= flat_η
        dgFast.state_auxiliary.Δη .=
            dgFast.state_auxiliary.η_c - dgFast.state_auxiliary.η_diag

        ## copy 2D model Eta over to 3D model
        index_η_c = varsindex(vars_state(fast, Auxiliary(), FT), :η_c)
        boxy_η_2D = @view data_fst[:, index_η_c, 1:nrealelemh]
        boxy_η_3D .= reshape(boxy_η_2D, Nq^2, 1, 1, 1, nrealelemh)

        # reset fast-state to end of time-step value
        Qfast.η .= dgFast.state_auxiliary.η_s
        Qfast.U .= dgFast.state_auxiliary.U_s
    end

    return nothing
end
