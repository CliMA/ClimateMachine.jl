#### Diagnose environment variables

"""
    environment_vars(state::Vars, aux::Vars, N_up::Int)

A NamedTuple of environment variables
"""
function environment_vars(state::Vars, aux::Vars, N_up::Int)
    return (
        a = environment_area(state, aux, N_up),
        w = environment_w(state, aux, N_up),
    )
end

"""
    environment_area(
        state::Vars,
        aux::Vars,
        N_up::Int,
    )

Returns the environmental area fraction, given:
 - `state`, state variables
 - `aux`, auxiliary variables
 - `N_up`, number of updrafts
"""
function environment_area(state::Vars, aux::Vars, N_up::Int)
    up = state.turbconv.updraft
    return 1 - sum(vuntuple(i -> up[i].ρa, N_up)) / state.ρ
end

"""
    environment_w(
        state::Vars,
        aux::Vars,
        N_up::Int,
    )

Returns the environmental vertical velocity, given:
 - `state`, state variables
 - `aux`, auxiliary variables
 - `N_up`, number of updrafts
"""
function environment_w(state::Vars, aux::Vars, N_up::Int)
    ρ_inv = 1 / state.ρ
    a_en = environment_area(state, aux, N_up)
    up = state.turbconv.updraft
    return (state.ρu[3] - sum(vuntuple(i -> up[i].ρaw, N_up))) / a_en * ρ_inv
end

"""
    grid_mean_b(
        state::Vars,
        aux::Vars,
        N_up::Int,
    )

Returns the grid-mean buoyancy with respect to the
reference state, given:
 - `state`, state variables
 - `aux`, auxiliary variables
 - `N_up`, number of updrafts
"""
function grid_mean_b(state::Vars, aux::Vars, N_up::Int)
    ρ_inv = 1 / state.ρ
    a_en = environment_area(state, aux, N_up)
    up = state.turbconv.updraft
    en_aux = aux.turbconv.environment
    up_aux = aux.turbconv.updraft
    up_buoy = sum(vuntuple(i -> up_aux[i].buoyancy * up[i].ρa * ρ_inv, N_up))
    return a_en * en_aux.buoyancy + up_buoy
end
