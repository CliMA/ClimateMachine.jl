#### Diagnose environment variables

"""
    environment_vars(state::Vars, N_up::Int)

A NamedTuple of environment variables
"""
function environment_vars(state::Vars, N_up::Int)
    return (a = environment_area(state, N_up), w = environment_w(state, N_up))
end

"""
    environment_area(
        state::Vars,
        N_up::Int,
    )

Returns the environmental area fraction, given:
 - `state`, state variables
 - `N_up`, number of updrafts
"""
function environment_area(state::Vars, N_up::Int)
    up = state.turbconv.updraft
    return 1 - sum(vuntuple(i -> up[i].ρa, N_up)) / state.ρ
end

"""
    environment_w(state::Vars, N_up::Int)

Returns the environmental vertical velocity, given:
 - `state`, state variables
 - `N_up`, number of updrafts
"""
function environment_w(state::Vars, N_up::Int)
    ρ_inv = 1 / state.ρ
    a_en = environment_area(state, N_up)
    up = state.turbconv.updraft
    return (state.ρu[3] - sum(vuntuple(i -> up[i].ρaw, N_up))) / a_en * ρ_inv
end

"""
    grid_mean_b(env, a_up, N_up::Int, buoyancy_up, buoyancy_en)

Returns the grid-mean buoyancy with respect to the
reference state, given:
 - `env`, environment variables
 - `a_up`, updraft area fractions
 - `N_up`, number of updrafts
 - `buoyancy_up`, updraft buoyancies
 - `buoyancy_en`, environment buoyancy
"""
function grid_mean_b(env, a_up, N_up::Int, buoyancy_up, buoyancy_en)
    ∑abuoyancy_up = sum(vuntuple(i -> buoyancy_up[i] * a_up[i], N_up))
    return env.a * buoyancy_en + ∑abuoyancy_up
end
