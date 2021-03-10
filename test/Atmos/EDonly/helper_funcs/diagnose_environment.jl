#### Diagnose environment variables

"""
    environment_vars(state::Vars)

A NamedTuple of environment variables
"""
function environment_vars(state::Vars)
    return (a = environment_area(state), w = environment_w(state))
end

"""
    environment_area(
        state::Vars,
    )

Returns the environmental area fraction, given:
 - `state`, state variables
"""
function environment_area(state::Vars)
    return 1
end

"""
    environment_w(state::Vars)

Returns the environmental vertical velocity, given:
 - `state`, state variables
"""
function environment_w(state::Vars)
    ρ_inv = 1 / state.ρ
    a_en = environment_area(state)
    return state.ρu[3]* ρ_inv
end