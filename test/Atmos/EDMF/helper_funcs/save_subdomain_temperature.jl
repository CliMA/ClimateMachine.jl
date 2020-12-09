# Convenience wrapper
save_subdomain_temperature!(m, state, aux) =
    save_subdomain_temperature!(m, m.moisture, state, aux)

using KernelAbstractions: @print

"""
    save_subdomain_temperature!(
        m::AtmosModel,
        moist::EquilMoist,
        state::Vars,
        aux::Vars,
    )

Updates the subdomain sensible temperature, given:
 - `m`, an `AtmosModel`
 - `moist`, an `EquilMoist` model
 - `state`, state variables
 - `aux`, auxiliary variables
"""
function save_subdomain_temperature!(
    m::AtmosModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
)
    N_up = n_updrafts(m.turbconv)
    ts = recover_thermo_state(m, state, aux)
    ts_up = new_thermo_state_up(m, state, aux, ts)
    ts_en = new_thermo_state_en(m, state, aux, ts)

    @unroll_map(N_up) do i
        aux.turbconv.updraft[i].T = air_temperature(ts_up[i])
    end
    aux.turbconv.environment.T = air_temperature(ts_en)
    return nothing
end

function validate_variables(m::AtmosModel, state, aux, caller)
    behaved = Dict()
    FT = eltype(state)
    up = state.turbconv.updraft
    a_min = m.turbconv.subdomains.a_min
    a_max = m.turbconv.subdomains.a_max
    ρa_min = state.ρ * a_min
    ρa_max = state.ρ-ρa_min
    # behaved["ρa_min"] = 0 < ρa_min < 1.3, ρa_min
    # behaved["ρa_max"] = 0 < ρa_max < 1.3, ρa_max
    behaved["ρ"] = (0 < state.ρ < 2, state.ρ)
    N_up = n_updrafts(m.turbconv)
    z = altitude(m, aux)
    ρ = state.ρ
    ε = 1000eps(FT)
    @unroll_map(N_up) do i
        behaved["up[$i].ρa"] = 0 < up[i].ρa < 2*(1-a_min), up[i].ρa
        behaved["up[$i].a"] = a_min - ε < up[i].ρa / ρ < a_max + ε, up[i].ρa / ρ
    end
    a_en = environment_area(state, aux, N_up)
    behaved["a_en"] = a_min - ε <= a_en <= a_max + ε, a_en

    if !all(first.(values(behaved)))
        @show caller
        for (k,v) in behaved
            println("$(v[1]) ,$k = $(v[2])")
        end
        @unroll_map(N_up) do i
            @show z, i, up[i].ρa, up[i].ρa/ρ
        end
        @show a_min
        @show up[1].ρa / ρ
        @show a_en
        @show a_max
        @show ρa_min
        @show ρa_max
        @show up[1].ρaθ_liq
        error("Misbehaved state.")
    end
end

# No need to save temperature for DryModel.
function save_subdomain_temperature!(
    m::AtmosModel,
    moist::DryModel,
    state::Vars,
    aux::Vars,
)
    validate_variables(m, state, aux, "save_subdomain_temperature! (dry)")
    N_up = n_updrafts(m.turbconv)
    ts = recover_thermo_state(m, state, aux)
    ts_up = new_thermo_state_up(m, state, aux, ts)
    ts_en = new_thermo_state_en(m, state, aux, ts)

end
