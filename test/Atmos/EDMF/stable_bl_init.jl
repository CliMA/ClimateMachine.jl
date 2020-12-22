
using ClimateMachine
ClimateMachine.init(;
    parse_clargs = true,
    output_dir = get(ENV, "CLIMATEMACHINE_SETTINGS_OUTPUT_DIR", "output"),
    fix_rng_seed = true,
)
using ClimateMachine.Checkpoint
using ClimateMachine.ODESolvers
using ClimateMachine.SingleStackUtils
using ClimateMachine.SystemSolvers
using ClimateMachine.BalanceLaws: vars_state
const clima_dir = dirname(dirname(pathof(ClimateMachine)));

include(joinpath(clima_dir, "experiments", "AtmosLES", "stable_bl_model.jl"))
include("edmf_model.jl")
include("edmf_kernels.jl")

"""
    init_state_prognostic!(
            turbconv::EDMF{FT},
            m::AtmosModel{FT},
            state::Vars,
            aux::Vars,
            localgeo,
            t::Real,
        ) where {FT}

Initialize EDMF state variables.
This method is only called at `t=0`.
"""
function init_state_prognostic!(
    turbconv::EDMF{FT},
    m::AtmosModel{FT},
    state::Vars,
    aux::Vars,
    localgeo,
    t::Real,
) where {FT}
    # Aliases:
    gm = state
    en = state.turbconv.environment
    up = state.turbconv.updraft
    N_up = n_updrafts(turbconv)
    # GCM setting - Initialize the grid mean profiles of prognostic variables (ρ,e_int,q_tot,u,v,w)
    z = altitude(m, aux)

    # SCM setting - need to have separate cases coded and called from a folder - see what LES does
    # a thermo state is used here to convert the input θ to e_int profile
    e_int = internal_energy(m, state, aux)

    ts = PhaseDry(m.param_set, e_int, state.ρ)
    T = air_temperature(ts)
    p = air_pressure(ts)
    q = PhasePartition(ts)
    θ_liq = liquid_ice_pottemp(ts)

    a_min = turbconv.subdomains.a_min
    @unroll_map(N_up) do i
        up[i].ρa = gm.ρ * a_min
        up[i].ρaw = gm.ρu[3] * a_min
        up[i].ρaθ_liq = gm.ρ * a_min * θ_liq
        up[i].ρaq_tot = FT(0)
    end

    # initialize environment covariance with zero for now
    if z <= FT(250)
        en.ρatke =
            gm.ρ *
            FT(0.4) *
            FT(1 - z / 250.0) *
            FT(1 - z / 250.0) *
            FT(1 - z / 250.0)
        en.ρaθ_liq_cv =
            gm.ρ *
            FT(0.4) *
            FT(1 - z / 250.0) *
            FT(1 - z / 250.0) *
            FT(1 - z / 250.0)
    else
        en.ρatke = FT(0)
        en.ρaθ_liq_cv = FT(0)
    end
    en.ρaq_tot_cv = FT(0)
    en.ρaθ_liq_q_tot_cv = FT(0)
    return nothing
end;
