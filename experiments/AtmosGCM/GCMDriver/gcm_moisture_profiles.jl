# GCM Initial Moisture Profiles
# This file contains helpers and lists currely avaiable options

abstract type AbstractMoistureProfile end
struct NoMoistureProfile <: AbstractMoistureProfile end
struct ZeroMoistureProfile <: AbstractMoistureProfile end
struct MoistLowTropicsMoistureProfile <: AbstractMoistureProfile end

# Helper for parsing `--init-moisture-profile` command line argument
function parse_moisture_profile_arg(arg)
    if arg === nothing
        moisture_profile = nothing
    elseif arg == "moist_low_tropics"
        moisture_profile = MoistLowTropicsMoistureProfile()
    elseif arg == "zero"
        moisture_profile = ZeroMoistureProfile()
    elseif arg == "dry"
        moisture_profile = NoMoistureProfile()
    else
        error("unknown moisture profile: " * arg)
    end

    return moisture_profile
end

# Initial moisture profile for a dry model setup
function init_moisture_profile(
    ::NoMoistureProfile,
    bl,
    state,
    aux,
    coords,
    t,
    p,
)
    FT = eltype(state)
    return FT(0)
end

# Initial moisture profile for a moist model setup with 0 initial moisture
function init_moisture_profile(
    ::ZeroMoistureProfile,
    bl,
    state,
    aux,
    coords,
    t,
    p,
)
    FT = eltype(state)
    return FT(0)
end

# Initial moisture profile following
# Ullrich et al. (2016) Dynamical Core Model Intercomparison Project (DCMIP2016) Test Case Document
function init_moisture_profile(
    ::MoistLowTropicsMoistureProfile,
    bl,
    state,
    aux,
    coords,
    t,
    p,
)
    FT = eltype(state)

    _p_0 = MSLP(bl.param_set)::FT

    φ = latitude(bl, aux)

    # Humidity parameters
    p_w::FT = 34e3              # Pressure width parameter for specific humidity
    η_crit::FT = p_w / _p_0     # Critical pressure coordinate
    q_0::FT = 0.018             # Maximum specific humidity (default: 0.018)
    q_t::FT = 1e-12             # Specific humidity above artificial tropopause
    φ_w::FT = 2π / 9            # Specific humidity latitude wind parameter

    # get q_tot profile if needed
    η = p / _p_0                # Pressure coordinate η
    if η > η_crit
        q_tot = q_0 * exp(-(φ / φ_w)^4) * exp(-((η - 1) * _p_0 / p_w)^2)
    else
        q_tot = q_t
    end

    return q_tot
end
