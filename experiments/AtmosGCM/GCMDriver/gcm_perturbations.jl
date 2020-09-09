# GCM Initial Perturbation
# This file contains helpers and lists currely avaiable options

using Distributions
using Random
using Distributions: Uniform
using Random: rand

abstract type AbstractPerturbation end
struct NoPerturbation <: AbstractPerturbation end
struct DeterministicPerturbation <: AbstractPerturbation end
struct RandomPerturbation <: AbstractPerturbation end

# Helper for parsing `--init-perturbation` command line argument
function parse_perturbation_arg(arg)
    if arg === nothing
        perturbation = nothing
    elseif arg == "deterministic"
        perturbation = DeterministicPerturbation()
    elseif arg == "zero"
        perturbation = NoPerturbation()
    elseif arg == "random"
        perturbation = RandomPerturbation()
    else
        error("unknown perturbation: " * arg)
    end

    return perturbation
end

function init_perturbation(::NoPerturbation, bl, state, aux, coords, t)
    FT = eltype(state)

    u′, v′, w′ = (FT(0), FT(0), FT(0))
    rand_pert = FT(1)

    return u′, v′, w′, rand_pert
end

# Velocity perturbation following
# Ullrich et al. (2016) Dynamical Core Model Intercomparison Project (DCMIP2016) Test Case Document
function init_perturbation(
    ::DeterministicPerturbation,
    bl,
    state,
    aux,
    coords,
    t,
)
    FT = eltype(state)

    # get parameters
    _a = planet_radius(bl.param_set)::FT

    φ = latitude(bl, aux)
    λ = longitude(bl, aux)
    z = altitude(bl, aux)

    # perturbation specific parameters
    z_t::FT = 15e3
    λ_c::FT = π / 9
    φ_c::FT = 2 * π / 9
    d_0::FT = _a / 6
    V_p::FT = 10

    F_z::FT = 1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3
    if z > z_t
        F_z = FT(0)
    end
    d::FT = _a * acos(sin(φ) * sin(φ_c) + cos(φ) * cos(φ_c) * cos(λ - λ_c))
    c3::FT = cos(π * d / 2 / d_0)^3
    s1::FT = sin(π * d / 2 / d_0)
    if 0 < d < d_0 && d != FT(_a * π)
        u′::FT =
            -16 * V_p / 3 / sqrt(3) *
            F_z *
            c3 *
            s1 *
            (-sin(φ_c) * cos(φ) + cos(φ_c) * sin(φ) * cos(λ - λ_c)) /
            sin(d / _a)
        v′::FT =
            16 * V_p / 3 / sqrt(3) * F_z * c3 * s1 * cos(φ_c) * sin(λ - λ_c) /
            sin(d / _a)
    else
        u′ = FT(0)
        v′ = FT(0)
    end
    w′ = FT(0)
    rand_pert = FT(1)

    return u′, v′, w′, rand_pert
end

function init_perturbation(::RandomPerturbation, bl, state, aux, coords, t)
    FT = eltype(state)
    u′, v′, w′ = (FT(0), FT(0), FT(0))
    rand_pert = FT(1.0 + rand(Uniform(-1e-3, 1e-3)))

    return u′, v′, w′, rand_pert
end
