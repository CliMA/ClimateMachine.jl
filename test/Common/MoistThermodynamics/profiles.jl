#### Tested moist thermodynamic profiles

#=
This file contains functions to compute all of the
thermodynamic _states_ that MoistThermodynamics is
tested with in runtests.jl
=#


"""
    fixed_lapse_rate(
        param_set::AbstractParameterSet,
        z::FT,
        T_surface::FT,
        T_min::FT,
        ) where {FT <: AbstractFloat}

Fixed lapse rate hydrostatic reference state given

 - `param_set` parameter set, used to dispatch planet parameter function calls
 - `z` altitude
 - `T_surface` surface temperature
 - `T_min` minimum temperature
"""
function fixed_lapse_rate(
    param_set::AbstractParameterSet,
    z::FT,
    T_surface::FT,
    T_min::FT,
) where {FT <: AbstractFloat}
    _grav::FT = grav(param_set)
    _cp_d::FT = cp_d(param_set)
    _R_d::FT = R_d(param_set)
    _MSLP::FT = MSLP(param_set)
    Γ = _grav / _cp_d
    z_tropopause = (T_surface - T_min) / Γ
    H_min = _R_d * T_min / _grav
    T = max(T_surface - Γ * z, T_min)
    p = _MSLP * (T / T_surface)^(_grav / (_R_d * Γ))
    T == T_min && (p = p * exp(-(z - z_tropopause) / H_min))
    ρ = p / (_R_d * T)
    return T, p, ρ
end

"""
    tested_profiles(param_set, n::Int, ::Type{FT})

A range of input arguments to thermodynamic state constructors

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `z_all` altitude
 - `e_int` internal energy
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity
 - `q_pt` phase partition
 - `T` air temperature
 - `θ_liq_ice` liquid-ice potential temperature

that are tested for convergence in saturation adjustment.

Note that the output vectors are of size ``n*n_RS``, and they
should span the input arguments to all of the constructors.
"""
function tested_profiles(
    param_set::AbstractParameterSet,
    n::Int,
    ::Type{FT},
) where {FT}

    n_RS1 = 10
    n_RS2 = 20
    n_RS = n_RS1 + n_RS2
    z_range = range(FT(0), stop = FT(2.5e4), length = n)
    relative_sat1 = range(FT(0), stop = FT(1), length = n_RS1)
    relative_sat2 = range(FT(1), stop = FT(1.02), length = n_RS2)
    relative_sat = [relative_sat1..., relative_sat2...]
    T_min = FT(150)
    T_surface = FT(350)

    T = zeros(FT, n, n_RS)
    p = zeros(FT, n, n_RS)
    ρ = zeros(FT, n, n_RS)
    RS = zeros(FT, n, n_RS)
    z_all = zeros(FT, n, n_RS)
    for i in eachindex(z_range)
        for j in eachindex(relative_sat)
            args = fixed_lapse_rate(param_set, z_range[i], T_surface, T_min)
            k = CartesianIndex(i, j)
            z_all[k] = z_range[i]
            T[k] = args[1]
            p[k] = args[2]
            ρ[k] = args[3]
            RS[k] = relative_sat[j]
        end
    end
    T = reshape(T, n * n_RS)
    p = reshape(p, n * n_RS)
    ρ = reshape(ρ, n * n_RS)
    RS = reshape(RS, n * n_RS)
    relative_sat = RS

    # Additional variables
    q_sat = q_vap_saturation.(Ref(param_set), T, ρ)
    q_tot = min.(relative_sat .* q_sat, FT(1))
    q_pt = PhasePartition_equil.(Ref(param_set), T, ρ, q_tot)
    e_int = internal_energy.(Ref(param_set), T, q_pt)
    θ_liq_ice = liquid_ice_pottemp.(Ref(param_set), T, ρ, q_pt)

    # Sort by altitude (for visualization):
    # TODO: Refactor, by avoiding phase partition copy, once
    # https://github.com/JuliaLang/julia/pull/33515 merges
    q_liq = getproperty.(q_pt, :liq)
    q_ice = getproperty.(q_pt, :ice)
    args = [z_all, e_int, ρ, q_tot, q_liq, q_ice, T, p, θ_liq_ice]
    args = collect(zip(args...))
    sort!(args)
    z_all = getindex.(args, 1)
    e_int = getindex.(args, 2)
    ρ = getindex.(args, 3)
    q_tot = getindex.(args, 4)
    q_liq = getindex.(args, 5)
    q_ice = getindex.(args, 6)
    T = getindex.(args, 7)
    p = getindex.(args, 8)
    θ_liq_ice = getindex.(args, 9)
    q_pt = PhasePartition.(q_tot, q_liq, q_ice)
    args = [z_all, e_int, ρ, q_tot, q_pt, T, p, θ_liq_ice]
    return args
end
