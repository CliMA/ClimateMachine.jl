#### Tested moist thermodynamic profiles

#=
This file contains functions to compute all of the
thermodynamic _states_ that Thermodynamics is
tested with in runtests.jl
=#

"""
    unpack_fields(_struct, syms...)

Unpack struct properties `syms`
from struct `_struct`

# Example
```julia
julia> struct Foo;a;b;c;end

julia> f = Foo(1,2,3)
Foo(1, 2, 3)

julia> @unpack_fields f a c; @show a c
a = 1
c = 3
```
"""
macro unpack_fields(_struct, syms...)
    thunk = Expr(:block)
    for sym in syms
        push!(
            thunk.args,
            :($(esc(sym)) = getproperty($(esc(_struct)), $(QuoteNode(sym)))),
        )
    end
    push!(thunk.args, nothing)
    return thunk
end

"""
    ProfileSet

A set of profiles used to test Thermodynamics.
"""
struct ProfileSet{FT}
    z::Array{FT}                        # Altitude
    T::Array{FT}                        # Temperature
    p::Array{FT}                        # Pressure
    RS::Array{FT}                       # Relative humidity
    e_int::Array{FT}                    # Internal energy
    ρ::Array{FT}                        # Density
    θ_liq_ice::Array{FT}                # Potential temperature
    q_tot::Array{FT}                    # Total specific humidity
    q_liq::Array{FT}                    # Liquid specific humidity
    q_ice::Array{FT}                    # Ice specific humidity
    q_pt::Array{PhasePartition{FT}}     # Phase partition
    RH::Array{FT}                       # Relative humidity
    SS::Array{FT}                       # Super saturation
end

"""
    input_config(
        FT;
        n=50,
        n_RS1=10,
        n_RS2=20,
        T_min=FT(150),
        T_surface=FT(350)
    ) where {FT}

Return input arguments to construct profiles
"""
function input_config(
    FT;
    n = 50,
    n_RS1 = 10,
    n_RS2 = 20,
    T_surface = FT(350),
    T_min = FT(150),
)
    n_RS = n_RS1 + n_RS2
    z_range = range(FT(0), stop = FT(2.5e4), length = n)
    relative_sat1 = range(FT(0), stop = FT(1), length = n_RS1)
    relative_sat2 = range(FT(1), stop = FT(1.02), length = n_RS2)
    relative_sat = [relative_sat1..., relative_sat2...]
    return z_range, relative_sat, T_surface, T_min
end

"""
    shared_profiles(
        param_set::AbstractParameterSet,
        z_range::AbstractArray,
        relative_sat::AbstractArray,
        T_surface::FT,
        T_min::FT,
    ) where {FT}

Compute profiles shared across `PhaseDry`,
`PhaseEquil` and `PhaseNonEquil` thermodynamic
states, including:
 - `z` altitude
 - `T_virt` virtual temperature
 - `p` pressure
 - `RS` relative saturation
"""
function shared_profiles(
    param_set::AbstractParameterSet,
    z_range::AbstractArray,
    relative_sat::AbstractArray,
    T_surface::FT,
    T_min::FT,
) where {FT}
    n_RS = length(relative_sat)
    n = length(z_range)
    T_virt = Array{FT}(undef, n * n_RS)
    p = Array{FT}(undef, n * n_RS)
    RS = Array{FT}(undef, n * n_RS)
    z = Array{FT}(undef, n * n_RS)
    linear_indices = LinearIndices((1:n, 1:n_RS))
    profile = DecayingTemperatureProfile{FT}(param_set, T_surface, T_min)
    for i in linear_indices.indices[1]
        for j in linear_indices.indices[2]
            k = linear_indices[i, j]
            z[k] = z_range[i]
            T_virt[k], p[k] = profile(param_set, z[k])
            RS[k] = relative_sat[j]
        end
    end
    return z, T_virt, p, RS
end

####
#### PhaseDry
####

"""
    PhaseDryProfiles(param_set, ::Type{FT})

Returns a `ProfileSet` used to test dry thermodynamic states.
"""
function PhaseDryProfiles(
    param_set::AbstractParameterSet,
    ::Type{FT},
) where {FT}

    z_range, relative_sat, T_surface, T_min = input_config(FT)
    z, T_virt, p, RS =
        shared_profiles(param_set, z_range, relative_sat, T_surface, T_min)
    _R_d::FT = R_d(param_set)
    T = T_virt
    ρ = p ./ (_R_d .* T)

    # Additional variables
    phase_type = PhaseDry
    q_tot = zeros(FT, length(RS))
    q_pt = PhasePartition_equil.(Ref(param_set), T, ρ, q_tot, Ref(phase_type))
    e_int = internal_energy.(Ref(param_set), T, q_pt)
    θ_liq_ice = liquid_ice_pottemp.(Ref(param_set), T, ρ, q_pt)
    q_liq = getproperty.(q_pt, :liq)
    q_ice = getproperty.(q_pt, :ice)
    RH = relative_humidity.(Ref(param_set), T, p, e_int, Ref(phase_type), q_pt)

    # TODO: Update this once a super saturation method exists
    # SS = super_saturation.(Ref(param_set), T, p, e_int, Ref(phase_type), q_pt)
    SS = zeros(FT, length(RH))

    return ProfileSet(
        z,
        T,
        p,
        RS,
        e_int,
        ρ,
        θ_liq_ice,
        q_tot,
        q_liq,
        q_ice,
        q_pt,
        RH,
        SS,
    )
end

####
#### PhaseEquil
####

"""
    PhaseEquilProfiles(param_set, ::Type{FT})

Returns a `ProfileSet` used to test moist states in thermodynamic equilibrium.
"""
function PhaseEquilProfiles(
    param_set::AbstractParameterSet,
    ::Type{FT},
) where {FT}

    z_range, relative_sat, T_surface, T_min = input_config(FT)

    z, T_virt, p, RS =
        shared_profiles(param_set, z_range, relative_sat, T_surface, T_min)
    _R_d::FT = R_d(param_set)
    T = T_virt
    ρ = p ./ (_R_d .* T)

    # Additional variables
    phase_type = PhaseEquil
    q_sat = q_vap_saturation.(Ref(param_set), T, ρ, Ref(phase_type))
    q_tot = min.(RS .* q_sat, FT(1))
    q_pt = PhasePartition_equil.(Ref(param_set), T, ρ, q_tot, Ref(phase_type))
    e_int = internal_energy.(Ref(param_set), T, q_pt)
    θ_liq_ice = liquid_ice_pottemp.(Ref(param_set), T, ρ, q_pt)
    q_liq = getproperty.(q_pt, :liq)
    q_ice = getproperty.(q_pt, :ice)
    RH = relative_humidity.(Ref(param_set), T, p, e_int, Ref(phase_type), q_pt)

    # TODO: Update this once a super saturation method exists
    # SS = super_saturation.(Ref(param_set), T, p, e_int, Ref(phase_type), q_pt)
    SS = zeros(FT, length(RH))

    return ProfileSet(
        z,
        T,
        p,
        RS,
        e_int,
        ρ,
        θ_liq_ice,
        q_tot,
        q_liq,
        q_ice,
        q_pt,
        RH,
        SS,
    )
end
