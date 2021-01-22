#### Tested moist thermodynamic profiles

#=
This file contains functions to compute all of the
thermodynamic _states_ that Thermodynamics is
tested with in runtests.jl
=#

using Random

"""
    ProfileSet

A set of profiles used to test Thermodynamics.
"""
struct ProfileSet{AFT, QPT, PT}
    z::AFT          # Altitude
    T::AFT          # Temperature
    p::AFT          # Pressure
    RS::AFT         # Relative saturation
    e_int::AFT      # Internal energy
    ρ::AFT          # Density
    θ_liq_ice::AFT  # Liquid Ice Potential temperature
    q_tot::AFT      # Total specific humidity
    q_liq::AFT      # Liquid specific humidity
    q_ice::AFT      # Ice specific humidity
    q_pt::QPT       # Phase partition
    RH::AFT         # Relative humidity
    e_pot::AFT      # gravitational potential
    u::AFT          # velocity (x component)
    v::AFT          # velocity (y component)
    w::AFT          # velocity (z component)
    e_kin::AFT      # kinetic energy
    phase_type::PT  # Phase type (e.g., `PhaseDry`, `PhaseEquil`)
end

function Base.iterate(ps::ProfileSet, state = 1)
    state > length(ps.z) && return nothing
    nt = (
        z = ps.z[state],
        T = ps.T[state],
        p = ps.p[state],
        RS = ps.RS[state],
        e_int = ps.e_int[state],
        ρ = ps.ρ[state],
        θ_liq_ice = ps.θ_liq_ice[state],
        q_tot = ps.q_tot[state],
        q_liq = ps.q_liq[state],
        q_ice = ps.q_ice[state],
        q_pt = ps.q_pt[state],
        RH = ps.RH[state],
        e_pot = ps.e_pot[state],
        u = ps.u[state],
        v = ps.v[state],
        w = ps.w[state],
        e_kin = ps.e_kin[state],
        phase_type = ps.phase_type,
    )
    return (nt, state + 1)
end
Base.IteratorSize(::ProfileSet) = Base.HasLength()
Base.length(ps::ProfileSet) = length(ps.z)

# Since we use `rand` to generate the ProfileSet,
# just initialize on the CPU, and provide convert
# function to move arrays to the GPU.
convert_profile_set(ps::ProfileSet, ArrayType, slice) = ProfileSet(
    ArrayType(ps.z[slice]),
    ArrayType(ps.T[slice]),
    ArrayType(ps.p[slice]),
    ArrayType(ps.RS[slice]),
    ArrayType(ps.e_int[slice]),
    ArrayType(ps.ρ[slice]),
    ArrayType(ps.θ_liq_ice[slice]),
    ArrayType(ps.q_tot[slice]),
    ArrayType(ps.q_liq[slice]),
    ArrayType(ps.q_ice[slice]),
    PhasePartition.(ps.q_tot[slice], ps.q_liq[slice], ps.q_ice[slice]),
    ArrayType(ps.RH[slice]),
    ArrayType(ps.e_pot[slice]),
    ArrayType(ps.u[slice]),
    ArrayType(ps.v[slice]),
    ArrayType(ps.w[slice]),
    ArrayType(ps.e_kin[slice]),
    ps.phase_type,
)

"""
    input_config(
        ArrayType;
        n=50,
        n_RS1=10,
        n_RS2=20,
        T_min=150,
        T_surface=340
    )

Return input arguments to construct profiles
"""
function input_config(
    ArrayType;
    n = 50,
    n_RS1 = 10,
    n_RS2 = 20,
    T_surface = 340,
    T_min = 150,
)
    n_RS = n_RS1 + n_RS2
    z_range = ArrayType(range(0, stop = 2.5e4, length = n))
    relative_sat1 = ArrayType(range(0, stop = 1, length = n_RS1))
    relative_sat2 = ArrayType(range(1, stop = 1.02, length = n_RS2))
    relative_sat = vcat(relative_sat1, relative_sat2)
    return z_range, relative_sat, T_surface, T_min
end

"""
    shared_profiles(
        param_set::AbstractParameterSet,
        z_range::AbstractArray,
        relative_sat::AbstractArray,
        T_surface,
        T_min,
    )

Compute profiles shared across `PhaseDry`,
`PhaseEquil` and `PhaseNonEquil` thermodynamic
states, including:
 - `z` altitude
 - `T` temperature
 - `p` pressure
 - `RS` relative saturation
"""
function shared_profiles(
    param_set::AbstractParameterSet,
    z_range::AbstractArray,
    relative_sat::AbstractArray,
    T_surface,
    T_min,
)
    FT = eltype(z_range)
    n_RS = length(relative_sat)
    n = length(z_range)
    T = similar(z_range, n * n_RS)
    p = similar(z_range, n * n_RS)
    RS = similar(z_range, n * n_RS)
    z = similar(z_range, n * n_RS)
    linear_indices = LinearIndices((1:n, 1:n_RS))
    # We take the virtual temperature, returned here,
    # as the temperature, and then compute a thermodynamic
    # state consistent with that temperature. This profile
    # will not be in hydrostatic balance, but this does not
    # matter for the thermodynamic test profiles.
    profile =
        DecayingTemperatureProfile{FT}(param_set, FT(T_surface), FT(T_min))
    for i in linear_indices.indices[1]
        for j in linear_indices.indices[2]
            k = linear_indices[i, j]
            z[k] = z_range[i]
            T[k], p[k] = profile(param_set, z[k])
            RS[k] = relative_sat[j]
        end
    end
    return z, T, p, RS
end

####
#### PhaseDry
####

"""
    PhaseDryProfiles(param_set, ::Type{ArrayType})

Returns a `ProfileSet` used to test dry thermodynamic states.
"""
function PhaseDryProfiles(
    param_set::AbstractParameterSet,
    ::Type{ArrayType},
) where {ArrayType}
    phase_type = PhaseDry

    z_range, relative_sat, T_surface, T_min = input_config(ArrayType)
    z, T_virt, p, RS =
        shared_profiles(param_set, z_range, relative_sat, T_surface, T_min)
    T = T_virt
    FT = eltype(T)
    _R_d::FT = R_d(param_set)
    _grav::FT = grav(param_set)
    ρ = p ./ (_R_d .* T)

    # Additional variables
    q_tot = similar(T)
    fill!(q_tot, 0)
    q_pt = PhasePartition_equil.(param_set, T, ρ, q_tot, phase_type)
    e_int = internal_energy.(param_set, T, q_pt)
    θ_liq_ice = liquid_ice_pottemp.(param_set, T, ρ, q_pt)
    q_liq = getproperty.(q_pt, :liq)
    q_ice = getproperty.(q_pt, :ice)
    RH = relative_humidity.(param_set, T, p, phase_type, q_pt)
    e_pot = _grav * z
    Random.seed!(15)
    u = rand(FT, size(T)) * 50
    v = rand(FT, size(T)) * 50
    w = rand(FT, size(T)) * 50
    e_kin = (u .^ 2 .+ v .^ 2 .+ w .^ 2) / 2


    return ProfileSet{typeof(T), typeof(q_pt), typeof(phase_type)}(
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
        e_pot,
        u,
        v,
        w,
        e_kin,
        phase_type,
    )
end

####
#### PhaseEquil
####

"""
    PhaseEquilProfiles(param_set, ::Type{ArrayType})

Returns a `ProfileSet` used to test moist states in thermodynamic equilibrium.
"""
function PhaseEquilProfiles(
    param_set::AbstractParameterSet,
    ::Type{ArrayType},
) where {ArrayType}
    phase_type = PhaseEquil

    # Prescribe z_range, relative_sat, T_surface, T_min
    z_range, relative_sat, T_surface, T_min = input_config(ArrayType)

    # Compute T, p, from DecayingTemperatureProfile, (reshape RS)
    z, T_virt, p, RS =
        shared_profiles(param_set, z_range, relative_sat, T_surface, T_min)
    T = T_virt

    FT = eltype(T)
    _R_d = FT(R_d(param_set))
    _grav::FT = grav(param_set)
    # Compute total specific humidity from temperature, pressure
    # and relative saturation, and partition the saturation excess
    # according to temperature.
    ρ = p ./ (_R_d .* T)
    q_tot = RS .* q_vap_saturation.(Ref(param_set), T, ρ, Ref(phase_type))
    q_pt = PhasePartition_equil.(Ref(param_set), T, ρ, q_tot, Ref(phase_type))

    # Extract phase partitioning and update pressure
    # to be thermodynamically consistent with T, ρ, q_pt
    q_liq = getproperty.(q_pt, :liq)
    q_ice = getproperty.(q_pt, :ice)
    p = air_pressure.(Ref(param_set), T, ρ, q_pt)

    e_int = internal_energy.(Ref(param_set), T, q_pt)
    θ_liq_ice = liquid_ice_pottemp.(Ref(param_set), T, ρ, q_pt)
    RH = relative_humidity.(Ref(param_set), T, p, Ref(phase_type), q_pt)
    e_pot = _grav * z
    Random.seed!(15)
    u = rand(FT, size(T)) * 50
    v = rand(FT, size(T)) * 50
    w = rand(FT, size(T)) * 50
    e_kin = (u .^ 2 .+ v .^ 2 .+ w .^ 2) / 2

    return ProfileSet{typeof(T), typeof(q_pt), typeof(phase_type)}(
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
        e_pot,
        u,
        v,
        w,
        e_kin,
        phase_type,
    )
end
