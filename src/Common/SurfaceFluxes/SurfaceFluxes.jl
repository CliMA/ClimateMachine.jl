"""
    SurfaceFluxes

## Interface
  - [`surface_conditions`](@ref) computes
    - Monin-Obukhov length
    - Potential temperature flux (if not given) using Monin-Obukhov theory
    - transport fluxes using Monin-Obukhov theory
    - friction velocity/temperature scale/tracer scales
    - exchange coefficients

## References
 - [Nishizawa2018](@cite)

"""
module SurfaceFluxes

using NonlinearSolvers
using KernelAbstractions: @print

using Thermodynamics
using DocStringExtensions
using CLIMAParameters: AbstractEarthParameterSet
using CLIMAParameters.Planet: molmass_ratio, grav
using CLIMAParameters.SubgridScale: von_karman_const
using StaticArrays

include("UniversalFunctions.jl")
using .UniversalFunctions
const UF = UniversalFunctions

abstract type SurfaceFluxesModel end

struct FVScheme end
struct DGScheme end

export surface_conditions,
    exchange_coefficients, recover_profile, monin_obukhov_length

"""
    SurfaceFluxConditions{FT}

Surface flux conditions, returned from `surface_conditions`.

# Fields

$(DocStringExtensions.FIELDS)
"""
struct SurfaceFluxConditions{FT, VFT}
    L_MO::FT
    wθ_flux_star::FT
    flux::VFT
    x_star::VFT
    C_exchange::VFT
end

function Base.show(io::IO, sfc::SurfaceFluxConditions)
    println(io, "----------------------- SurfaceFluxConditions")
    println(io, "L_MO           = ", sfc.L_MO)
    println(io, "wθ_flux_star = ", sfc.wθ_flux_star)
    println(io, "flux           = ", sfc.flux)
    println(io, "x_star         = ", sfc.x_star)
    println(io, "C_exchange     = ", sfc.C_exchange)
    println(io, "-----------------------")
end

function surface_fluxes_f!(F, x, nt)
    param_set = nt.param_set
    wθ_flux_star = nt.wθ_flux_star
    z_in = nt.z_in
    z_0 = length(nt.z_0) > 1 ? Tuple(nt.z_0) : nt.z_0
    x_in = Tuple(nt.x_in)
    x_s = Tuple(nt.x_s)
    n_vars = nt.n_vars
    scheme = nt.scheme
    universal_func = nt.universal_func
    θ_scale = nt.θ_scale

    x_tup = Tuple(x)

    u_star, θ_star = x_tup[2], x_tup[3]
    if wθ_flux_star == nothing
        wθ_surf_flux = -u_star * θ_star
    else
        wθ_surf_flux = wθ_flux_star
    end
    L_MO = monin_obukhov_length(param_set, u_star, θ_scale, wθ_surf_flux)
    uf = universal_func(param_set, L_MO)
    F_nt = ntuple(Val(n_vars + 1)) do i
        if i == 1
            F_i =
                x_tup[1] - monin_obukhov_length(
                    param_set,
                    u_star,
                    θ_scale,
                    wθ_surf_flux,
                )
        else
            ϕ = x_tup[i]
            transport = i - 1 == 1 ? MomentumTransport() : HeatTransport()
            F_i =
                ϕ - compute_physical_scale(
                    uf,
                    z_in,
                    length(nt.z_0) > 1 ? z_0[i - 1] : z_0,
                    x_in[i - 1],
                    x_s[i - 1],
                    transport,
                    scheme,
                )
        end
        F_i
    end
    F .= F_nt
end

"""
    surface_conditions
Surface conditions given
 - `MO_param_guess` initial guess for solution (`L_MO, u_star, θ_star, ϕ_star, ...`)
 - `x_in` Inner values for state variable array `x`. They correspond to volume averages in FV and
    nodal point values in DG
 - `x_s` surface values for state variable array `x`
 - `z_0` roughness lengths for state variable array `x`
 - `θ_scale` virtual dry static energy (i.e., basic potential temperature)
 - `z_in` Input height for the similarity functions. It is Δz for FV and the height
    of the first nodal point for DG
 - `wθ_flux_star` potential temperature flux (optional)
  - `universal_func` family of flux-gradient universal functions used (optional)

If `wθ_flux_star` is not given, then it is computed by iteration
of equations 3, 17, and 18 in Nishizawa2018.
"""
function surface_conditions(
    param_set::AbstractEarthParameterSet,
    MO_param_guess::AbstractVector,
    x_in::AbstractVector,
    x_s::AbstractVector,
    z_0::Union{AbstractVector, FT},
    θ_scale::FT,
    z_in::FT,
    scheme,
    wθ_flux_star::Union{Nothing, FT} = nothing,
    universal_func::Union{Nothing, F} = Businger,
) where {FT <: AbstractFloat, AbstractEarthParameterSet, F}

    n_vars = length(MO_param_guess) - 1
    @assert length(MO_param_guess) == n_vars + 1
    @assert length(x_in) == n_vars
    @assert length(x_s) == n_vars
    local sol

    args = (;
        param_set,
        wθ_flux_star,
        z_in,
        z_0,
        n_vars,
        x_in,
        x_s,
        scheme,
        θ_scale,
        MO_param_guess,
        universal_func,
    )

    # Define closure over args
    f!(F, x_all) = surface_fluxes_f!(F, x_all, args)

    nls = NewtonsMethodAD(f!, MO_param_guess)
    # sol = solve!(nls, CompactSolution(), ResidualTolerance(FT(10)), 1)
    sol = solve!(nls, CompactSolution())

    root_tup = Tuple(sol.root)
    if sol.converged
        L_MO, x_star = root_tup[1], SVector(root_tup[2:end])
        u_star, θ_star = x_star[1], x_star[2]
    else
        @print("Warning: Unconverged Surface Fluxes\n")
        L_MO, x_star = root_tup[1], SVector(root_tup[2:end])
        u_star, θ_star = x_star[1], x_star[2]
    end

    _grav::FT = grav(param_set)
    _von_karman_const::FT = von_karman_const(param_set)
    wθ_flux_star = -u_star^3 * θ_scale / (_von_karman_const * _grav * L_MO)
    flux = -u_star * x_star

    C_exchange = get_flux_coefficients(
        param_set,
        z_in,
        x_star,
        x_s,
        L_MO,
        z_0,
        scheme,
        universal_func,
    )

    VFT = typeof(flux)
    return SurfaceFluxConditions{FT, VFT}(
        L_MO,
        wθ_flux_star,
        flux,
        x_star,
        C_exchange,
    )
end

"""
    get_energy_flux(surf_conds::SurfaceFluxConditions{FT, VFT})

Returns the potential temperature and q_tot fluxes needed to
compute the energy flux.
"""
function get_energy_flux(surf_conds::SurfaceFluxConditions)
    θ_flux = surf_conds.flux[2]
    q_tot_flux = surf_conds.flux[3]
    return θ_flux, q_tot_flux
end

"""
    compute_physical_scale(uf, z_in, z_0, x_in, x_s, transport, ::FVScheme)

Returns u_star, θ_star, ... given equations (17), (18) for FV and (8), (9) for DG
from Nishizawa & Kitamura (2018).

## Arguments
 - `uf` universal function family
 - `z_in` Input height for the similarity functions. It is Δz for FV and the height
    of the first nodal point for DG
 - `z_0` roughness lengths for state variable array `x`
 - `x_in` Inner values for state variable array `x`. They correspond to volume averages in FV and
    nodal point values in DG
 - `x_s` surface values for state variable array `x`
 - `transport` Type of transport (momentum or heat)

"""
function compute_physical_scale(
    uf::AbstractUniversalFunction{FT},
    z_in,
    z_0,
    x_in,
    x_s,
    transport,
    ::FVScheme,
) where {FT}
    _von_karman_const::FT = von_karman_const(uf.param_set)
    _π_group = FT(UF.π_group(uf, transport))
    _π_group⁻¹ = (1 / _π_group)
    R_z0 = 1 - z_0 / z_in
    temp1 = log(z_in / z_0)
    temp2 = -Psi(uf, z_in / uf.L, transport)
    temp3 = z_0 / z_in * Psi(uf, z_0 / uf.L, transport)
    temp4 = R_z0 * (psi(uf, z_0 / uf.L, transport) - 1)
    Σterms = temp1 + temp2 + temp3 + temp4
    return _π_group⁻¹ * _von_karman_const / Σterms * (x_in - x_s)
end

function compute_physical_scale(
    uf::AbstractUniversalFunction{FT},
    z_in,
    z_0,
    x_in,
    x_s,
    transport,
    ::DGScheme,
) where {FT}
    _von_karman_const::FT = von_karman_const(uf.param_set)
    _π_group = FT(UF.π_group(uf, transport))
    _π_group⁻¹ = (1 / _π_group)
    temp1 = log(z_in / z_0)
    temp2 = -psi(uf, z_in / uf.L, transport)
    temp3 = psi(uf, z_0 / uf.L, transport)
    Σterms = temp1 + temp2 + temp3
    return _π_group⁻¹ * _von_karman_const / Σterms * (x_in - x_s)
end

"""
    recover_profile(z, x_star, x_s, z_0, L_MO, transport, ::DGScheme, uf)

Recover vertical profiles u(z), θ(z), ... using equations (4) and (5)
for DG and (12), (13) for FV from Nishizawa & Kitamura (2018).

## Arguments
 - `z` Input height for evaluation
 - `x_star` Surface layer scales for state variable array `x`.
 - `x_s` surface values for state variable array `x`
 - `z_0` roughness lengths for state variable array `x`
 - `L_MO` Obukhov length
 - `transport` Type of transport (momentum or heat)
 - `uf` universal function family

"""
function recover_profile(
    param_set::AbstractEarthParameterSet,
    z,
    x_star,
    x_s,
    z_0::Union{AbstractVector, FT},
    L_MO,
    transport,
    ::DGScheme,
    universal_func = Businger,
) where {FT}
    uf = universal_func(param_set, L_MO)
    _von_karman_const::FT = von_karman_const(param_set)
    _π_group = FT(UF.π_group(uf, transport))
    temp1 = log(z / z_0)
    temp2 = -psi(uf, z / uf.L, transport)
    temp3 = psi(uf, z_0 / uf.L, transport)
    Σterms = temp1 + temp2 + temp3
    return _π_group * x_star * Σterms / _von_karman_const + x_s
end

function recover_profile(
    param_set::AbstractEarthParameterSet,
    z,
    x_star,
    x_s,
    z_0::Union{AbstractVector, FT},
    L_MO,
    transport,
    ::FVScheme,
    universal_func = Businger,
) where {FT}
    uf = universal_func(param_set, L_MO)
    _von_karman_const::FT = von_karman_const(param_set)
    _π_group = FT(UF.π_group(uf, transport))
    R_z0 = 1 - z_0 / z
    temp1 = log(z / z_0)
    temp2 = -Psi(uf, z / uf.L, transport)
    temp3 = z_0 / z * Psi(uf, z_0 / uf.L, transport)
    temp4 = R_z0 * (psi(uf, z_0 / uf.L, transport) - 1)
    Σterms = temp1 + temp2 + temp3 + temp4
    return _π_group * x_star * Σterms / _von_karman_const + x_s
end

### Generic terms

"""
    monin_obukhov_length(param_set, u_star, θ_scale, wθ_surf_flux)

Returns the Monin-Obukhov length.
"""
function monin_obukhov_length(
    param_set::AbstractEarthParameterSet,
    u_star,
    θ_scale,
    wθ_surf_flux,
)
    FT = typeof(u_star)
    _grav::FT = grav(param_set)
    _von_karman_const::FT = von_karman_const(param_set)
    return -u_star^3 * θ_scale /
           (_von_karman_const * _grav * wθ_surf_flux + eps(FT))
end

monin_obukhov_length(sfc::SurfaceFluxConditions) = sfc.L_MO

"""
    exchange_coefficients(
        param_set,
        z,
        F_exchange,
        x_star::VFT,
        θ_scale,
        L_MO,
        universal_func,
    )

Computes exchange transfer coefficients
  - `K_exchange` exchange coefficients
"""
function exchange_coefficients(
    param_set,
    z,
    F_exchange,
    x_star::VFT,
    θ_scale,
    L_MO,
    universal_func,
) where {VFT}
    N = length(F_exchange)
    FT = typeof(z)
    _von_karman_const::FT = von_karman_const(param_set)
    uf = universal_func(param_set, L_MO)
    x_star_tup = Tuple(x_star)
    K_exchange = similar(x_star)
    F_exchange_tup = Tuple(F_exchange)
    K_exchange .= ntuple(Val(length(x_star))) do i
        transport = i == 1 ? MomentumTransport() : HeatTransport()
        phi_t = phi(uf, z / L_MO, transport)
        _π_group = FT(UF.π_group(uf, transport))
        num = -F_exchange_tup[i] * _von_karman_const * z
        den = _π_group * (x_star_tup[i] * phi_t)
        K_exch = num / den # Eq. 19 in
    end
    return K_exchange
end

"""
    get_flux_coefficients(
        param_set,
        z,
        x_star::VFT,
        θ_scale,
        L_MO,
        z0,
        universal_func,
    )

Returns the exchange coefficients for bulk transfer formulas associated
with the frictional parameters `x_star` and the surface layer similarity
profiles. Taken from equations (36) and (37) of Byun (1990).
"""
function get_flux_coefficients(
    param_set,
    z_in,
    x_star::VFT,
    x_s,
    L_MO,
    z0::Union{AbstractVector, FT},
    scheme,
    universal_func,
) where {VFT, FT}
    N = length(x_star)
    z0_tup = Tuple(z0)
    x_star_tup = Tuple(x_star)
    x_s_tup = Tuple(x_s)
    u_in = recover_profile(
        param_set,
        z_in,
        x_star_tup[1],
        x_s_tup[1],
        (length(z0) > 1 ? z0_tup[1] : z0),
        L_MO,
        MomentumTransport(),
        scheme,
        universal_func,
    )
    C = similar(x_star)
    C .= ntuple(Val(length(x_star))) do i
        if i == 1
            C_i = x_star[i]^2 / (u_in - x_s_tup[i])^2
        else
            ϕ_in = recover_profile(
                param_set,
                z_in,
                x_star_tup[i],
                x_s_tup[i],
                (length(z0) > 1 ? z0_tup[i] : z0),
                L_MO,
                HeatTransport(),
                scheme,
                universal_func,
            )
            C_i =
                x_star[1] * x_star[i] / (u_in - x_s_tup[1]) /
                (ϕ_in - x_s_tup[i])
        end
        C_i
    end
    return C
end

end # SurfaceFluxes module
