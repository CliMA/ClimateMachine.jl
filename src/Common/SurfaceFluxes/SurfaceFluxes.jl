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

using ..Thermodynamics
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

export surface_conditions, exchange_coefficients

"""
    SurfaceFluxConditions{FT}

Surface flux conditions, returned from `surface_conditions`.

# Fields

$(DocStringExtensions.FIELDS)
"""
struct SurfaceFluxConditions{FT, VFT}
    L_MO::FT
    VDSE_flux_star::FT
    flux::VFT
    x_star::VFT
    K_exchange::VFT
end

function Base.show(io::IO, sfc::SurfaceFluxConditions)
    println(io, "----------------------- SurfaceFluxConditions")
    println(io, "L_MO           = ", sfc.L_MO)
    println(io, "VDSE_flux_star = ", sfc.VDSE_flux_star)
    println(io, "flux           = ", sfc.flux)
    println(io, "x_star         = ", sfc.x_star)
    println(io, "K_exchange     = ", sfc.K_exchange)
    println(io, "-----------------------")
end

function surface_fluxes_f!(F, x, nt)
    param_set = nt.param_set
    VDSE_flux_star = nt.VDSE_flux_star
    Δz = nt.Δz
    z_0 = Tuple(nt.z_0)
    x_ave = Tuple(nt.x_ave)
    x_s = Tuple(nt.x_s)
    n_vars = nt.n_vars
    scheme = nt.scheme
    VDSE_scale = nt.VDSE_scale

    x_tup = Tuple(x)

    u, θ = x_tup[2], x_tup[3]
    if VDSE_flux_star == nothing
        flux = -u * θ
    else
        flux = VDSE_flux_star
    end
    L_MO = monin_obukhov_length(param_set, u, VDSE_scale, flux)
    uf = Businger(param_set, L_MO)
    F_nt = ntuple(Val(n_vars + 1)) do i
        if i == 1
            F_i =
                x_tup[1] - monin_obukhov_length(param_set, u, VDSE_scale, flux)
        else
            ϕ = x_tup[i]
            transport = i - 1 == 1 ? MomentumTransport() : HeatTransport()
            F_i =
                ϕ - compute_physical_scale(
                    uf,
                    Δz,
                    z_0[i - 1],
                    x_ave[i - 1],
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
 - `x_initial` initial guess for solution (`L_MO, u_star, θ_star, ϕ_star, ...`)
 - `x_ave` volume-averaged value for variable `x`
 - `x_s` surface value for variable `x`
 - `z_0` roughness length for variable `x`
 - `F_exchange` flux at the top for variable `x`
 - `VDSE_scale` virtual dry static energy (i.e., basic potential temperature)
 - `Δz` layer thickness (not spatial discretization)
 - `z` coordinate axis
 - `a` free model parameter with prescribed value of 4.7
 - `VDSE_flux_star` potential temperature flux (optional)

If `VDSE_flux_star` is not given, then it is computed by iteration
of equations 3, 17, and 18 in Nishizawa2018.
"""
function surface_conditions(
    param_set::AbstractEarthParameterSet,
    x_initial::AbstractVector,
    x_ave::AbstractVector,
    x_s::AbstractVector,
    z_0::AbstractVector,
    F_exchange::AbstractVector,
    VDSE_scale::FT,
    qt_bar::FT,
    Δz::FT,
    z::FT,
    a::FT,
    scheme,
    VDSE_flux_star::Union{Nothing, FT} = nothing,
) where {FT <: AbstractFloat, AbstractEarthParameterSet}

    n_vars = length(x_initial) - 1
    @assert length(x_initial) == n_vars + 1
    @assert length(x_ave) == n_vars
    @assert length(x_s) == n_vars
    @assert length(z_0) == n_vars
    @assert length(F_exchange) == n_vars
    local sol

    args = (
        param_set = param_set,
        VDSE_flux_star = VDSE_flux_star,
        Δz = Δz,
        z_0 = z_0,
        n_vars = n_vars,
        x_ave = x_ave,
        x_s = x_s,
        scheme = scheme,
        VDSE_scale = VDSE_scale,
        x_initial = x_initial,
    )

    # Define closure over args
    f!(F, x_all) = surface_fluxes_f!(F, x_all, args)

    nls = NewtonsMethodAD(f!, x_initial)
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
    VDSE_flux_star = -u_star^3 * VDSE_scale / (_von_karman_const * _grav * L_MO)
    flux = -u_star * x_star

    K_exchange = exchange_coefficients(
        param_set,
        z,
        F_exchange,
        a,
        x_star,
        VDSE_scale,
        L_MO,
    )

    C_exchange =
        get_flux_coefficients(param_set, z, a, x_star, VDSE_scale, L_MO, z_0)

    VFT = typeof(flux)
    return SurfaceFluxConditions{FT, VFT}(
        L_MO,
        VDSE_flux_star,
        flux,
        x_star,
        C_exchange,
    )
end

"""
    compute_physical_scale
    (3), (17), (18), (19)
"""
function compute_physical_scale(
    uf::AbstractUniversalFunction{FT},
    Δz,
    z_0,
    x_ave,
    x_s,
    transport,
    ::FVScheme,
) where {FT}
    _von_karman_const::FT = von_karman_const(uf.param_set)
    _π_group = FT(UF.π_group(uf, transport))
    _π_group⁻¹ = (1 / _π_group)
    R_z0 = 1 - z_0 / Δz
    temp1 = log(Δz / z_0)
    temp2 = -Psi(uf, Δz / uf.L, transport)
    temp3 = z_0 / Δz * Psi(uf, z_0 / uf.L, transport)
    temp4 = R_z0 * (psi(uf, z_0 / uf.L, transport) - 1)
    Σterms = temp1 + temp2 + temp3 + temp4
    return _π_group⁻¹ * _von_karman_const / Σterms * (x_ave - x_s)
end

function compute_physical_scale(
    uf::AbstractUniversalFunction{FT},
    Δz,
    z_0,
    x_ave,
    x_s,
    transport,
    ::DGScheme,
) where {FT}
    _von_karman_const::FT = von_karman_const(uf.param_set)
    _π_group = FT(UF.π_group(uf, transport))
    _π_group⁻¹ = (1 / _π_group)
    temp1 = log(Δz / z_0)
    temp2 = -psi(uf, Δz / uf.L, transport)
    temp3 = psi(uf, z_0 / uf.L, transport)
    Σterms = temp1 + temp2 + temp3
    return _π_group⁻¹ * _von_karman_const / Σterms * (x_ave - x_s)
end

### Generic terms

"""
    monin_obukhov_length(param_set, u, θ, flux)

Monin-Obukhov length. Eq. 3
"""
function monin_obukhov_length(
    param_set::AbstractEarthParameterSet,
    u_star,
    VDSE_scale,
    flux,
)
    FT = typeof(u_star)
    _grav::FT = grav(param_set)
    _von_karman_const::FT = von_karman_const(param_set)
    return -u_star^3 * VDSE_scale / (_von_karman_const * _grav * flux)
end

"""
    exchange_coefficients(z, F_exchange, a, x_star, VDSE_scale, L_MO)

Computes exchange transfer coefficients
  - `K_exchange` exchange coefficients
"""
function exchange_coefficients(
    param_set,
    z,
    F_exchange,
    a,
    x_star::VFT,
    VDSE_scale,
    L_MO,
) where {VFT}
    N = length(F_exchange)
    FT = typeof(z)
    _von_karman_const::FT = von_karman_const(param_set)
    uf = Businger(param_set, L_MO)
    x_star_tup = Tuple(x_star)
    K_exchange = similar(F_exchange)
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

function get_flux_coefficients(
    param_set,
    z,
    a,
    x_star::VFT,
    VDSE_scale,
    L_MO,
    z0,
) where {VFT}
    N = length(x_star)
    FT = typeof(z)
    _von_karman_const::FT = von_karman_const(param_set)
    uf = Businger(param_set, L_MO)
    psi_m = psi(uf, z / L_MO, MomentumTransport())
    psi_h = psi(uf, z / L_MO, HeatTransport())
    z0_tup = Tuple(z0)
    C = similar(x_star)
    C .= ntuple(Val(length(x_star))) do i
        logζ_ψ_m = log(z / z0_tup[i] - psi_m)
        logζ_ψ_h = log(z / z0_tup[i] - psi_h)
        if i == 1
            C_i = _von_karman_const^2 / logζ_ψ_m^2
        else
            C_i = _von_karman_const^2 / logζ_ψ_m / logζ_ψ_h
        end
        C_i
    end
    return C
end

end # SurfaceFluxes module
