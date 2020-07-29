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

@article{nishizawa2018surface,
  title={A Surface Flux Scheme Based on the Monin-Obukhov Similarity for Finite Volume Models},
  author={Nishizawa, S and Kitamura, Y},
  journal={Journal of Advances in Modeling Earth Systems},
  volume={10},
  number={12},
  pages={3159--3175},
  year={2018},
  publisher={Wiley Online Library}
}

@article{byun1990analytical,
  title={On the analytical solutions of flux-profile relationships for the atmospheric surface layer},
  author={Byun, Daewon W},
  journal={Journal of Applied Meteorology},
  volume={29},
  number={7},
  pages={652--657},
  year={1990}
}
"""
module SurfaceFluxes

using ..Thermodynamics
using KernelAbstractions: @print
using DocStringExtensions
using NLsolve
using CLIMAParameters: AbstractParameterSet
using CLIMAParameters
using CLIMAParameters.Planet: molmass_ratio, grav
using CLIMAParameters.SubgridScale: von_karman_const

const APS = AbstractParameterSet
abstract type SurfaceFluxesModel end

struct Momentum end
struct Heat end

export surface_conditions

"""
    SurfaceFluxConditions{FT, VFT}

Surface flux conditions, returned from `surface_conditions`.

# Fields

$(DocStringExtensions.FIELDS)
"""
struct SurfaceFluxConditions{FT, VFT}
    L_MO::FT
    pottemp_flux_star::FT
    flux::VFT
    x_star::VFT
    K_exchange::VFT
end

"""
    surface_conditions

Surface conditions given
 - `x_initial` initial guess for solution (`L_MO, u_star, θ_star, ϕ_star, ...`)
 - `x_ave` volume-averaged value for variable `x`
 - `x_s` surface value for variable `x`
 - `z_0` roughness length for variable `x`
 - `F_exchange` flux at the top for variable `x`
 - `dimensionless_number` dimensionless turbulent transport coefficient:
      - Momentum: 1
      - Heat: Turbulent Prantl number at neutral stratification
      - Mass: Turbulent Schmidt number
      - ...
 - `θ_bar` basic potential temperature
 - `Δz` layer thickness (not spatial discretization)
 - `z` coordinate axis
 - `a` free model parameter with prescribed value of 4.7
 - `pottemp_flux_given` potential temperature flux (optional)

If `pottemp_flux` is not given, then it is computed by iteration
of equations 3, 17, and 18 in Nishizawa2018.
"""
function surface_conditions(
    param_set::APS,
    x_initial::Vector{FT},
    x_ave::Vector{FT},
    x_s::Vector{FT},
    z_0::Vector{FT},
    F_exchange::Vector{FT},
    dimensionless_number::Vector{FT},
    θ_bar::FT,
    Δz::FT,
    z::FT,
    a::FT,
    pottemp_flux_given::Union{Nothing, FT} = nothing,
) where {FT <: AbstractFloat, APS <: AbstractParameterSet}

    n_vars = length(x_initial) - 1
    @assert length(x_initial) == n_vars + 1
    @assert length(x_ave) == n_vars
    @assert length(x_s) == n_vars
    @assert length(z_0) == n_vars
    @assert length(F_exchange) == n_vars
    @assert length(dimensionless_number) == n_vars
    local sol
    let param_set = param_set
        function f!(F, x_all)
            L_MO, x_vec = x_all[1], x_all[2:end]
            u, θ = x_vec[1], x_vec[2]
            pottemp_flux =
                pottemp_flux_given == nothing ? -u * θ : pottemp_flux_given
            F[1] = L_MO - monin_obukhov_len(param_set, u, θ_bar, pottemp_flux)
            for i in 1:n_vars
                ϕ = x_vec[i]
                transport = i == 1 ? Momentum() : Heat()
                F[i + 1] =
                    ϕ - compute_physical_scale(
                        param_set,
                        ϕ,
                        u,
                        Δz,
                        a,
                        θ_bar,
                        pottemp_flux,
                        z_0[i],
                        x_ave[i],
                        x_s[i],
                        dimensionless_number[i],
                        transport,
                    )
            end
        end
        sol = nlsolve(f!, x_initial, autodiff = :forward)
    end
    if converged(sol)
        L_MO, x_star = sol.zero[1], sol.zero[2:end]
        u_star, θ_star = x_star[1], x_star[2]
    else
        L_MO, x_star = sol.zero[1], sol.zero[2:end]
        u_star, θ_star = x_star[1], x_star[2]
        @print("Non-converged surface fluxes")
    end

    _grav::FT = grav(param_set)
    _von_karman_const::FT = von_karman_const(param_set)
    pottemp_flux_star = -u_star^3 * θ_bar / (_von_karman_const * _grav * L_MO)
    flux = -u_star * x_star
    K_exchange = compute_exchange_coefficients(
        param_set,
        z,
        F_exchange,
        a,
        x_star,
        θ_bar,
        dimensionless_number,
        L_MO,
    )

    return SurfaceFluxConditions(
        L_MO,
        pottemp_flux_star,
        flux,
        x_star,
        K_exchange,
    )
end


function compute_physical_scale(
    param_set::APS,
    x,
    u,
    Δz,
    a,
    θ_bar,
    pottemp_flux,
    z_0,
    x_ave,
    x_s,
    dimensionless_number,
    transport,
)
    FT = typeof(u)
    _von_karman_const::FT = von_karman_const(param_set)
    L = monin_obukhov_len(param_set, u, θ_bar, pottemp_flux)
    R_z0 = compute_R_z0(z_0, Δz)
    term1 = log(Δz / z_0)
    term2 = -compute_Psi(Δz / L, L, a, dimensionless_number, transport)
    term3 =
        z_0 / Δz * compute_Psi(z_0 / L, L, a, dimensionless_number, transport)
    term4 =
        R_z0 * (compute_psi(z_0 / L, L, a, dimensionless_number, transport) - 1)
    return (1 / dimensionless_number) * _von_karman_const /
           (term1 + term2 + term3 + term4) * (x_ave - x_s)
end

""" Computes `R_z0` expression, defined after Eq. 15 """
compute_R_z0(z_0, Δz) = 1 - z_0 / Δz

""" Computes f_m in Eq. A7 """
compute_f_m(ζ) = sqrt(sqrt(1 - 15 * ζ))

""" Computes f_h in Eq. A8 """
compute_f_h(ζ) = sqrt(1 - 9 * ζ)

""" Computes psi_m in Eq. A3 """
function compute_psi(ζ, L, a, dimensionless_number, ::Momentum)
    f_m = compute_f_m(ζ)
    return L >= 0 ? -a * ζ :
           log((1 + f_m)^2 * (1 + f_m^2) / 8) - 2 * atan(f_m) + FT(π) / 2
end

""" Computes psi_h in Eq. A4 """
function compute_psi(ζ, L, a, dimensionless_number, ::Heat)
    L >= 0 ? -a * ζ / dimensionless_number : 2 * log((1 + compute_f_h(ζ)) / 2)
end

""" Computes Psi_m in Eq. A5 """
function compute_Psi(ζ, L, a, dimensionless_number, ::Momentum)
    FT = typeof(L)
    if abs(ζ) < eps(FT)
        return ζ >= 0 ? -a * ζ / 2 : -FT(15) * ζ / FT(8) # Computes Psi_m in Eq. A13
    else
        f_m = compute_f_m(ζ)
        # Note that "1-f^3" in is a typo, it is supposed to be "1-f_m^3".
        # This was confirmed by communication with the author.
        return L >= 0 ? -a * ζ / 2 :
               log((1 + f_m)^2 * (1 + f_m^2) / 8) - 2 * atan(f_m) + FT(π) / 2 -
               1 + (1 - f_m^3) / (12 * ζ)
    end
end

""" Computes Psi_h in Eq. A6 """
function compute_Psi(ζ, L, a, dimensionless_number, ::Heat)
    if abs(ζ) < eps(typeof(L))
        return ζ >= 0 ? -a * ζ / (2 * dimensionless_number) : -9 * ζ / 4 # Computes Psi_h in Eq. A14
    else
        f_h = compute_f_h(ζ)
        return L >= 0 ? -a * ζ / (2 * dimensionless_number) :
               2 * log((1 + f_h) / 2) + 2 * (1 - f_h) / (9 * ζ)
    end
end

"""
    monin_obukhov_len(param_set, u, θ, flux)

Monin-Obukhov length. Eq. 3
"""
function monin_obukhov_len(param_set::APS, u, θ_bar, flux)
    FT = typeof(u)
    _grav::FT = grav(param_set)
    _von_karman_const::FT = von_karman_const(param_set)
    return -u^3 * θ_bar / (_von_karman_const * _grav * flux)
end

"""
    compute_exchange_coefficients(z, F_exchange, a, x_star, θ_bar, dimensionless_number, L_MO)

Computes exchange transfer coefficients

  - `K_exchange` exchange coefficients
"""
function compute_exchange_coefficients(
    param_set,
    z,
    F_exchange,
    a,
    x_star,
    θ_bar,
    dimensionless_number,
    L_MO,
)

    N = length(F_exchange)
    FT = typeof(z)
    K_exchange = Vector{FT}(undef, N)
    _von_karman_const::FT = von_karman_const(param_set)
    for i in 1:N
        transport = i == 1 ? Momentum() : Heat()
        psi = compute_psi(z / L_MO, L_MO, a, dimensionless_number[i], transport)
        K_exchange[i] =
            -F_exchange[i] * _von_karman_const * z / (x_star[i] * psi) # Eq. 19 in
    end

    return K_exchange
end

end # SurfaceFluxes module
