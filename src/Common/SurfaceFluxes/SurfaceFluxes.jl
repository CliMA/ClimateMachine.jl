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

"""
module SurfaceFluxes

using RootSolvers
using NLsolve

using ..Thermodynamics
using DocStringExtensions
using CLIMAParameters: AbstractEarthParameterSet
using CLIMAParameters.Planet: molmass_ratio, grav
using CLIMAParameters.SubgridScale: von_karman_const
using StaticArrays

abstract type SurfaceFluxesModel end

struct Momentum end
struct Heat end

export surface_conditions, 
    compute_R_z0,
    compute_exchange_coefficients,
    compute_f_h,
    compute_f_m,
    compute_Psi,
    compute_psi,
    Momentum,
    Heat

"""
    SurfaceFluxConditions{FT}

Surface flux conditions, returned from `surface_conditions`.

# Fields

$(DocStringExtensions.FIELDS)
"""
struct SurfaceFluxConditions{FT}
    L_MO::FT
    VDSE_flux_star::FT
    flux::AbstractVector
    x_star::AbstractVector
    K_exchange::AbstractVector
end

function Base.show(io::IO, sfc::SurfaceFluxConditions)
    println(io, "----------------------- SurfaceFluxConditions")
    println(io, "L_MO              = ", sfc.L_MO)
    println(io, "VDSE_flux_star = ", sfc.VDSE_flux_star)
    println(io, "flux              = ", sfc.flux)
    println(io, "x_star            = ", sfc.x_star)
    println(io, "K_exchange        = ", sfc.K_exchange)
    println(io, "-----------------------")
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
 - `VDSE_scale` basic potential temperature
 - `Δz` layer thickness (not spatial discretization)
 - `z` coordinate axis
 - `a` free model parameter with prescribed value of 4.7
 - `VDSE_flux_star_given` potential temperature flux (optional)

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
    dimensionless_number::AbstractVector,
    VDSE_scale::FT,
    qt_bar::FT,
    Δz::FT,
    z::FT,
    a::FT,
    VDSE_flux_star_given::Union{Nothing, FT} = nothing,
) where {FT <: AbstractFloat, AbstractEarthParameterSet}

    n_vars = length(x_initial) - 1
    @assert length(x_initial) == n_vars + 1
    @assert length(x_ave) == n_vars
    @assert length(x_s) == n_vars
    @assert length(z_0) == n_vars
    @assert length(F_exchange) == n_vars
    @assert length(dimensionless_number) == n_vars
    local sol
    u, θ, qt = x_initial[1], x_initial[2], x_initial[3]
    let param_set = param_set
        function f!(F, x_all)
            L_MO, x_vec = x_all[1], x_all[2:end]
            u, θ = x_vec[1], x_vec[2]
            VDSE_flux_star =
                VDSE_flux_star_given == nothing ? -u * θ : VDSE_flux_star_given
            flux = VDSE_flux_star
            F[1] = L_MO - monin_obukhov_length(param_set, u, VDSE_scale, flux)
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
                        VDSE_scale,
                        flux,
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
        L_MO, x_star = sol.zero[1], collect(sol.zero[2:end])
        u_star, θ_star = x_star[1], x_star[2]
    else
        #println("Warning: Unconverged Surface Fluxes")
        L_MO, x_star = sol.zero[1], sol.zero[2:end]
        u_star, θ_star = x_star[1], x_star[2]
    end

    _grav::FT = grav(param_set)
    _von_karman_const::FT = von_karman_const(param_set)
    VDSE_flux_star = -u_star^3 * VDSE_scale / (_von_karman_const * _grav * L_MO)
    flux = -u_star * x_star

    K_exchange = compute_exchange_coefficients(
        param_set,
        z,
        F_exchange,
        a,
        x_star,
        VDSE_scale,
        dimensionless_number,
        L_MO,
    )
    
    C_exchange = get_flux_coefficients(
        param_set,
        z,
        a,
        x_star,
        VDSE_scale,
        dimensionless_number,
        L_MO,
        z_0
    )

    #=
    @show(C_exchange)
    @show(L_MO)
    @show(x_star[1], x_star[2])
    @show(x_star .* x_star[1])
    =# 
    return SurfaceFluxConditions(
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
    param_set::AbstractEarthParameterSet,
    x,
    u,
    Δz,
    a,
    VDSE_scale,
    flux,
    z_0,
    x_ave,
    x_s,
    dimensionless_number,
    transport,
)
    FT = typeof(u)
    _von_karman_const::FT = von_karman_const(param_set)
    L = monin_obukhov_length(param_set, u, VDSE_scale, flux)
    R_z0 = compute_R_z0(z_0, Δz)
    temp1 = log(Δz / z_0)
    temp2 = -compute_Psi(Δz / L, L, a, dimensionless_number, transport)
    temp3 =
        z_0 / Δz * compute_Psi(z_0 / L, L, a, dimensionless_number, transport)
    temp4 =
        R_z0 * (compute_psi(z_0 / L, L, a, dimensionless_number, transport) - 1)
    return (1 / dimensionless_number) * _von_karman_const /
           (temp1 + temp2 + temp3 + temp4) * (x_ave - x_s)
end

function compute_physical_scale2(
    param_set::AbstractEarthParameterSet,
    x,
    u,
    Δz,
    a,
    VDSE_scale,
    flux,
    z_0,
    x_ave,
    x_s,
    dimensionless_number,
    transport,
)
    FT = eltype(u)
    _von_karman_const::FT = von_karman_const(param_set)
    L = monin_obukhov_length(param_set, u, VDSE_scale, flux)
    temp1 = log(Δz / z_0)
    temp2 = -compute_psi(Δz/L, L, a, dimensionless_number, transport)
    temp3 = compute_psi(z_0/L, L, a, dimensionless_number, transport)
    return (1 / dimensionless_number) * _von_karman_const /
        (temp1 + temp2 + temp3) * (x_ave - x_s)
end


### Nishizawa equations ### 

""" Computes R_z0 expression, defined after Eq. 15 """
compute_R_z0(z_0, Δz) = 1 - z_0 / Δz

""" Computes f_m in Eq. A7 """
compute_f_m(ζ) = sqrt(sqrt(1 - 15 * ζ))

""" Computes f_h in Eq. A8 """
compute_f_h(ζ) = sqrt(1 - 9 * ζ)

""" Computes phi_m in Eq. A1 """
function compute_phi(ζ, L, a, dimensionless_number, ::Momentum)
    FT = eltype(ζ)
    return L >= 0 ? a * ζ + 1 : 1/(sqrt(sqrt((1 - 15 * ζ))))
end

""" Computes phi_h in Eq. A2 """
function compute_phi(ζ, L, a, dimensionless_number, ::Heat)
    FT = eltype(ζ)
    return L >= 0 ? a * ζ / dimensionless_number + 1 : 1/(sqrt((1 - 9 * ζ)))
end

""" Computes psi_m in Eq. A3 """
function compute_psi(ζ, L, a, dimensionless_number, ::Momentum)
    FT = eltype(L)
    f_m = compute_f_m(ζ)
    return L >= 0 ? -a * ζ :
           log((1 + f_m)^2 * (1 + f_m^2) / 8) - 2 * atan(f_m) + FT(π) / 2
end

""" Computes psi_h in Eq. A4 """
function compute_psi(ζ, L, a, dimensionless_number, ::Heat)
    f_h = compute_f_h(ζ)
    L >= 0 ? -a * ζ / dimensionless_number : 2 * log((1 + f_h) / 2)
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
               log((1 + f_m)^2 * (1 + f_m^2) / 8) - 2 * atan(f_m) + FT(π) / 2 - 1 +
               (1 - f_m^3) / (12 * ζ)
    end
end

""" Computes Psi_h in Eq. A6 """
function compute_Psi(ζ, L, a, dimensionless_number, ::Heat)
    if abs(ζ) < eps(typeof(L))
        return ζ >= 0 ? -a * ζ / (2 * dimensionless_number) : -9 * ζ / 4 # Computes Psi_h in Eq. A14
    else
        f_h = compute_f_h(ζ)
        return L >= 0 ? -a * ζ / (2 * dimensionless_number) :
               2 * log((1 + f_h) / 2) + 2 * (1 - f_h) / (9 * ζ) - 1
    end
end

### Gryanik et. al. 2020 equations ### 
# Typical values for empirical coefficients
# Pr_0 = 0.98
# a_m = 5.0
# a_h = 5.0
# b_m = 0.3
# b_h = 0.4

"""
    compute_phi_GLGS(ζ, L, a, dimensionless_number, ::Momentum)
Compute ϕ_m, stability function following Gryanik et al. 2020
"""
function compute_phi_GLGS(ζ, L, a, dimensionless_number, ::Momentum)
    if 0 <= ζ < 100 
        return 1 + (a_m * ζ)/(1 + b_m * ζ)^(2/3)
    end
end

"""
    compute_phi_GLGS(ζ, L, a, dimensionless_number, ::Heat)
Compute ϕ_h, stability function following Gryanik et al. 2020
"""
function compute_phi_GLGS(ζ, L, a, dimensionless_number, ::Heat)
    if 0 <= ζ < 100 
        return Pr_0 * (1 + (a_h * ζ)/(1 + b_h * ζ))
    end
end

"""
    compute_psi_GLGS(ζ, L, a, dimensionless_number, ::Momentum)
Compute ψ_m, stability correction function following Gryanik et al. 2020
"""
function compute_psi_GLGS(ζ, L, a, dimensionless_number, ::Momentum)
    if 0 <= ζ < 100
        return -3 * (a_m/b_m) * ((1 + b_m * ζ)^(1/3) - 1)
    end
end

"""
    compute_psi_GLGS(ζ, L, a, dimensionless_number, ::Momentum)
Compute ψ_h, stability correction function following Gryanik et al. 2020
"""
function compute_psi_GLGS(ζ, L, a, dimensionless_number, ::Heat)
    if 0 <= ζ < 100
        return -Pr_0 * (a_h / b_h) * log(1 + b_h * ζ) 
    end
end

"""
    compute_bulk_Richardson(ζ, z_0, L, Pr_0)
Compute Bulk Richardson number, equal to Ri/Pr_0.
Required for computation of exchange coefficients C_d and C_h
"""
function compute_bulk_Richardson(ζ, z_0, L, Pr_0)
    term1 = (1-1/ε_m)^2/(1-1/ε_t)*ζ
    term2 = log(ε_t) - 1/Pr_0 * (compute_psi_GLGS(ζ, L, a, dimensionless_number, Heat())+compute_psi_GLGS(ζ/ε_t, L, a, dimensionless_number, Heat()))
    term3 = log(ε_m) - compute_psi_GLGS(ζ, L, a, dimensionless_number, Momentum() + compute_psi_GLGS(ζ/ε_m, L, a, dimensionless_number, ))
    return term1 + term2/term3
end

### Generic terms

"""
    monin_obukhov_length(param_set, u, θ, flux)

Monin-Obukhov length. Eq. 3
"""
function monin_obukhov_length(param_set::AbstractEarthParameterSet, u_star, VDSE_scale, flux)
    FT = typeof(u_star)
    _grav::FT = grav(param_set)
    _von_karman_const::FT = von_karman_const(param_set)
    return -u_star^3 * VDSE_scale / (_von_karman_const * _grav * flux)
end

"""
    compute_exchange_coefficients(z, F_exchange, a, x_star, VDSE_scale, dimensionless_number, L_MO)

Computes exchange transfer coefficients
  - `K_exchange` exchange coefficients
"""
function compute_exchange_coefficients(
    param_set,
    z,
    F_exchange,
    a,
    x_star,
    VDSE_scale,
    dimensionless_number,
    L_MO,
)
    N = length(F_exchange)
    FT = typeof(z)
    _von_karman_const::FT = von_karman_const(param_set)
    K_exchange = zeros(length(x_star))
    for i in 1:N
        transport = i == 1 ? Momentum() : Heat()
        phi = compute_phi(z / L_MO, L_MO, a, dimensionless_number[i], transport)
        K_exchange[i] =
        -F_exchange[i] * _von_karman_const * z / dimensionless_number[i] / (x_star[i] * phi) # Eq. 19 in
    end
    return K_exchange
end

function get_flux_coefficients(
    param_set,
    z,
    a,
    x_star,
    VDSE_scale,
    dimensionless_number,
    L_MO,
    z0
)
    N = length(x_star)
    L = L_MO
    FT = typeof(z)
    _von_karman_const::FT = von_karman_const(param_set)
    C = zeros(length(x_star))
    for i in 1:N
        if i == 1
            psi = compute_psi(z/L, L, a, dimensionless_number, Momentum())
            C[i] = _von_karman_const^2/(log(z/z0[i] - psi))^2
        else
            psi_m = compute_psi(z/L, L, a, dimensionless_number, Momentum())
            psi_h = compute_psi(z/L, L, a, dimensionless_number, Heat())
            C[i] = _von_karman_const^2/(log(z/z0[i] - psi_m))/(log(z/z0[i] - psi_h))
        end
    end
    return (C)

end

end # SurfaceFluxes module
