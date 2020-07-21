"""
    SurfaceFluxes

  Surface flux functions, e.g., for buoyancy flux,
  friction velocity, and exchange coefficients.

## Sub-modules
  - module Byun1990
  - module Nishizawa2018

## Interface
  - [`compute_buoyancy_flux`](@ref) computes the buoyancy flux
  - In addition, each sub-module has the following functions:
    - `monin_obukhov_len` computes the Monin-Obukhov length
    - `compute_friction_velocity` computes the friction velocity
    - `compute_exchange_coefficients` computes the exchange coefficients

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

using RootSolvers
using ..Thermodynamics
using CLIMAParameters
using CLIMAParameters.Planet: molmass_ratio, grav

"""
    compute_buoyancy_flux(param_set, shf, lhf, T_b, q, α_0)

Computes buoyancy flux given
 - `shf` sensible heat flux
 - `lhf` latent heat flux
 - `T_b` surface boundary temperature
 - `q` phase partition (see [`PhasePartition`](@ref))
 - `α_0` specific volume
"""
function compute_buoyancy_flux(
    param_set::AbstractParameterSet,
    shf,
    lhf,
    T_b,
    q,
    α_0,
)
    FT = typeof(shf)
    _molmass_ratio::FT = molmass_ratio(param_set)
    _grav::FT = grav(param_set)
    cp_ = cp_m(param_set, q)
    lv = latent_heat_vapor(param_set, T_b)
    temp1 = (_molmass_ratio - 1)
    temp2 = (shf + temp1 * cp_ * T_b * lhf / lv)
    return (_grav * α_0 / cp_ / T_b * temp2)
end

module Byun1990

using RootSolvers
using ...Thermodynamics
using CLIMAParameters
using CLIMAParameters.SubgridScale: von_karman_const

""" Computes ψ_m for stable case. See Eq. 12 """
ψ_m_stable(ζ, ζ_0, β_m) = -β_m * (ζ - ζ_0)

""" Computes ψ_h for stable case. See Eq. 13 """
ψ_h_stable(ζ, ζ_0, β_h) = -β_h * (ζ - ζ_0)

""" Computes ψ_m for unstable case. See Eq. 14 """
function ψ_m_unstable(ζ, ζ_0, γ_m)
    x(ζ) = sqrt(sqrt(1 - γ_m * ζ))
    temp(ζ, ζ_0) = log1p((ζ - ζ_0) / (1 + ζ_0))  # log((1 + ζ)/(1 + ζ_0))
    ψ_m = (
        2 * temp(x(ζ), x(ζ_0)) + temp(x(ζ)^2, x(ζ_0)^2) - 2 * atan(x(ζ)) +
        2 * atan(x(ζ_0))
    )
    return ψ_m
end

""" Computes ψ_h for unstable case. See Eq. 15 """
function ψ_h_unstable(ζ, ζ_0, γ_h)
    y(ζ) = sqrt(1 - γ_h * ζ)
    ψ_h = 2 * log1p((y(ζ) - y(ζ_0)) / (1 + y(ζ_0))) # log((1 + y(ζ))/(1 + y(ζ_0)))
    return ψ_h
end

"""
    monin_obukhov_len(param_set, u, flux)

Computes the Monin-Obukhov length (Eq. 3)
"""
function monin_obukhov_len(param_set::AbstractParameterSet, u, flux)
    FT = typeof(u)
    _von_karman_const::FT = von_karman_const(param_set)
    return -u^3 / (flux * _von_karman_const)
end

"""
    compute_friction_velocity(param_set, flux, z_0, z_1, β_m, γ_m, tol_abs, iter_max)

Computes roots of friction velocity equation (Eq. 10)

`u_ave = u_* ( ln(z/z_0) - ψ_m(z/L, z_0/L) ) /κ`
"""
function compute_friction_velocity(
    param_set::AbstractParameterSet,
    u_ave::FT,
    flux::FT,
    z_0::FT,
    z_1::FT,
    β_m::FT,
    γ_m::FT,
    tol_abs::FT,
    iter_max::IT,
) where {FT, IT}

    _von_karman_const = FT(von_karman_const(param_set))

    ustar_0 = u_ave * _von_karman_const / log(z_1 / z_0)
    ustar = ustar_0
    let u_ave = u_ave,
        flux = flux,
        z_0 = z_0,
        z_1 = z_1,
        β_m = β_m,
        γ_m = γ_m,
        param_set = param_set

        # use neutral condition as first guess
        stable = z_1 / monin_obukhov_len(param_set, ustar_0, flux) >= 0
        function compute_ψ_m(u)
            L_MO = monin_obukhov_len(param_set, u, flux)
            ζ = z_1 / L_MO
            ζ_0 = z_0 / L_MO
            return stable ? ψ_m_stable(ζ, ζ_0, β_m) : ψ_m_unstable(ζ, ζ_0, γ_m)
        end
        function compute_u_ave_over_ustar(u)
            _von_karman_const = FT(von_karman_const(param_set))
            return (log(z_1 / z_0) - compute_ψ_m(u)) / _von_karman_const # Eq. 10
        end
        compute_ustar(u) = u_ave / compute_u_ave_over_ustar(u)

        if (abs(flux) > 0)
            ustar_1 = compute_ustar(ustar_0)
            sol = RootSolvers.find_zero(
                u -> u_ave - u * compute_u_ave_over_ustar(u),
                SecantMethod(ustar_0, ustar_1),
                CompactSolution(),
                SolutionTolerance(tol_abs),
                iter_max,
            )
            ustar = sol.root
        end

    end

    return ustar
end

"""
    compute_exchange_coefficients(param_set, Ri, z_b, z_0, γ_m, γ_h, β_m, β_h, Pr_0)

Computes exchange transfer coefficients:
 - `C_D`  momentum exchange coefficient      (Eq. 36)
 - `C_H`  thermodynamic exchange coefficient (Eq. 37)
 - `L_mo` Monin-Obukhov length               (re-arranged Eq. 3)

TODO: `Pr_0` should come from CLIMAParameters
"""
function compute_exchange_coefficients(
    param_set::AbstractParameterSet,
    Ri::FT,
    z_b::FT,
    z_0::FT,
    γ_m::FT,
    γ_h::FT,
    β_m::FT,
    β_h::FT,
    Pr_0::FT,
) where {FT}
    logz = log(z_b / z_0)
    zfactor = z_b / (z_b - z_0) * logz
    s_b = Ri / Pr_0
    _von_karman_const::FT = von_karman_const(param_set)
    if Ri > 0
        temp = ((1 - 2 * β_h * Ri) - sqrt(1 + 4 * (β_h - β_m) * s_b))
        ζ = zfactor / (2 * β_h * (β_m * Ri - 1)) * temp                    # Eq. 19
        L_mo = z_b / ζ                                             # LHS of Eq. 3
        ζ_0 = z_0 / L_mo
        ψ_m = ψ_m_stable(ζ, ζ_0, β_m)
        ψ_h = ψ_h_stable(ζ, ζ_0, β_h)
    else
        Q_b = FT(1 / 9) * (1 / (γ_m^2) + 3 * γ_h / γ_m * s_b^2)           # Eq. 31
        P_b = FT(1 / 54) * (-2 / (γ_m^3) + 9 / γ_m * (-γ_h / γ_m + 3) * s_b^2) # Eq. 32
        crit = Q_b^3 - P_b^2
        if crit < 0
            T_b = cbrt(sqrt(-crit) + abs(P_b))                     # Eq. 34
            ζ = zfactor * (1 / (3 * γ_m) - (T_b + Q_b / T_b))              # Eq. 29
        else
            θ_b = acos(P_b / sqrt(Q_b^3))                            # Eq. 33
            ζ = zfactor * (-2 * sqrt(Q_b) * cos(θ_b / 3) + 1 / (3 * γ_m))  # Eq. 28
        end
        L_mo = z_b / ζ                                             # LHS of Eq. 3
        ζ_0 = z_0 / L_mo
        ψ_m = ψ_m_unstable(ζ, ζ_0, γ_m)
        ψ_h = ψ_h_unstable(ζ, ζ_0, γ_h)
    end
    cu = _von_karman_const / (logz - ψ_m)            # Eq. 10, solved for u^*
    cth = _von_karman_const / (logz - ψ_h) / Pr_0    # Eq. 11, solved for h^*
    C_D = cu^2                                       # Eq. 36
    C_H = cu * cth                                   # Eq. 37
    return C_D, C_H, L_mo
end

end # Byun1990 module

module Nishizawa2018
using RootSolvers
using ...Thermodynamics
using CLIMAParameters
using CLIMAParameters.Planet: grav
using CLIMAParameters.SubgridScale: von_karman_const

""" Computes `R_z0` expression, defined after Eq. 15 """
compute_R_z0(z_0, Δz) = 1 - z_0 / Δz

""" Computes `f_m` in Eq. A7 """
compute_f_m(ζ) = sqrt(sqrt(1 - 15 * ζ))

""" Computes `f_h` in Eq. A8 """
compute_f_h(ζ) = sqrt(1 - 9 * ζ)

""" Computes `ψ_m` in Eq. A3 """
function compute_ψ_m(ζ, L, a)
    FT = typeof(ζ)
    f_m = compute_f_m(ζ)
    return L >= 0 ? -a * ζ :
           log((1 + f_m)^2 * (1 + f_m^2) / 8) - 2 * atan(f_m) + FT(π / 2)
end

""" Computes `ψ_h` in Eq. A4 """
compute_ψ_h(ζ, L, a, Pr) =
    L >= 0 ? -a * ζ / Pr : 2 * log((1 + compute_f_h(ζ)) / 2)

""" Computes `Ψ_m` in Eq. A5 """
function compute_Ψ_m(ζ, L, a, tol)
    FT = typeof(ζ)
    if ζ < tol
        return ζ >= 0 ? -a * ζ / 2 : -FT(15) * ζ / FT(8) # Computes Ψ_m in Eq. A13
    else
        f_m = compute_f_m(ζ)
        # Note that "1-f^3" in is a typo, it is supposed to be "1-f_m^3".
        # This was confirmed by communication with the author.
        return L >= 0 ? -a * ζ / 2 :
               log((1 + f_m)^2 * (1 + f_m^2) / 8) - 2 * atan(f_m) + FT(π / 2) -
               1 + (1 - f_m^3) / (12 * ζ)
    end
end

""" Computes `Ψ_h` in Eq. A6 """
function compute_Ψ_h(ζ, L, a, Pr, tol)
    FT = typeof(ζ)
    if ζ < tol
        return ζ >= 0 ? -a * ζ / (2 * Pr) : -9 * ζ / 4 # Computes Ψ_h in Eq. A14
    else
        f_h = compute_f_h(ζ)
        return L >= 0 ? -a * ζ / (2 * Pr) :
               2 * log((1 + f_h) / 2) + 2 * (1 - f_h) / (9 * ζ)
    end
end

"""
    monin_obukhov_len(param_set, u, θ, flux)

Computes Monin-Obukhov length. Eq. 3
"""
function monin_obukhov_len(param_set::AbstractParameterSet, u, θ, flux)
    FT = typeof(u)
    _von_karman_const::FT = von_karman_const(param_set)
    _grav::FT = grav(param_set)
    return -u^3 * θ / (_von_karman_const * _grav * flux)
end

"""
    compute_friction_velocity(param_set, u_ave, θ, flux, Δz, z_0, a, Ψ_m_tol, tol_abs, iter_max)

Computes friction velocity, in Eq. 12 in, by solving the
non-linear equation:

`u_ave = ustar/κ * ( ln(Δz/z_0) - Ψ_m(Δz/L) + z_0/Δz * Ψ_m(z_0/L) + R_z0 [ψ_m(z_0/L) - 1] )`

where `L` is a non-linear function of `ustar` (see [`monin_obukhov_len`](@ref)).
"""
function compute_friction_velocity(
    param_set::AbstractParameterSet,
    u_ave::FT,
    θ::FT,
    flux::FT,
    Δz::FT,
    z_0::FT,
    a::FT,
    Ψ_m_tol::FT,
    tol_abs::FT,
    iter_max::IT,
) where {FT, IT}
    _von_karman_const::FT = von_karman_const(param_set)
    ustar_0 = u_ave * _von_karman_const / log(Δz / z_0)
    ustar = ustar_0
    let u_ave = u_ave,
        _von_karman_const = _von_karman_const,
        θ = θ,
        flux = flux,
        Δz = Δz,
        z_0 = z_0,
        a = a,
        Ψ_m_tol = Ψ_m_tol,
        tol_abs = tol_abs,
        iter_max = iter_max
        # Note the lowercase psi (ψ) and uppercase psi (Ψ):
        Ψ_m_closure(ζ, L) = compute_Ψ_m(ζ, L, a, Ψ_m_tol)
        ψ_m_closure(ζ, L) = compute_ψ_m(ζ, L, a)
        function compute_u_ave_over_ustar(u)
            L = monin_obukhov_len(param_set, u, θ, flux)
            R_z0 = compute_R_z0(z_0, Δz)
            temp1 = log(Δz / z_0)
            temp2 = -Ψ_m_closure(Δz / L, L)
            temp3 = z_0 / Δz * Ψ_m_closure(z_0 / L, L)
            temp4 = R_z0 * (ψ_m_closure(z_0 / L, L) - 1)
            return (temp1 + temp2 + temp3 + temp4) / _von_karman_const
        end
        compute_ustar(u) = u_ave / compute_u_ave_over_ustar(u)
        ustar_1 = compute_ustar(ustar_0)
        sol = RootSolvers.find_zero(
            u -> u_ave - u * compute_u_ave_over_ustar(u),
            SecantMethod(ustar_0, ustar_1),
            CompactSolution(),
            SolutionTolerance(tol_abs),
            iter_max,
        )
        ustar = sol.root
    end
    return ustar
end

"""
    compute_exchange_coefficients(param_set, z, F_m, F_h, a, u_star, θ, flux, Pr)

Computes exchange transfer coefficients:

  - `K_D`  momentum exchange coefficient
  - `K_H`  thermodynamic exchange coefficient
  - `L_mo` Monin-Obukhov length

TODO: `Pr` should come from CLIMAParameters
"""
function compute_exchange_coefficients(
    param_set::AbstractParameterSet,
    z::FT,
    F_m::FT,
    F_h::FT,
    a::FT,
    u_star::FT,
    θ::FT,
    flux::FT,
    Pr::FT,
) where {FT}

    _von_karman_const::FT = von_karman_const(param_set)
    L_mo = monin_obukhov_len(param_set, u_star, θ, flux)
    ψ_m = compute_ψ_m(z / L_mo, L_mo, a)
    ψ_h = compute_ψ_h(z / L_mo, L_mo, a, Pr)

    K_m = -F_m * _von_karman_const * z / (u_star * ψ_m) # Eq. 19
    K_h = -F_h * _von_karman_const * z / (Pr * θ * ψ_h) # Eq. 20

    return K_m, K_h, L_mo
end

end # Nishizawa2018 module

end # SurfaceFluxes module
