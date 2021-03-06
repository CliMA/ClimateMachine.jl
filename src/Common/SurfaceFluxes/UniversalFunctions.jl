"""
    UniversalFunctions

Universal stability and stability correction
functions for `SurfaceFluxes` module.
"""
module UniversalFunctions

using DocStringExtensions
using CLIMAParameters
using CLIMAParameters.SurfaceFluxes.UniversalFunctions

# Transport Types
export MomentumTransport, HeatTransport

# Types
export AbstractUniversalFunction, Gryanik, Grachev, Businger

# Methods
export phi, psi, Psi

const FTypes = Union{Real, AbstractArray}

abstract type AbstractUniversalFunction{FT <: FTypes} end
const AUF = AbstractUniversalFunction
const APS = AbstractParameterSet

Base.eltype(uf::AbstractUniversalFunction{FT}) where {FT} = FT

#####
##### Interface
#####

abstract type AbstractTransportType end
struct MomentumTransport <: AbstractTransportType end
struct HeatTransport <: AbstractTransportType end

Base.broadcastable(tt::AbstractUniversalFunction) = Ref(tt)
Base.broadcastable(tt::AbstractTransportType) = Ref(tt)

"""
    phi

Universal stability function for wind shear
(ϕ_m) and temperature gradient (ϕ_h)
"""
function phi end

"""
    psi

Universal stability correction function for
momentum (ψ_m) and heat (ψ_h)
"""
function psi end

"""
    Psi

Integral of universal stability correction
function for momentum (ψ_m) and heat (ψ_h)
"""
function Psi end

#####
##### Forwarding methods for free parameters
#####

Pr_0(uf::AUF) = Pr_0(uf.param_set, uf)
a_m(uf::AUF) = a_m(uf.param_set, uf)
a_h(uf::AUF) = a_h(uf.param_set, uf)
b_m(uf::AUF) = b_m(uf.param_set, uf)
b_h(uf::AUF) = b_h(uf.param_set, uf)
c_h(uf::AUF) = c_h(uf.param_set, uf)

π_group(uf::AUF, ::HeatTransport) = Pr_0(uf)
π_group(::AUF, ::MomentumTransport) = 1

#####
##### Businger
#####

"""
    Businger

# Reference

 - [Nishizawa2018](@cite)

# Original research

 - [Businger1971](@cite)

# Equations in reference:

    ϕ_m: Eq. A1
    ϕ_h: Eq. A2
    ψ_m: Eq. A3
    ψ_h: Eq. A4

# Fields

$(DocStringExtensions.FIELDS)
"""
struct Businger{FT, PS} <: AbstractUniversalFunction{FT}
    "Parameter set"
    param_set::PS
    "Monin-Obhukov Length"
    L::FT
    Businger(param_set::PS, L::FT) where {FT, PS} = new{FT, PS}(param_set, L)
end

# CLIMAParameters wrapper
Pr_0(param_set::APS, ::Businger) = Pr_0_Businger(param_set)
a_m(param_set::APS, ::Businger) = a_m_Businger(param_set)
a_h(param_set::APS, ::Businger) = a_h_Businger(param_set)

f_momentum(uf::Businger, ζ) = sqrt(sqrt(1 - 15 * ζ))

f_heat(uf::Businger, ζ) = sqrt(1 - 9 * ζ)

function phi(uf::Businger, ζ, ::MomentumTransport)
    if ζ < 0
        f_m = f_momentum(uf, ζ)
        return 1 / f_m
    else
        FT = eltype(uf)
        _a_m = FT(a_m(uf))
        return _a_m * ζ + 1
    end
end

function phi(uf::Businger, ζ, tt::HeatTransport)
    if ζ < 0
        f_h = f_heat(uf, ζ)
        return 1 / f_h
    else
        FT = eltype(uf)
        _a_h = FT(a_h(uf))
        _π_group = FT(π_group(uf, tt))
        return _a_h * ζ / _π_group + 1
    end
end

function psi(uf::Businger, ζ, ::MomentumTransport)
    FT = eltype(uf.L)
    if ζ < 0
        f_m = f_momentum(uf, ζ)
        log_term = log((1 + f_m)^2 * (1 + f_m^2) / 8)
        return log_term - 2 * atan(f_m) + FT(π) / 2
    else
        _a_m = FT(a_m(uf))
        return -_a_m * ζ
    end
end

function psi(uf::Businger, ζ, tt::HeatTransport)
    if ζ < 0
        f_h = f_heat(uf, ζ)
        return 2 * log((1 + f_h) / 2)
    else
        FT = eltype(uf)
        _a_h = FT(a_h(uf))
        _π_group = FT(π_group(uf, tt))
        return -_a_h * ζ / _π_group
    end
end

function Psi(uf::Businger, ζ, tt::MomentumTransport)
    FT = eltype(uf)
    if abs(ζ) < eps(FT)
        # Psi_m in Eq. A13
        if ζ >= 0
            _a_m = FT(a_m(uf))
            return -_a_m * ζ / 2
        else
            return -FT(15) * ζ / FT(8)
        end
    else
        # Note that "1-f^3" in is a typo, it is
        # supposed to be "1-f_m^3". This was
        # confirmed by communication with the
        # author.
        if ζ < 0
            f_m = f_momentum(uf, ζ)
            log_term = log((1 + f_m)^2 * (1 + f_m^2) / 8)
            π_term = FT(π) / 2
            tan_term = 2 * atan(f_m)
            cubic_term = (1 - f_m^3) / (12 * ζ)
            return log_term - tan_term + π_term - 1 + cubic_term
        else
            _a_m = FT(a_m(uf))
            return -_a_m * ζ / 2
        end
    end
end

function Psi(uf::Businger, ζ, tt::HeatTransport)
    FT = eltype(uf)
    _a_h = FT(a_h(uf))
    if abs(ζ) < eps(typeof(uf.L))
        # Psi_h in Eq. A14
        if ζ >= 0
            _π_group = FT(π_group(uf, tt))
            return -_a_h * ζ / (2 * _π_group)
        else
            return -9 * ζ / 4
        end
    else
        if ζ >= 0
            _π_group = FT(π_group(uf, tt))
            return -_a_h * ζ / (2 * _π_group)
        else
            f_h = f_heat(uf, ζ)
            log_term = 2 * log((1 + f_h) / 2)
            return log_term + 2 * (1 - f_h) / (9 * ζ) - 1
        end
    end
end

#####
##### Gryanik
#####

"""
    Gryanik <: AbstractUniversalFunction{FT}

# References
 - [Gryanik2020](@cite)

# Equations in reference:

    ϕ_m: Eq. 13
    ϕ_h: Eq. 13
    ψ_m: Eq. 14
    ψ_h: Eq. 14

# Fields

$(DocStringExtensions.FIELDS)
"""
struct Gryanik{FT, PS} <: AbstractUniversalFunction{FT}
    "Parameter set"
    param_set::PS
    "Monin-Obhukov Length"
    L::FT
    Gryanik(param_set::PS, L::FT) where {FT, PS} = new{FT, PS}(param_set, L)
end

# CLIMAParameters wrapper
Pr_0(param_set::APS, ::Gryanik) = Pr_0_Gryanik(param_set)
a_m(param_set::APS, ::Gryanik) = a_m_Gryanik(param_set)
a_h(param_set::APS, ::Gryanik) = a_h_Gryanik(param_set)
b_m(param_set::APS, ::Gryanik) = b_m_Gryanik(param_set)
b_h(param_set::APS, ::Gryanik) = b_h_Gryanik(param_set)

function phi(uf::Gryanik, ζ, tt::MomentumTransport)
    if 0 < ζ
        FT = eltype(uf)
        _a_m = FT(a_m(uf))
        _b_m = FT(b_m(uf))
        return 1 + (_a_m * ζ) / (1 + _b_m * ζ)^(FT(2 / 3))
    else
        return phi(Businger(uf), ζ, tt)
    end
end

function phi(uf::Gryanik, ζ, tt::HeatTransport)
    if 0 < ζ
        FT = eltype(uf)
        _Pr_0 = FT(Pr_0(uf))
        _a_h = FT(a_h(uf))
        _b_h = FT(b_h(uf))
        return _Pr_0 * (1 + (_a_h * ζ) / (1 + _b_h * ζ))
    else
        return phi(Businger(uf), ζ, tt)
    end
end

function psi(uf::Gryanik, ζ, tt::MomentumTransport)
    if 0 < ζ
        FT = eltype(uf)
        _a_m = FT(a_m(uf))
        _b_m = FT(b_m(uf))
        return -3 * (_a_m / _b_m) * ((1 + _b_m * ζ)^(FT(1 / 3)) - 1)
    else
        return psi(Businger(uf), ζ, tt)
    end
end

function psi(uf::Gryanik, ζ, tt::HeatTransport)
    if 0 < ζ
        FT = eltype(uf)
        _Pr_0 = FT(Pr_0(uf))
        _a_h = FT(a_h(uf))
        _b_h = FT(b_h(uf))
        return -_Pr_0 * (_a_h / _b_h) * log1p(_b_h * ζ)
    else
        return psi(Businger(uf), ζ, tt)
    end
end


#####
##### Grachev
#####

"""
    Grachev <: AbstractUniversalFunction{FT}

# References
 - [Grachev2007](@cite)

Equations in reference:

    ϕ_m: Eq. 13
    ϕ_h: Eq. 13
    ψ_m: Eq. 14
    ψ_h: Eq. 14

# Fields

$(DocStringExtensions.FIELDS)
"""
struct Grachev{FT, PS} <: AbstractUniversalFunction{FT}
    "Parameter set"
    param_set::PS
    "Monin-Obhukov Length"
    L::FT
    Grachev(param_set::PS, L::FT) where {FT, PS} = new{FT, PS}(param_set, L)
end

# CLIMAParameters wrapper
Pr_0(param_set::APS, ::Grachev) = Pr_0_Grachev(param_set)
a_m(param_set::APS, ::Grachev) = a_m_Grachev(param_set)
a_h(param_set::APS, ::Grachev) = a_h_Grachev(param_set)
b_m(param_set::APS, ::Grachev) = b_m_Grachev(param_set)
b_h(param_set::APS, ::Grachev) = b_h_Grachev(param_set)
c_h(param_set::APS, ::Grachev) = c_h_Grachev(param_set)

function phi(uf::Grachev, ζ, tt::MomentumTransport)
    if 0 < ζ
        FT = eltype(uf)
        _a_m = FT(a_m(uf))
        _b_m = FT(b_m(uf))
        return 1 + _a_m * ζ * (1 + ζ)^FT(1 / 3) / (1 + _b_m * ζ)
    else
        return phi(Businger(uf), ζ, tt)
    end
end

function phi(uf::Grachev, ζ, tt::HeatTransport)
    if 0 < ζ
        FT = eltype(uf)
        _a_h = FT(a_h(uf))
        _b_h = FT(b_h(uf))
        _c_h = FT(c_h(uf))
        return 1 + (_a_h * ζ + _b_h * ζ^2) / (1 + _c_h * ζ + ζ^2)
    else
        return phi(Businger(uf), ζ, tt)
    end
end

function psi(uf::Grachev, ζ, tt::MomentumTransport)
    if 0 < ζ
        FT = eltype(uf)
        _a_m = FT(a_m(uf))
        _b_m = FT(b_m(uf))
        B_m = cbrt(1 / _b_m - 1)
        x = cbrt(1 + ζ)
        sqrt3 = FT(sqrt(3))
        # Note: there is a mismatch between
        # Gryanik, Eq. 26, and Grachev Eq. 12.
        # We use the Grachev Eq. 12.
        linear_term = -3 * (_a_m / _b_m) * (x - 1)
        log_term_1 = 2 * log((x + B_m) / (1 + B_m))
        log_term_2 = log((x^2 - x * B_m + B_m^2) / (1 - B_m + B_m^2))
        atan_term_1 = atan((2 * x - B_m) / (sqrt3 * B_m))
        atan_term_2 = atan((2 - B_m) / (sqrt3 * B_m))
        atan_terms = atan_term_1 - atan_term_2
        bracket_term = log_term_1 - log_term_2 + 2 * sqrt3 * atan_terms
        return linear_term + _a_m * B_m / (2 * _b_m) * bracket_term
    else
        return psi(Businger(uf), ζ, tt)
    end
end

function psi(uf::Grachev, ζ, tt::HeatTransport)
    if 0 < ζ
        FT = eltype(uf)
        _Pr_0 = FT(Pr_0(uf))
        _a_h = FT(a_h(uf))
        _b_h = FT(b_h(uf))
        _c_h = FT(c_h(uf))
        B_h = sqrt(_c_h^2 - 4)
        coeff = _a_h / B_h - _b_h * _c_h / (2 * B_h)
        log_term_1 = log((2 * ζ + _c_h - B_h) / (2 * ζ + _c_h + B_h))
        log_term_2 = log((_c_h - B_h) / (_c_h + B_h))
        log_terms = log_term_1 - log_term_2
        term_2 = _b_h / 2 * log1p(_c_h * ζ + ζ^2)
        return -coeff * log_terms - term_2
    else
        return psi(Businger(uf), ζ, tt)
    end
end

#####
##### Conversions
#####

Businger(uf::Grachev) = Businger(uf.param_set, uf.L)
Businger(uf::Gryanik) = Businger(uf.param_set, uf.L)

Grachev(uf::Businger) = Grachev(uf.param_set, uf.L)
Grachev(uf::Gryanik) = Grachev(uf.param_set, uf.L)

Gryanik(uf::Businger) = Gryanik(uf.param_set, uf.L)
Gryanik(uf::Grachev) = Gryanik(uf.param_set, uf.L)

end # module
