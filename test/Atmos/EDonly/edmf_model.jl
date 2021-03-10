#### EDMF model
using DocStringExtensions
using CLIMAParameters: AbstractEarthParameterSet
using CLIMAParameters.Atmos.EDMF
using CLIMAParameters.SubgridScale

"""
    EntrainmentDetrainment

An Entrainment-Detrainment model for EDMF, containing
all related model and free parameters.

# Fields
$(DocStringExtensions.FIELDS)
"""

"""
    SurfaceModel

A surface model for EDMF, containing all boundary
values and parameters needed by the model.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct SurfaceModel{FT <: AbstractFloat}
    "Surface covariance stability coefficient"
    ψϕ_stab::FT
    "Square ratio of rms turbulent velocity to friction velocity"
    κ_star²::FT
    "Temperature ‵[k]‵"
    T::FT = 300.4
    "Liquid water potential temperature ‵[k]‵"
    θ_liq::FT = 299.1
    "Specific humidity ‵[kg/kg]‵"
    q_tot::FT = 22.45e-3
    "Sensible heat flux ‵[w/m^2]‵"
    shf::FT = 9.5
    "Latent heat flux ‵[w/m^2]‵"
    lhf::FT = 147.2
    "Friction velocity"
    ustar::FT = 0.28
    "Monin - Obukhov length"
    obukhov_length::FT = 0
    "Height of the lowest level"
    zLL::FT = 60
end

"""
    SurfaceModel{FT}(param_set) where {FT}

Constructor for `SurfaceModel` for EDMF, given:
 - `param_set`, an AbstractEarthParameterSet
"""
function SurfaceModel{FT}(param_set::AbstractEarthParameterSet) where {FT}
    κ_star²_ = κ_star²(param_set)
    ψϕ_stab_ = ψϕ_stab(param_set)
    return SurfaceModel{FT}(;
        κ_star² = κ_star²_,
        ψϕ_stab = ψϕ_stab_,
    )
end



"""
    MixingLengthModel

A mixing length model for EDMF, containing
all related model and free parameters.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct MixingLengthModel{FT <: AbstractFloat}
    "dissipation coefficient"
    c_d::FT
    "Eddy Viscosity"
    c_m::FT
    "Static Stability coefficient"
    c_b::FT
    "Empirical stability function coefficient"
    a1::FT
    "Empirical stability function coefficient"
    a2::FT
    "Von Karmen constant"
    κ::FT
    "Prandtl number empirical coefficient"
    ω_pr::FT
    "Prandtl number scale"
    Pr_n::FT
    "Critical Richardson number"
    Ri_c::FT
    "smooth minimum's fractional upper bound"
    smin_ub::FT
    "smooth minimum's regularization minimum"
    smin_rm::FT
    "Maximum mixing length"
    max_length::FT
    "Random small number variable that should be addressed"
    random_minval::FT
end

"""
    MixingLengthModel{FT}(param_set) where {FT}

Constructor for `MixingLengthModel` for EDMF, given:
 - `param_set`, an AbstractEarthParameterSet
"""
function MixingLengthModel{FT}(param_set::AbstractEarthParameterSet) where {FT}
    c_d_ = c_d(param_set)
    c_m_ = c_m(param_set)
    c_b_ = c_b(param_set)
    a1_ = a1(param_set)
    a2_ = a2(param_set)
    κ = von_karman_const(param_set)
    ω_pr_ = ω_pr(param_set)
    Pr_n_ = Pr_n(param_set)
    Ri_c_ = Ri_c(param_set)
    smin_ub_ = smin_ub(param_set)
    smin_rm_ = smin_rm(param_set)
    max_length = 1e6
    random_minval = 1e-9

    args = (
        c_d_,
        c_m_,
        c_b_,
        a1_,
        a2_,
        κ,
        ω_pr_,
        Pr_n_,
        Ri_c_,
        smin_ub_,
        smin_rm_,
        max_length,
        random_minval,
    )

    return MixingLengthModel{FT}(args...)
end

abstract type AbstractStatisticalModel end
struct SubdomainMean <: AbstractStatisticalModel end
struct GaussQuad <: AbstractStatisticalModel end
struct LogNormalQuad <: AbstractStatisticalModel end

"""
    MicrophysicsModel

A microphysics model for EDMF, containing
all related model and free parameters and
assumed subdomain distributions.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct MicrophysicsModel{FT <: AbstractFloat, SM}
    "Subdomain statistical model"
    statistical_model::SM
end

"""
    MicrophysicsModel(
        FT;
        statistical_model = SubdomainMean()
    )

Constructor for `MicrophysicsModel` for EDMF, given:
 - `FT`, the float type used
 - `statistical_model`, the assumed environmental distribution
            of thermodynamic variables.
"""
function MicrophysicsModel(FT; statistical_model = SubdomainMean())
    args = (statistical_model,)
    return MicrophysicsModel{FT, typeof(statistical_model)}(args...)
end

"""
    Environment <: BalanceLaw
A `BalanceLaw` for the environment subdomain arising in EDMF.
"""
Base.@kwdef struct Environment{FT <: AbstractFloat, N_quad} <: BalanceLaw end

"""
    EDMF <: TurbulenceConvectionModel

A turbulence convection model for the EDMF
scheme, containing all closure models and
free parameters.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct EDMF{FT <: AbstractFloat, EN, S, MP, ML} <:
                   TurbulenceConvectionModel
    "Environment"
    environment::EN
    "Surface model"
    surface::S
    "Microphysics model"
    micro_phys::MP
    "Mixing length model"
    mix_len::ML
end

"""
    EDMF(
        FT, N_quad, param_set;
        environment = Environment{FT, N_quad}(),
        surface = SurfaceModel{FT}(param_set),
        micro_phys = MicrophysicsModel(FT),
        mix_len = MixingLengthModel{FT}(param_set),
    )
Constructor for `EDMF` subgrid-scale scheme, given:
 - `FT`, the AbstractFloat type used
 - `N_quad`, the quadrature order. `N_quad^2` is
        the total number of quadrature points
        used for environmental distributions.
 - `environment`, the environment BalanceLaw
 - `surface`, a `SurfaceModel`
 - `micro_phys`, a `MicrophysicsModel`
 - `mix_len`, a `MixingLengthModel`
"""
function EDMF(
    FT,
    N_quad,
    param_set;
    environment = Environment{FT, N_quad}(),
    surface = SurfaceModel{FT}(param_set),
    micro_phys = MicrophysicsModel(FT),
    mix_len = MixingLengthModel{FT}(param_set),
)
    args = (
        environment,
        surface,
        micro_phys,
        mix_len,
    )
    return EDMF{FT, typeof.(args)...}(args...)
end


import ClimateMachine.TurbulenceConvection: turbconv_sources, turbconv_bcs

"""
    EDMFTopBC <: TurbConvBC

Boundary conditions for the top of the EDMF.
"""
struct EDMFTopBC <: TurbConvBC end

"""
    EDMFBottomBC <: TurbConvBC

Boundary conditions for the bottom of the EDMF.
"""
struct EDMFBottomBC <: TurbConvBC end


turbconv_filters(m::TurbulenceConvectionModel) = ()
turbconv_filters(m::EDMF) = (
    "turbconv.environment.ρatke",
    "turbconv.environment.ρaθ_liq_cv",
    "turbconv.environment.ρaq_tot_cv",
)
n_quad_points(m::Environment{FT, N_quad}) where {FT, N_quad} = N_quad
turbconv_sources(m::EDMF) = ()
turbconv_bcs(::EDMF) = (EDMFBottomBC(), EDMFTopBC())