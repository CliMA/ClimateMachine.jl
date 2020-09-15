#### EDMF model
using DocStringExtensions

#### Entrainment-Detrainment model

"""
    EntrainmentDetrainment

An Entrainment-Detrainment model for EDMF, containing
all related model and free parameters.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct EntrainmentDetrainment{FT <: AbstractFloat}
    "Entrainment TKE scale"
    c_λ::FT = 0.3
    "Entrainment factor"
    c_ε::FT = 0.13
    "Detrainment factor"
    c_δ::FT = 0.52
    "Turbulent Entrainment factor"
    c_t::FT = 0.1
    "Detrainment RH power"
    β::FT = 2
    "Logistic function scale ‵[1/s]‵"
    μ_0::FT = 0.0004
    "Updraft mixing fraction"
    χ::FT = 0.25
    "Minimum updraft velocity"
    w_min::FT = 0.1
    "Exponential area limiter scale"
    lim_ϵ::FT = 0.0001
    "Exponential area limiter amplitude"
    lim_amp::FT = 10
    "Minimum value for turb entr"
    εt_min::FT = 1e-4
end

"""
    SubdomainModel

A subdomain model for EDMF, containing
all related model and free parameters.

TODO: `a_max` is valid for all subdomains,
    but it is insufficient to ensure `a_en`
    is not negative. Limits can be imposed
    for updrafts, but this is a limit
    is dictated by 1 - Σᵢ aᵢ, which must be
    somehow satisfied by regularizing prognostic
    source terms.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct SubdomainModel{FT <: AbstractFloat}
    "Minimum area fraction for any subdomain"
    a_min::FT
    "Maximum area fraction for any subdomain"
    a_max::FT
end

function SubdomainModel(
    ::Type{FT},
    N_up;
    a_min::FT = 0.001,
    a_max::FT = 1 - N_up * a_min,
) where {FT}
    return SubdomainModel(; a_min = a_min, a_max = a_max)
end

"""
    SurfaceModel

A surface model for EDMF, containing all boundary
values and parameters needed by the model.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct SurfaceModel{FT <: AbstractFloat, SV}
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
    "Area"
    a::FT
    "Scalar coefficient"
    scalar_coeff::SV = 0
    "Friction velocity"
    ustar::FT = 0.28
    "Monin - Obukhov length"
    obukhov_length::FT = -100
    "Surface covariance stability coefficient"
    ψϕ_stab::FT = 8.3
    "Square ratio of rms turbulent velocity to friction velocity"
    κ_star²::FT = 3.75
    "Height of the lowest level"
    zLL::FT = 60
end

"""
    SurfaceModel{FT}(N_up) where {FT}

Constructor for `SurfaceModel` for EDMF, given:
 - `N_up`, the number of updrafts
"""
function SurfaceModel{FT}(N_up;) where {FT}
    a_surf::FT = 0.1

    surface_scalar_coeff = SVector(
        ntuple(N_up) do i
            percentile_bounds_mean_norm(
                1 - a_surf + (i - 1) * FT(a_surf / N_up),
                1 - a_surf + i * FT(a_surf / N_up),
                1000,
            )
        end,
    )
    SV = typeof(surface_scalar_coeff)
    return SurfaceModel{FT, SV}(;
        scalar_coeff = surface_scalar_coeff,
        a = a_surf,
    )
end


"""
    PressureModel

A pressure model for EDMF, containing
all related model and free parameters.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct PressureModel{FT <: AbstractFloat}
    "Pressure drag"
    α_d::FT = 10.0
    "Pressure advection"
    α_a::FT = 0.1
    "Pressure buoyancy"
    α_b::FT = 0.12
    "Default updraft height"
    H_up::FT = 500
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
    c_d::FT = 0.22
    "Eddy Viscosity"
    c_m::FT = 0.14
    "Static Stability coefficient"
    c_b::FT = 0.63
    "Empirical stability function coefficient"
    a1::FT = 0.2
    "Empirical stability function coefficient"
    a2::FT = 100
    "Von Karmen constant"
    κ::FT = 0.4
    "Maximum mixing length"
    max_length::FT = 1e6
    "Prandtl number empirical coefficient"
    ω_pr::FT = 53.0 / 13.0
    "Prandtl number scale"
    Pr_n::FT = 1 #0.74
    "Critical Richardson number"
    Ri_c::FT = 0.25
    "Random small number variable that should be addressed"
    random_minval::FT = 1e-9
    "smooth minimum's fractional upper bound"
    smin_ub::FT = 0.1
    "smooth minimum's regularization minimum"
    smin_rm::FT = 1.5
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
    Updraft <: BalanceLaw
A `BalanceLaw` for the updraft subdomains arising in EDMF.
"""
Base.@kwdef struct Updraft{FT <: AbstractFloat} <: BalanceLaw end

"""
    EDMF <: TurbulenceConvectionModel

A turbulence convection model for the EDMF
scheme, containing all closure models and
free parameters.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct EDMF{FT <: AbstractFloat, N, UP, EN, ED, P, S, MP, ML, SD} <:
                   TurbulenceConvectionModel
    "Updrafts"
    updraft::UP
    "Environment"
    environment::EN
    "Entrainment-Detrainment model"
    entr_detr::ED
    "Pressure model"
    pressure::P
    "Surface model"
    surface::S
    "Microphysics model"
    micro_phys::MP
    "Mixing length model"
    mix_len::ML
    "Subdomain model"
    subdomains::SD
end

"""
    EDMF(
        FT, N_up, N_quad;
        updraft = ntuple(i -> Updraft{FT}(), N_up),
        environment = Environment{FT, N_quad}(),
        entr_detr = EntrainmentDetrainment{FT}(),
        pressure = PressureModel{FT}(),
        surface = SurfaceModel{FT}(N_up),
        micro_phys = MicrophysicsModel(FT),
        mix_len = MixingLengthModel{FT}(),
        subdomain = SubdomainModel(FT, N_up),
    )
Constructor for `EDMF` subgrid-scale scheme, given:
 - `FT`, the AbstractFloat type used
 - `N_up`, the number of updrafts
 - `N_quad`, the quadrature order. `N_quad^2` is
        the total number of quadrature points
        used for environmental distributions.
 - `updraft`, a tuple containing N_up updraft BalanceLaws
 - `environment`, the environment BalanceLaw
 - `entr_detr`, an `EntrainmentDetrainment` model
 - `pressure`, a `PressureModel`
 - `surface`, a `SurfaceModel`
 - `micro_phys`, a `MicrophysicsModel`
 - `mix_len`, a `MixingLengthModel`
 - `subdomain`, a `SubdomainModel`
"""
function EDMF(
    FT,
    N_up,
    N_quad;
    updraft = ntuple(i -> Updraft{FT}(), N_up),
    environment = Environment{FT, N_quad}(),
    entr_detr = EntrainmentDetrainment{FT}(),
    pressure = PressureModel{FT}(),
    surface = SurfaceModel{FT}(N_up),
    micro_phys = MicrophysicsModel(FT),
    mix_len = MixingLengthModel{FT}(),
    subdomain = SubdomainModel(FT, N_up),
)
    args = (
        updraft,
        environment,
        entr_detr,
        pressure,
        surface,
        micro_phys,
        mix_len,
        subdomain,
    )
    return EDMF{FT, N_up, typeof.(args)...}(args...)
end


import ClimateMachine.TurbulenceConvection: turbconv_sources, turbconv_bcs

"""
    EDMFBCs <: TurbConvBC

Boundary conditions for EDMF.
"""
struct EDMFBCs <: TurbConvBC end
n_updrafts(m::EDMF{FT, N_up}) where {FT, N_up} = N_up
n_updrafts(m::TurbulenceConvectionModel) = 0
turbconv_filters(m::TurbulenceConvectionModel) = ()
turbconv_filters(m::EDMF) = (
    "turbconv.environment.ρatke",
    "turbconv.environment.ρaθ_liq_cv",
    "turbconv.environment.ρaq_tot_cv",
    "turbconv.updraft",
)
n_quad_points(m::Environment{FT, N_quad}) where {FT, N_quad} = N_quad
turbconv_sources(m::EDMF) = (turbconv_source!,)
turbconv_bcs(::EDMF) = EDMFBCs()
