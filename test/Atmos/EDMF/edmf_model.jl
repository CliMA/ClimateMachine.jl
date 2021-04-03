#### EDMF model
using DocStringExtensions
using CLIMAParameters: AbstractEarthParameterSet
using CLIMAParameters.Atmos.EDMF
using CLIMAParameters.SubgridScale
using ClimateMachine.Mesh.Filters: AbstractFilterTarget
"""
    EntrainmentDetrainment

An Entrainment-Detrainment model for EDMF, containing
all related model and free parameters.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct EntrainmentDetrainment{FT <: AbstractFloat}
    "Entrainment TKE scale"
    c_λ::FT
    "Entrainment factor"
    c_ε::FT
    "Detrainment factor"
    c_δ::FT
    "Turbulent Entrainment factor"
    c_t::FT
    "Detrainment RH power"
    β::FT
    "Logistic function scale ‵[1/s]‵"
    μ_0::FT
    "Updraft mixing fraction"
    χ::FT
    "Minimum updraft velocity"
    w_min::FT
    "Exponential area limiter scale"
    lim_ϵ::FT
    "Exponential area limiter amplitude"
    lim_amp::FT
end

"""
    EntrainmentDetrainment{FT}(param_set) where {FT}

Constructor for `EntrainmentDetrainment` for EDMF, given:
 - `param_set`, an AbstractEarthParameterSet
"""
function EntrainmentDetrainment{FT}(
    param_set::AbstractEarthParameterSet,
) where {FT}
    c_λ_ = c_λ(param_set)
    c_ε_ = c_ε(param_set)
    c_δ_ = c_δ(param_set)
    c_t_ = c_t(param_set)
    β_ = β(param_set)
    μ_0_ = μ_0(param_set)
    χ_ = χ(param_set)
    w_min_ = w_min(param_set)
    lim_ϵ_ = lim_ϵ(param_set)
    lim_amp_ = lim_amp(param_set)

    args = (c_λ_, c_ε_, c_δ_, c_t_, β_, μ_0_, χ_, w_min_, lim_ϵ_, lim_amp_)

    return EntrainmentDetrainment{FT}(args...)
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
    a_min::FT = FT(0),
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
    "Area"
    a::FT
    "Surface covariance stability coefficient"
    ψϕ_stab::FT
    "Square ratio of rms turbulent velocity to friction velocity"
    κ_star²::FT
    "Updraft normalized standard deviation at the surface"
    upd_surface_std::SV
    # The following will be deleted after SurfaceFlux coupling
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
    SurfaceModel{FT}(N_up, param_set) where {FT}

Constructor for `SurfaceModel` for EDMF, given:
 - `N_up`, the number of updrafts
 - `param_set`, an AbstractEarthParameterSet
"""
function SurfaceModel{FT}(N_up, param_set::AbstractEarthParameterSet) where {FT}
    a_surf_ = a_surf(param_set)
    κ_star²_ = κ_star²(param_set)
    ψϕ_stab_ = ψϕ_stab(param_set)

    if a_surf_ > FT(0)
        upd_surface_std = SVector(
            ntuple(N_up) do i
                percentile_bounds_mean_norm(
                    1 - a_surf_ + (i - 1) * FT(a_surf_ / N_up),
                    1 - a_surf_ + i * FT(a_surf_ / N_up),
                    1000,
                )
            end,
        )
    else
        upd_surface_std = SVector(ntuple(i -> FT(0), N_up))
    end
    SV = typeof(upd_surface_std)
    return SurfaceModel{FT, SV}(;
        upd_surface_std = upd_surface_std,
        a = a_surf_,
        κ_star² = κ_star²_,
        ψϕ_stab = ψϕ_stab_,
    )
end

"""
    NeutralDrySurfaceModel

A surface model for EDMF simulations in a
dry, neutral environment, containing all boundary
values and parameters needed by the model.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct NeutralDrySurfaceModel{FT <: AbstractFloat}
    "Area"
    a::FT
    "Square ratio of rms turbulent velocity to friction velocity"
    κ_star²::FT
    "Friction velocity"
    ustar::FT = 0.3
    "Height of the lowest level"
    zLL::FT = 60
    "Monin - Obukhov length"
    obukhov_length::FT = 0
end

"""
    NeutralDrySurfaceModel{FT}(N_up, param_set) where {FT}

Constructor for `NeutralDrySurfaceModel` for EDMF, given:
 - `param_set`, an AbstractEarthParameterSet
"""
function NeutralDrySurfaceModel{FT}(
    param_set::AbstractEarthParameterSet,
) where {FT}
    a_surf_ = a_surf(param_set)
    κ_star²_ = κ_star²(param_set)
    return NeutralDrySurfaceModel{FT}(; a = a_surf_, κ_star² = κ_star²_)
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
    α_d::FT
    "Pressure advection"
    α_a::FT
    "Pressure buoyancy"
    α_b::FT
    "Minimum diagnostic updraft height for closures"
    H_up_min::FT
end

"""
    PressureModel{FT}(param_set) where {FT}

Constructor for `PressureModel` for EDMF, given:
 - `param_set`, an AbstractEarthParameterSet
"""
function PressureModel{FT}(param_set::AbstractEarthParameterSet) where {FT}
    α_d_ = α_d(param_set)
    α_a_ = α_a(param_set)
    α_b_ = α_b(param_set)
    H_up_min_ = H_up_min(param_set)

    args = (α_d_, α_a_, α_b_, H_up_min_)

    return PressureModel{FT}(args...)
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
    Updraft <: BalanceLaw
A `BalanceLaw` for the updraft subdomains arising in EDMF.
"""
Base.@kwdef struct Updraft{FT <: AbstractFloat} <: BalanceLaw end

abstract type Coupling end

"""
    Decoupled <: Coupling

Dispatch on decoupled model (default)

 - The EDMF SGS tendencies do not modify the grid-mean equations.
"""
struct Decoupled <: Coupling end

"""
    Coupled <: Coupling

Dispatch on coupled model

 - The EDMF SGS tendencies modify the grid-mean equations.
"""
struct Coupled <: Coupling end

"""
    EDMF <: TurbulenceConvectionModel

A turbulence convection model for the EDMF
scheme, containing all closure models and
free parameters.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct EDMF{
    FT <: AbstractFloat,
    N,
    UP,
    EN,
    ED,
    P,
    S,
    MP,
    ML,
    SD,
    C,
} <: TurbulenceConvectionModel
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
    "Coupling mode"
    coupling::C
end

"""
    EDMF(
        FT, N_up, N_quad, param_set;
        updraft = ntuple(i -> Updraft{FT}(), N_up),
        environment = Environment{FT, N_quad}(),
        entr_detr = EntrainmentDetrainment{FT}(param_set),
        pressure = PressureModel{FT}(param_set),
        surface = SurfaceModel{FT}(N_up, param_set),
        micro_phys = MicrophysicsModel(FT),
        mix_len = MixingLengthModel{FT}(param_set),
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
 - `coupling`, a coupling type
"""
function EDMF(
    FT,
    N_up,
    N_quad,
    param_set;
    updraft = ntuple(i -> Updraft{FT}(), N_up),
    environment = Environment{FT, N_quad}(),
    entr_detr = EntrainmentDetrainment{FT}(param_set),
    pressure = PressureModel{FT}(param_set),
    surface = SurfaceModel{FT}(N_up, param_set),
    micro_phys = MicrophysicsModel(FT),
    mix_len = MixingLengthModel{FT}(param_set),
    subdomain = SubdomainModel(FT, N_up),
    coupling = Decoupled(),
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
        coupling,
    )
    return EDMF{FT, N_up, typeof.(args)...}(args...)
end

Base.@kwdef struct IntrinsicEDMFFilter{A, B <: Bool} <: AbstractFilterTarget
    "The AtmosModel"
    atmos::A
    "filter edmf updraft a"
    ρa::B = false
    "filter edmf updraft w"
    ρaw::B = false
    "filter edmf updraft θ"
    ρaθ_liq::B = false
    "filter edmf updraft Q"
    ρaq_tot::B = false
    "filter edmf environment TKE"
    ρatke::B = false
    "filter edmf environment cv_θ"
    ρaθ_liq_cv::B = false
    "filter edmf environment cv_Q"
    ρaq_tot_cv::B = false
    "filter edmf environment cv_θQ"
    ρaθ_liq_q_tot_cv::B = false
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
turbconv_sources(m::EDMF) = ()
turbconv_bcs(::EDMF) = (EDMFBottomBC(), EDMFTopBC())
