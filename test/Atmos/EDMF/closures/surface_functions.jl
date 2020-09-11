#### Surface model kernels

using Statistics

"""
    subdomain_surface_values(
        m::SurfaceModel,
        turbconv::EDMF{FT},
        atmos::AtmosModel{FT},
        state::Vars,
        aux::Vars,
        zLL::FT,
    ) where {FT}

Returns the surface values of updraft area fraction, updraft
liquid water potential temperature (`θ_liq`), updraft total
water specific humidity (`q_tot`), environmental variances of
`θ_liq` and `q_tot`, environmental covariance of `θ_liq` with
`q_tot`, and environmental TKE, given:
 - `m`, a `SurfaceModel`
 - `turbconv`, an `EDMF` model
 - `atmos`, an `AtmosModel`
 - `state`, state variables
 - `aux`, auxiliary variables
 - `zLL`, height of the lowest nodal level
"""
function subdomain_surface_values(
    m::SurfaceModel,
    turbconv::EDMF{FT},
    atmos::AtmosModel{FT},
    state::Vars,
    aux::Vars,
    zLL::FT,
) where {FT}

    turbconv = atmos.turbconv
    N_up = n_updrafts(turbconv)
    gm = state
    en = state.turbconv.environment
    up = state.turbconv.updraft
    # TODO: change to new_thermo_state
    ts_gm = recover_thermo_state(atmos, state, aux)
    q = PhasePartition(ts_gm)
    _cp_m = cp_m(ts_gm)
    lv = latent_heat_vapor(ts_gm)
    Π = exner(ts_gm)
    ρinv = 1 / gm.ρ
    surface_scalar_coeff = turbconv.surface.scalar_coeff

    θ_liq_surface_flux = m.surface_shf / Π / _cp_m
    q_tot_surface_flux = m.surface_lhf / lv
    # these value should be given from the SurfaceFluxes.jl once it is merged
    oblength = turbconv.surface.obukhov_length
    ustar = turbconv.surface.ustar

    unstable = oblength < 0
    fact = unstable ? (1 - m.ψϕ_stab * zLL / oblength)^(-FT(2 // 3)) : 1
    tke_fact = unstable ? cbrt(zLL / oblength * zLL / oblength) : 0
    ustar² = ustar^2
    θ_liq_cv = 4 * (θ_liq_surface_flux * θ_liq_surface_flux) / (ustar²) * fact
    q_tot_cv = 4 * (q_tot_surface_flux * q_tot_surface_flux) / (ustar²) * fact
    θ_liq_q_tot_cv =
        4 * (θ_liq_surface_flux * q_tot_surface_flux) / (ustar²) * fact
    tke = ustar² * (m.κ_star² + tke_fact)

    upd_a_surf = ntuple(i -> FT(m.a_surf / N_up), N_up)
    e_int = internal_energy(atmos, state, aux)
    ts_gm_new = new_thermo_state(atmos, state, aux)
    gm_θ_liq = liquid_ice_pottemp(ts_gm_new)

    upd_θ_liq_surf = ntuple(N_up) do i
        gm_θ_liq + surface_scalar_coeff[i] * sqrt(max(θ_liq_cv, 0))
    end

    upd_q_tot_surf = ntuple(N_up) do i
        gm.moisture.ρq_tot * ρinv + surface_scalar_coeff[i] * sqrt(max(q_tot_cv, 0))
    end

    return (
        upd_a_surf = upd_a_surf,
        upd_θ_liq_surf = upd_θ_liq_surf,
        upd_q_tot_surf = upd_q_tot_surf,
        θ_liq_cv = θ_liq_cv,
        q_tot_cv = q_tot_cv,
        θ_liq_q_tot_cv = θ_liq_q_tot_cv,
        tke = tke,
    )
end;

"""
    percentile_bounds_mean_norm(
        low_percentile::FT,
        high_percentile::FT,
        n_samples::Int,
    ) where {FT <: AbstractFloat}

Returns the mean of all instances of a standard Gaussian random
variable that have a CDF higher than low_percentile and lower
than high_percentile, given a total of n_samples of the standard
Gaussian, given:
 - `low_percentile`, lower limit of the CDF
 - `high_percentile`, higher limit of the CDF
 - `n_samples`, the total number of samples drawn from the Gaussian
"""
function percentile_bounds_mean_norm(
    low_percentile::FT,
    high_percentile::FT,
    n_samples::Int,
) where {FT <: AbstractFloat}
    x = rand(Normal(), n_samples)
    xp_low = quantile(Normal(), low_percentile)
    xp_high = quantile(Normal(), high_percentile)
    filter!(y -> xp_low < y < xp_high, x)
    return Statistics.mean(x)
end
