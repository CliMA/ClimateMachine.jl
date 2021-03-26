#### Surface model kernels

using Statistics

"""
    subdomain_surface_values(
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

 - `atmos`, an `AtmosModel`
 - `state`, state variables
 - `aux`, auxiliary variables
 - `zLL`, height of the lowest nodal level
"""
function subdomain_surface_values(
    atmos::AtmosModel,
    state::Vars,
    aux::Vars,
    zLL,
)
    turbconv = turbconv_model(atmos)
    subdomain_surface_values(turbconv.surface, turbconv, atmos, state, aux, zLL)
end

function subdomain_surface_values(
    surf::SurfaceModel,
    turbconv::EDMF{FT},
    atmos::AtmosModel{FT},
    state::Vars,
    aux::Vars,
    zLL::FT,
) where {FT}

    turbconv = turbconv_model(atmos)
    N_up = n_updrafts(turbconv)
    gm = state
    # TODO: change to new_thermo_state
    ts = recover_thermo_state(atmos, state, aux)
    q = PhasePartition(ts)
    _cp_m = cp_m(ts)
    lv = latent_heat_vapor(ts)
    Π = exner(ts)
    ρ_inv = 1 / gm.ρ
    upd_surface_std = turbconv.surface.upd_surface_std

    θ_liq_surface_flux = surf.shf / Π / _cp_m
    q_tot_surface_flux = surf.lhf / lv
    # these value should be given from the SurfaceFluxes.jl once it is merged
    oblength = turbconv.surface.obukhov_length
    ustar = turbconv.surface.ustar

    unstable = oblength < 0
    fact = unstable ? (1 - surf.ψϕ_stab * zLL / oblength)^(-FT(2 // 3)) : 1
    tke_fact = unstable ? cbrt(zLL / oblength * zLL / oblength) : 0
    ustar² = ustar^2
    θ_liq_cv = FT(0)
    q_tot_cv = FT(0)
    θ_liq_q_tot_cv = FT(0)
    tke = FT(0.4) #ustar² * (surf.κ_star² + tke_fact)

    a_up_surf = ntuple(i -> FT(surf.a / N_up), N_up)
    e_int = internal_energy(atmos, state, aux)
    ts_new = new_thermo_state(atmos, state, aux)
    θ_liq = liquid_ice_pottemp(ts_new)

    θ_liq_up_surf = ntuple(N_up) do i
        θ_liq + upd_surface_std[i] * sqrt(max(θ_liq_cv, 0))
    end

    ρq_tot = atmos.moisture isa DryModel ? FT(0) : gm.moisture.ρq_tot
    q_tot_up_surf = ntuple(N_up) do i
        ρq_tot * ρ_inv + upd_surface_std[i] * sqrt(max(q_tot_cv, 0))
    end

    return (;
        a_up_surf,
        θ_liq_up_surf,
        q_tot_up_surf,
        θ_liq_cv,
        q_tot_cv,
        θ_liq_q_tot_cv,
        tke,
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
