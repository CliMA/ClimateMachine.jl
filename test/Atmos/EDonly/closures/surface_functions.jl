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
    subdomain_surface_values(
        atmos.turbconv.surface,
        atmos.turbconv,
        atmos,
        state,
        aux,
        zLL,
    )
end

function subdomain_surface_values(
    surf::SurfaceModel,
    turbconv::EDMF{FT},
    atmos::AtmosModel{FT},
    state::Vars,
    aux::Vars,
    zLL::FT,
) where {FT}

    turbconv = atmos.turbconv
    gm = state
    # TODO: change to new_thermo_state
    ts = recover_thermo_state(atmos, state, aux)
    q = PhasePartition(ts)
    _cp_m = cp_m(ts)
    lv = latent_heat_vapor(ts)
    Π = exner(ts)
    ρ_inv = 1 / gm.ρ

    θ_liq_surface_flux = surf.shf / Π / _cp_m
    q_tot_surface_flux = surf.lhf / lv
    # these value should be given from the SurfaceFluxes.jl once it is merged
    oblength = turbconv.surface.obukhov_length
    ustar = turbconv.surface.ustar

    unstable = oblength < 0
    fact = unstable ? (1 - surf.ψϕ_stab * zLL / oblength)^(-FT(2 // 3)) : 1
    tke_fact = unstable ? cbrt(zLL / oblength * zLL / oblength) : 0
    ustar² = ustar^2
    θ_liq_cv = 4 * (θ_liq_surface_flux * θ_liq_surface_flux) / (ustar²) * fact
    q_tot_cv = 4 * (q_tot_surface_flux * q_tot_surface_flux) / (ustar²) * fact
    θ_liq_q_tot_cv =
        4 * (θ_liq_surface_flux * q_tot_surface_flux) / (ustar²) * fact
    tke = ustar² * (surf.κ_star² + tke_fact)

    e_int = internal_energy(atmos, state, aux)
    ts_new = new_thermo_state(atmos, state, aux)
    θ_liq = liquid_ice_pottemp(ts_new)

    ρq_tot = atmos.moisture isa DryModel ? FT(0) : gm.moisture.ρq_tot

    return (;
        θ_liq_cv,
        q_tot_cv,
        θ_liq_q_tot_cv,
        tke,
    )
end;
