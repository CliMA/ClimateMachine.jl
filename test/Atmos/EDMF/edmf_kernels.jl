#### EDMF model kernels

using CLIMAParameters.Planet: e_int_v0, grav, day, R_d, R_v, molmass_ratio
using Printf
using ClimateMachine.Atmos: nodal_update_auxiliary_state!

using ClimateMachine.BalanceLaws: number_states

using ClimateMachine.MPIStateArrays: MPIStateArray
using ClimateMachine.DGMethods: LocalGeometry, DGModel

import ClimateMachine.BalanceLaws:
    vars_state,
    update_auxiliary_state!,
    init_state_prognostic!,
    flux_first_order!,
    flux_second_order!,
    compute_gradient_argument!,
    compute_gradient_flux!

import ClimateMachine.TurbulenceConvection:
    init_aux_turbconv!,
    turbconv_nodal_update_auxiliary_state!,
    turbconv_boundary_state!,
    turbconv_normal_boundary_flux_second_order!

using ClimateMachine.Thermodynamics: air_pressure, air_density


include(joinpath("helper_funcs", "nondimensional_exchange_functions.jl"))
include(joinpath("helper_funcs", "lamb_smooth_minimum.jl"))
include(joinpath("helper_funcs", "utility_funcs.jl"))
include(joinpath("helper_funcs", "subdomain_statistics.jl"))
include(joinpath("helper_funcs", "diagnose_environment.jl"))
include(joinpath("helper_funcs", "subdomain_thermo_states.jl"))
include(joinpath("helper_funcs", "save_subdomain_temperature.jl"))
include(joinpath("closures", "entr_detr.jl"))
include(joinpath("closures", "pressure.jl"))
include(joinpath("closures", "mixing_length.jl"))
include(joinpath("closures", "turbulence_functions.jl"))
include(joinpath("closures", "surface_functions.jl"))


function vars_state(m::NTuple{N, Updraft}, st::Auxiliary, FT) where {N}
    return Tuple{ntuple(i -> vars_state(m[i], st, FT), N)...}
end

function vars_state(::Updraft, ::Auxiliary, FT)
    @vars(
        buoyancy::FT,
        updraft_top::FT,
        a::FT,
        ε_dyn::FT,
        δ_dyn::FT,
        ε_trb::FT,
        T::FT,
        θ_liq::FT,
        q_tot::FT,
        w::FT,
    )
end

function vars_state(::Environment, ::Auxiliary, FT)
    @vars(T::FT, cld_frac::FT, buoyancy::FT)
end

function vars_state(m::EDMF, st::Auxiliary, FT)
    @vars(
        environment::vars_state(m.environment, st, FT),
        updraft::vars_state(m.updraft, st, FT)
    )
end

function vars_state(::Updraft, ::Prognostic, FT)
    @vars(ρa::FT, ρaw::FT, ρaθ_liq::FT, ρaq_tot::FT,)
end

function vars_state(::Environment, ::Prognostic, FT)
    @vars(ρatke::FT, ρaθ_liq_cv::FT, ρaq_tot_cv::FT, ρaθ_liq_q_tot_cv::FT,)
end

function vars_state(m::NTuple{N, Updraft}, st::Prognostic, FT) where {N}
    return Tuple{ntuple(i -> vars_state(m[i], st, FT), N)...}
end

function vars_state(m::EDMF, st::Prognostic, FT)
    @vars(
        environment::vars_state(m.environment, st, FT),
        updraft::vars_state(m.updraft, st, FT)
    )
end

function vars_state(::Updraft, ::Gradient, FT)
    @vars(w::FT,)
end

function vars_state(::Environment, ::Gradient, FT)
    @vars(
        θ_liq::FT,
        q_tot::FT,
        w::FT,
        tke::FT,
        θ_liq_cv::FT,
        q_tot_cv::FT,
        θ_liq_q_tot_cv::FT,
        θv::FT,
        e::FT,
    )
end

function vars_state(m::NTuple{N, Updraft}, st::Gradient, FT) where {N}
    return Tuple{ntuple(i -> vars_state(m[i], st, FT), N)...}
end

function vars_state(m::EDMF, st::Gradient, FT)
    @vars(
        environment::vars_state(m.environment, st, FT),
        updraft::vars_state(m.updraft, st, FT)
    )
end

function vars_state(m::NTuple{N, Updraft}, st::GradientFlux, FT) where {N}
    return Tuple{ntuple(i -> vars_state(m[i], st, FT), N)...}
end

function vars_state(::Updraft, st::GradientFlux, FT)
    @vars(∇w::SVector{3, FT},)
end

function vars_state(::Environment, ::GradientFlux, FT)
    @vars(
        ∇θ_liq::SVector{3, FT},
        ∇q_tot::SVector{3, FT},
        ∇w::SVector{3, FT},
        ∇tke::SVector{3, FT},
        ∇θ_liq_cv::SVector{3, FT},
        ∇q_tot_cv::SVector{3, FT},
        ∇θ_liq_q_tot_cv::SVector{3, FT},
        ∇θv::SVector{3, FT},
        ∇e::SVector{3, FT},
    )

end

function vars_state(m::EDMF, st::GradientFlux, FT)
    @vars(
        S²::FT, # should be conditionally grabbed from atmos.turbulence
        environment::vars_state(m.environment, st, FT),
        updraft::vars_state(m.updraft, st, FT)
    )
end

"""
    init_aux_turbconv!(
        turbconv::EDMF{FT},
        m::AtmosModel{FT},
        aux::Vars,
        geom::LocalGeometry,
    ) where {FT}

Initialize EDMF auxiliary variables.
"""
function init_aux_turbconv!(
    turbconv::EDMF{FT},
    m::AtmosModel{FT},
    aux::Vars,
    geom::LocalGeometry,
) where {FT}
    N_up = n_updrafts(turbconv)

    # Aliases:
    en_a = aux.turbconv.environment
    up_a = aux.turbconv.updraft

    en_a.cld_frac = FT(0)
    en_a.buoyancy = FT(0)

    @unroll_map(N_up) do i
        up_a[i].buoyancy = FT(0)
        up_a[i].updraft_top = FT(500)
        up_a[i].θ_liq = FT(0)
        up_a[i].q_tot = FT(0)
        up_a[i].w = FT(0)
    end
end;

function turbconv_nodal_update_auxiliary_state!(
    turbconv::EDMF{FT},
    m::AtmosModel{FT},
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT}
    N_up = n_updrafts(turbconv)
    save_subdomain_temperature!(m, state, aux)

    en_a = aux.turbconv.environment
    up_a = aux.turbconv.updraft
    gm = state
    en = state.turbconv.environment
    up = state.turbconv.updraft

    # Recover thermo states
    ts = recover_thermo_state_all(m, state, aux)

    # Get environment variables
    env = environment_vars(state, aux, N_up)

    # Compute buoyancies of subdomains
    gm_p = air_pressure(ts.gm)
    ρinv = 1 / gm.ρ
    _grav::FT = grav(m.param_set)

    z = altitude(m, aux)

    en_ρ = air_density(ts.en)
    en_a.buoyancy = -_grav * (en_ρ - aux.ref_state.ρ) * ρinv

    @unroll_map(N_up) do i
        ρ_i = air_density(ts.up[i])
        up_a[i].buoyancy = -_grav * (ρ_i - aux.ref_state.ρ) * ρinv
        up_a[i].a = up[i].ρa * ρinv
        up_a[i].θ_liq = up[i].ρaθ_liq / up[i].ρa
        up_a[i].q_tot = up[i].ρaq_tot / up[i].ρa
        up_a[i].w = up[i].ρaw / up[i].ρa
    end
    b_gm = grid_mean_b(state, aux, N_up)

    # remove the gm_b from all subdomains
    @unroll_map(N_up) do i
        up_a[i].buoyancy -= b_gm
    end
    en_a.buoyancy -= b_gm

    res = ntuple(N_up) do i
        entr_detr(m, m.turbconv.entr_detr, state, aux, t, ts, env, i)
    end

    ε_dyn, δ_dyn, ε_trb = ntuple(i -> map(x -> x[i], res), 3)

    @unroll_map(N_up) do i
        up_a[i].ε_dyn = ε_dyn[i]
        up_a[i].δ_dyn = δ_dyn[i]
        up_a[i].ε_trb = ε_trb[i]
    end

end;

function compute_gradient_argument!(
    turbconv::EDMF{FT},
    m::AtmosModel{FT},
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT}
    N_up = n_updrafts(turbconv)
    z = altitude(m, aux)

    # Aliases:
    up_t = transform.turbconv.updraft
    en_t = transform.turbconv.environment
    gm = state
    up = state.turbconv.updraft
    en = state.turbconv.environment

    # Recover thermo states
    ts = recover_thermo_state_all(m, state, aux)

    # Get environment variables
    env = environment_vars(state, aux, N_up)

    @unroll_map(N_up) do i
        up_t[i].w = up[i].ρaw / up[i].ρa
    end
    _grav::FT = grav(m.param_set)

    ρinv = 1 / gm.ρ
    en_θ_liq = liquid_ice_pottemp(ts.en)
    en_q_tot = total_specific_humidity(ts.en)

    # populate gradient arguments
    en_t.θ_liq = en_θ_liq
    en_t.q_tot = en_q_tot
    en_t.w = env.w

    en_t.tke = enforce_positivity(en.ρatke) / (env.a * gm.ρ)
    en_t.θ_liq_cv = enforce_positivity(en.ρaθ_liq_cv) / (env.a * gm.ρ)
    en_t.q_tot_cv = enforce_positivity(en.ρaq_tot_cv) / (env.a * gm.ρ)
    en_t.θ_liq_q_tot_cv = en.ρaθ_liq_q_tot_cv / (env.a * gm.ρ)

    # TODO: is this supposed to be grabbed from grid mean?
    en_t.θv = virtual_pottemp(ts.gm)
    gm_p = air_pressure(ts.gm)
    e_kin = FT(1 // 2) * ((gm.ρu[1] * ρinv)^2 + (gm.ρu[2] * ρinv)^2 + env.w^2)
    en_t.e = total_energy(e_kin, _grav * z, ts.en)
end;

function compute_gradient_flux!(
    turbconv::EDMF{FT},
    m::AtmosModel{FT},
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT}
    N_up = n_updrafts(turbconv)

    # Aliases:
    gm = state
    gm_d = diffusive
    up_d = diffusive.turbconv.updraft
    tc_d = diffusive.turbconv
    up_∇t = ∇transform.turbconv.updraft
    en_d = diffusive.turbconv.environment
    en_∇t = ∇transform.turbconv.environment
    tc_∇t = ∇transform.turbconv

    @unroll_map(N_up) do i
        up_d[i].∇w = up_∇t[i].w
    end

    ρinv = 1 / gm.ρ
    # first moment grid mean coming from environment gradients only
    en_d.∇θ_liq = en_∇t.θ_liq
    en_d.∇q_tot = en_∇t.q_tot
    en_d.∇w = en_∇t.w
    # second moment env cov
    en_d.∇tke = en_∇t.tke
    en_d.∇θ_liq_cv = en_∇t.θ_liq_cv
    en_d.∇q_tot_cv = en_∇t.q_tot_cv
    en_d.∇θ_liq_q_tot_cv = en_∇t.θ_liq_q_tot_cv

    en_d.∇θv = en_∇t.θv
    en_d.∇e = en_∇t.e

    tc_d.S² = ∇transform.u[3, 1]^2 + ∇transform.u[3, 2]^2 + en_d.∇w[3]^2
end;


function turbconv_source!(
    m::AtmosModel{FT},
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
) where {FT}
    turbconv = m.turbconv
    N_up = n_updrafts(turbconv)

    # Aliases:
    gm = state
    en = state.turbconv.environment
    up = state.turbconv.updraft
    gm_s = source
    gm_d = diffusive
    en_s = source.turbconv.environment
    up_s = source.turbconv.updraft
    en_d = diffusive.turbconv.environment
    up_a = aux.turbconv.updraft

    # Recover thermo states
    ts = recover_thermo_state_all(m, state, aux)

    # Get environment variables
    env = environment_vars(state, aux, N_up)

    res = ntuple(N_up) do i
        entr_detr(m, m.turbconv.entr_detr, state, aux, t, ts, env, i)
    end
    ε_dyn, δ_dyn, ε_trb = ntuple(i -> map(x -> x[i], res), 3)

    # get environment values
    _grav::FT = grav(m.param_set)
    ρinv = 1 / gm.ρ
    en_θ_liq = liquid_ice_pottemp(ts.en)
    en_q_tot = total_specific_humidity(ts.en)
    tke_env = enforce_positivity(en.ρatke) * ρinv / env.a
    gm_θ_liq = liquid_ice_pottemp(ts.gm)

    @unroll_map(N_up) do i
        # upd vars
        ρaᵢ = enforce_unit_bounds(up[i].ρa, turbconv.entr_detr.a_min)
        wᵢ = up[i].ρaw / ρaᵢ
        ρaᵢ_inv = FT(1) / ρaᵢ

        # first moment sources - for now we compute these as aux variable
        dpdz, dpdz_tke_i = perturbation_pressure(
            m,
            m.turbconv.pressure,
            state,
            diffusive,
            aux,
            t,
            env,
            i,
        )

        # entrainment and detrainment
        up_s[i].ρa += up[i].ρaw * (ε_dyn[i] - δ_dyn[i])
        up_s[i].ρaw +=
            up[i].ρaw *
            ((ε_dyn[i] + ε_trb[i]) * env.w - (δ_dyn[i] + ε_trb[i]) * wᵢ)
        up_s[i].ρaθ_liq +=
            up[i].ρaw * (
                (ε_dyn[i] + ε_trb[i]) * en_θ_liq -
                (δ_dyn[i] + ε_trb[i]) * up[i].ρaθ_liq * ρaᵢ_inv
            )
        up_s[i].ρaq_tot +=
            up[i].ρaw * (
                (ε_dyn[i] + ε_trb[i]) * en_q_tot -
                (δ_dyn[i] + ε_trb[i]) * up[i].ρaq_tot * ρaᵢ_inv
            )

        # add buoyancy and perturbation pressure in subdomain w equation
        up_s[i].ρaw += up[i].ρa * (up_a[i].buoyancy - dpdz)
        # microphysics sources should be applied here

        # environment second moments:
        en_s.ρatke += (
            up[i].ρaw * δ_dyn[i] * (wᵢ - env.w) * (wᵢ - env.w) * FT(0.5) +
            up[i].ρaw * ε_trb[i] * (env.w - gm.ρu[3] * ρinv) * (env.w - wᵢ) - up[i].ρaw * (ε_dyn[i] + ε_trb[i]) * tke_env
        )

        en_s.ρaθ_liq_cv += (
            up[i].ρaw *
            δ_dyn[i] *
            (up[i].ρaθ_liq * ρaᵢ_inv - en_θ_liq) *
            (up[i].ρaθ_liq * ρaᵢ_inv - en_θ_liq) +
            up[i].ρaw *
            ε_trb[i] *
            (en_θ_liq - gm_θ_liq) *
            (en_θ_liq - up[i].ρaθ_liq * ρaᵢ_inv) +
            up[i].ρaw *
            ε_trb[i] *
            (en_θ_liq - gm_θ_liq) *
            (en_θ_liq - up[i].ρaθ_liq * ρaᵢ_inv) -
            up[i].ρaw * (ε_dyn[i] + ε_trb[i]) * en.ρaθ_liq_cv
        )

        en_s.ρaq_tot_cv += (
            up[i].ρaw *
            δ_dyn[i] *
            (up[i].ρaq_tot * ρaᵢ_inv - en_q_tot) *
            (up[i].ρaq_tot * ρaᵢ_inv - en_q_tot) +
            up[i].ρaw *
            ε_trb[i] *
            (en_q_tot - gm.moisture.ρq_tot * ρinv) *
            (en_q_tot - up[i].ρaq_tot * ρaᵢ_inv) +
            up[i].ρaw *
            ε_trb[i] *
            (en_q_tot - gm.moisture.ρq_tot * ρinv) *
            (en_q_tot - up[i].ρaq_tot * ρaᵢ_inv) -
            up[i].ρaw * (ε_dyn[i] + ε_trb[i]) * en.ρaq_tot_cv
        )

        en_s.ρaθ_liq_q_tot_cv += (
            up[i].ρaw *
            δ_dyn[i] *
            (up[i].ρaθ_liq * ρaᵢ_inv - en_θ_liq) *
            (up[i].ρaq_tot * ρaᵢ_inv - en_q_tot) +
            up[i].ρaw *
            ε_trb[i] *
            (en_θ_liq - gm_θ_liq) *
            (en_q_tot - up[i].ρaq_tot * ρaᵢ_inv) +
            up[i].ρaw *
            ε_trb[i] *
            (en_q_tot - gm.moisture.ρq_tot * ρinv) *
            (en_θ_liq - up[i].ρaθ_liq * ρaᵢ_inv) -
            up[i].ρaw * (ε_dyn[i] + ε_trb[i]) * en.ρaθ_liq_q_tot_cv
        )

        # pressure tke source from the i'th updraft
        en_s.ρatke += ρaᵢ * dpdz_tke_i
    end
    l_mix = mixing_length(
        m,
        m.turbconv.mix_len,
        state,
        diffusive,
        aux,
        t,
        δ_dyn,
        ε_trb,
        ts,
        env,
    )
    K_eddy = m.turbconv.mix_len.c_m * l_mix * sqrt(tke_env)
    Shear² = diffusive.turbconv.S²
    ρa₀ = gm.ρ * env.a
    Diss₀ = m.turbconv.mix_len.c_d * sqrt(tke_env) / l_mix

    # production from mean gradient and Dissipation
    en_s.ρatke += ρa₀ * (K_eddy * Shear² - Diss₀ * tke_env)
    en_s.ρaθ_liq_cv +=
        ρa₀ * (K_eddy * en_d.∇θ_liq[3] * en_d.∇θ_liq[3] - Diss₀ * en.ρaθ_liq_cv)
    en_s.ρaq_tot_cv +=
        ρa₀ * (K_eddy * en_d.∇q_tot[3] * en_d.∇q_tot[3] - Diss₀ * en.ρaq_tot_cv)
    en_s.ρaθ_liq_q_tot_cv +=
        ρa₀ *
        (K_eddy * en_d.∇θ_liq[3] * en_d.∇q_tot[3] - Diss₀ * en.ρaθ_liq_q_tot_cv)
    # covariance microphysics sources should be applied here
end;

# # in the EDMF first order (advective) fluxes exist only in the grid mean (if <w> is nonzero) and the uprdafts
function flux_first_order!(
    turbconv::EDMF{FT},
    m::AtmosModel{FT},
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT}
    # Aliases:
    gm = state
    up = state.turbconv.updraft
    up_f = flux.turbconv.updraft
    N_up = n_updrafts(turbconv)

    ρinv = 1 / gm.ρ
    ẑ = vertical_unit_vector(m, aux)
    # in future GCM implementations we need to think about grid mean advection
    @unroll_map(N_up) do i
        ρa_i = enforce_unit_bounds(up[i].ρa, turbconv.entr_detr.a_min)
        up_f[i].ρa = up[i].ρaw * ẑ
        w = up[i].ρaw / ρa_i
        up_f[i].ρaw = up[i].ρaw * w * ẑ
        up_f[i].ρaθ_liq = w * up[i].ρaθ_liq * ẑ
        up_f[i].ρaq_tot = w * up[i].ρaq_tot * ẑ
    end
end;

# in the EDMF second order (diffusive) fluxes
# exist only in the grid mean and the environment
function flux_second_order!(
    turbconv::EDMF{FT},
    m::AtmosModel{FT},
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
) where {FT}
    N_up = n_updrafts(turbconv)

    # Aliases:
    gm = state
    up = state.turbconv.updraft
    en = state.turbconv.environment
    gm_f = flux
    up_f = flux.turbconv.updraft
    en_f = flux.turbconv.environment
    en_d = diffusive.turbconv.environment
    up_a = aux.turbconv.updraft

    # Recover thermo states
    ts = recover_thermo_state_all(m, state, aux)

    # Get environment variables
    env = environment_vars(state, aux, N_up)

    ρinv = FT(1) / gm.ρ
    _grav::FT = grav(m.param_set)
    z = altitude(m, aux)
    a_min = turbconv.entr_detr.a_min

    res = ntuple(N_up) do i
        entr_detr(m, m.turbconv.entr_detr, state, aux, t, ts, env, i)
    end

    ε_dyn, δ_dyn, ε_trb = ntuple(i -> map(x -> x[i], res), 3)

    l_mix = mixing_length(
        m,
        turbconv.mix_len,
        state,
        diffusive,
        aux,
        t,
        δ_dyn,
        ε_trb,
        ts,
        env,
    )
    tke_env = enforce_positivity(en.ρatke) / env.a * ρinv
    K_eddy = m.turbconv.mix_len.c_m * l_mix * sqrt(tke_env)

    #TotalFlux(ϕ) = Eddy_Diffusivity(ϕ) + MassFlux(ϕ)
    e_int = internal_energy(m, state, aux)
    gm_p = air_pressure(ts.gm)

    e_kin = vuntuple(N_up) do i
        FT(1 // 2) * (
            (gm.ρu[1] * ρinv)^2 +
            (gm.ρu[2] * ρinv)^2 +
            (up[i].ρaw / up[i].ρa)^2
        )
    end
    up_e = ntuple(i -> total_energy(e_kin[i], _grav * z, ts.up[i]), N_up)
    ρa_i = vuntuple(i -> enforce_unit_bounds(up[i].ρa, a_min), N_up) # TODO: should this be ρa_min?

    massflux_e = sum(
        vuntuple(N_up) do i
            (gm.ρe * ρinv - up_e[i]) * (gm.ρu[3] * ρinv - up[i].ρaw / ρa_i[i])
        end,
    )

    massflux_q_tot = sum(
        vuntuple(N_up) do i
            up[i].ρa *
            ρinv *
            (gm.moisture.ρq_tot * ρinv - up[i].ρaq_tot / up[i].ρa) *
            (
                gm.ρu[3] * ρinv -
                up[i].ρaw / enforce_unit_bounds(up[i].ρa, a_min) # TODO: should this be ρa_min?
            )
        end,
    )

    massflux_w = sum(
        vuntuple(N_up) do i
            up[i].ρa *
            ρinv *
            (gm.ρu[3] * ρinv - up[i].ρaw / up[i].ρa) *
            (
                gm.ρu[3] * ρinv -
                up[i].ρaw / enforce_unit_bounds(up[i].ρa, a_min) # TODO: should this be ρa_min?
            )
        end,
    )

    # update grid mean flux_second_order
    ρe_sgs_flux = -gm.ρ * env.a * K_eddy * en_d.∇e[3] + massflux_e
    ρq_tot_sgs_flux = -gm.ρ * env.a * K_eddy * en_d.∇q_tot[3] + massflux_q_tot
    ρu_sgs_flux = -gm.ρ * env.a * K_eddy * en_d.∇w[3] + massflux_w

    # for now the coupling to the dycore is commented out

    # gm_f.ρe              += SVector{3,FT}(0,0,ρe_sgs_flux)
    # gm_f.moisture.ρq_tot += SVector{3,FT}(0,0,ρq_tot_sgs_flux)
    # gm_f.ρu              += SMatrix{3, 3, FT, 9}(
    #     0, 0, 0,
    #     0, 0, 0,
    #     0, 0, ρu_sgs_flux,
    # )

    ẑ = vertical_unit_vector(m, aux)
    # env second moment flux_second_order
    en_f.ρatke = -gm.ρ * env.a * K_eddy * en_d.∇tke[3] * ẑ
    en_f.ρaθ_liq_cv = -gm.ρ * env.a * K_eddy * en_d.∇θ_liq_cv[3] * ẑ
    en_f.ρaq_tot_cv = -gm.ρ * env.a * K_eddy * en_d.∇q_tot_cv[3] * ẑ
    en_f.ρaθ_liq_q_tot_cv =
        -gm.ρ * env.a * K_eddy * en_d.∇θ_liq_q_tot_cv[3] * ẑ
end;

# First order boundary conditions
function turbconv_boundary_state!(
    nf,
    bc::EDMFBCs,
    m::AtmosModel{FT},
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    state_int::Vars,
    aux_int::Vars,
) where {FT}

    turbconv = m.turbconv
    N_up = n_updrafts(turbconv)
    up = state⁺.turbconv.updraft
    en = state⁺.turbconv.environment
    gm = state⁺
    gm_a = aux⁺
    if bctype == 1 # bottom
        zLL = altitude(m, aux_int)
        upd_a_surf,
        upd_θ_liq_surf,
        upd_q_tot_surf,
        θ_liq_cv,
        q_tot_cv,
        θ_liq_q_tot_cv,
        tke = subdomain_surface_values(
            turbconv.surface,
            turbconv,
            m,
            gm,
            gm_a,
            zLL,
        )

        @unroll_map(N_up) do i
            up[i].ρaw = FT(0)
            up[i].ρa = upd_a_surf[i] * gm.ρ
            up[i].ρaθ_liq = up[i].ρa * upd_θ_liq_surf[i]
            up[i].ρaq_tot = up[i].ρa * upd_q_tot_surf[i]
        end
        en_area = environment_area(gm, gm_a, N_up)
        en.ρatke = gm.ρ * en_area * tke
        en.ρaθ_liq_cv = gm.ρ * en_area * θ_liq_cv
        en.ρaq_tot_cv = gm.ρ * en_area * q_tot_cv
        en.ρaθ_liq_q_tot_cv = gm.ρ * en_area * θ_liq_q_tot_cv
    end
end;

# The boundary conditions for second-order unknowns
function turbconv_normal_boundary_flux_second_order!(
    nf,
    bc::EDMFBCs,
    m::AtmosModel{FT},
    fluxᵀn::Vars,
    n⁻,
    state⁻::Vars,
    diff⁻::Vars,
    hyperdiff⁻::Vars,
    aux⁻::Vars,
    state⁺::Vars,
    diff⁺::Vars,
    hyperdiff⁺::Vars,
    aux⁺::Vars,
    bctype,
    t,
    _...,
) where {FT}
    turbconv = m.turbconv
    N = n_updrafts(turbconv)
    up_f = fluxᵀn.turbconv.updraft
    en_f = fluxᵀn.turbconv.environment
    if bctype == 2 # top
        @unroll_map(N_up) do i
            up_f[i].ρaw = -n⁻ * FT(0)
            up_f[i].ρa = -n⁻ * FT(0)
            up_f[i].ρaθ_liq = -n⁻ * FT(0)
            up_f[i].ρaq_tot = -n⁻ * FT(0)
        end
        en_f.∇tke = -n⁻ * FT(0)
        en_f.∇e_int_cv = -n⁻ * FT(0)
        en_f.∇q_tot_cv = -n⁻ * FT(0)
        en_f.∇e_int_q_tot_cv = -n⁻ * FT(0)
    end
end;
