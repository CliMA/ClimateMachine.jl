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
    en_aux = aux.turbconv.environment
    up_aux = aux.turbconv.updraft

    en_aux.cld_frac = FT(0)
    en_aux.buoyancy = FT(0)

    @unroll_map(N_up) do i
        up_aux[i].buoyancy = FT(0)
        up_aux[i].θ_liq = FT(0)
        up_aux[i].q_tot = FT(0)
        up_aux[i].w = FT(0)
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

    en_aux = aux.turbconv.environment
    up_aux = aux.turbconv.updraft
    gm = state
    en = state.turbconv.environment
    up = state.turbconv.updraft

    # Recover thermo states
    ts = recover_thermo_state_all(m, state, aux)

    # Get environment variables
    env = environment_vars(state, aux, N_up)

    # Compute buoyancies of subdomains
    ρ_inv = 1 / gm.ρ
    _grav::FT = grav(m.param_set)

    z = altitude(m, aux)

    ρ_en = air_density(ts.en)
    en_aux.buoyancy = -_grav * (ρ_en - aux.ref_state.ρ) * ρ_inv

    @unroll_map(N_up) do i
        ρ_i = air_density(ts.up[i])
        up_aux[i].buoyancy = -_grav * (ρ_i - aux.ref_state.ρ) * ρ_inv
        up_aux[i].a = up[i].ρa * ρ_inv
        up_aux[i].θ_liq = up[i].ρaθ_liq / up[i].ρa
        up_aux[i].q_tot = up[i].ρaq_tot / up[i].ρa
        up_aux[i].w = up[i].ρaw / up[i].ρa
    end
    b_gm = grid_mean_b(state, aux, N_up)

    # remove the gm_b from all subdomains
    @unroll_map(N_up) do i
        up_aux[i].buoyancy -= b_gm
    end
    en_aux.buoyancy -= b_gm

    εδ_up = ntuple(N_up) do i
        entr_detr(m, m.turbconv.entr_detr, state, aux, t, ts, env, i)
    end

    ε_dyn, δ_dyn, ε_trb = ntuple(i -> map(x -> x[i], εδ_up), 3)

    @unroll_map(N_up) do i
        up_aux[i].ε_dyn = ε_dyn[i]
        up_aux[i].δ_dyn = δ_dyn[i]
        up_aux[i].ε_trb = ε_trb[i]
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
    up_tf = transform.turbconv.updraft
    en_tf = transform.turbconv.environment
    gm = state
    up = state.turbconv.updraft
    en = state.turbconv.environment

    # Recover thermo states
    ts = recover_thermo_state_all(m, state, aux)

    # Get environment variables
    env = environment_vars(state, aux, N_up)

    @unroll_map(N_up) do i
        up_tf[i].w = up[i].ρaw / up[i].ρa
    end
    _grav::FT = grav(m.param_set)

    ρ_inv = 1 / gm.ρ
    θ_liq_en = liquid_ice_pottemp(ts.en)
    q_tot_en = total_specific_humidity(ts.en)

    # populate gradient arguments
    en_tf.θ_liq = θ_liq_en
    en_tf.q_tot = q_tot_en
    en_tf.w = env.w

    en_tf.tke = enforce_positivity(en.ρatke) / (env.a * gm.ρ)
    en_tf.θ_liq_cv = enforce_positivity(en.ρaθ_liq_cv) / (env.a * gm.ρ)
    en_tf.q_tot_cv = enforce_positivity(en.ρaq_tot_cv) / (env.a * gm.ρ)
    en_tf.θ_liq_q_tot_cv = en.ρaθ_liq_q_tot_cv / (env.a * gm.ρ)

    # TODO: is this supposed to be grabbed from grid mean?
    en_tf.θv = virtual_pottemp(ts.gm)
    e_kin = FT(1 // 2) * ((gm.ρu[1] * ρ_inv)^2 + (gm.ρu[2] * ρ_inv)^2 + env.w^2)
    en_tf.e = total_energy(e_kin, _grav * z, ts.en)
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
    up_dif = diffusive.turbconv.updraft
    tc_dif = diffusive.turbconv
    up_∇tf = ∇transform.turbconv.updraft
    en_dif = diffusive.turbconv.environment
    en_∇tf = ∇transform.turbconv.environment

    @unroll_map(N_up) do i
        up_dif[i].∇w = up_∇tf[i].w
    end

    ρ_inv = 1 / gm.ρ
    # first moment grid mean coming from environment gradients only
    en_dif.∇θ_liq = en_∇tf.θ_liq
    en_dif.∇q_tot = en_∇tf.q_tot
    en_dif.∇w = en_∇tf.w
    # second moment env cov
    en_dif.∇tke = en_∇tf.tke
    en_dif.∇θ_liq_cv = en_∇tf.θ_liq_cv
    en_dif.∇q_tot_cv = en_∇tf.q_tot_cv
    en_dif.∇θ_liq_q_tot_cv = en_∇tf.θ_liq_q_tot_cv

    en_dif.∇θv = en_∇tf.θv
    en_dif.∇e = en_∇tf.e

    tc_dif.S² = ∇transform.u[3, 1]^2 + ∇transform.u[3, 2]^2 + en_dif.∇w[3]^2
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
    en_src = source.turbconv.environment
    up_src = source.turbconv.updraft
    en_dif = diffusive.turbconv.environment
    up_aux = aux.turbconv.updraft

    # Recover thermo states
    ts = recover_thermo_state_all(m, state, aux)

    # Get environment variables
    env = environment_vars(state, aux, N_up)

    εδ_up = ntuple(N_up) do i
        entr_detr(m, m.turbconv.entr_detr, state, aux, t, ts, env, i)
    end
    ε_dyn, δ_dyn, ε_trb = ntuple(i -> map(x -> x[i], εδ_up), 3)

    # get environment values
    _grav::FT = grav(m.param_set)
    ρ_inv = 1 / gm.ρ
    θ_liq_en = liquid_ice_pottemp(ts.en)
    q_tot_en = total_specific_humidity(ts.en)
    tke_en = enforce_positivity(en.ρatke) * ρ_inv / env.a
    θ_liq = liquid_ice_pottemp(ts.gm)
    a_min = turbconv.subdomains.a_min
    a_max = turbconv.subdomains.a_max

    ρa_up = vuntuple(N_up) do i
        gm.ρ * enforce_unit_bounds(up[i].ρa * ρ_inv, a_min, a_max)
    end

    @unroll_map(N_up) do i

        ρa_up_i = ρa_up[i]
        w_up_i = up[i].ρaw / ρa_up_i
        ρa_up_i_inv = FT(1) / ρa_up_i

        # first moment sources - for now we compute these as aux variable
        dpdz = perturbation_pressure(
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
        up_src[i].ρa += up[i].ρaw * (ε_dyn[i] - δ_dyn[i])
        up_src[i].ρaw +=
            up[i].ρaw *
            ((ε_dyn[i] + ε_trb[i]) * env.w - (δ_dyn[i] + ε_trb[i]) * w_up_i)
        up_src[i].ρaθ_liq +=
            up[i].ρaw * (
                (ε_dyn[i] + ε_trb[i]) * θ_liq_en -
                (δ_dyn[i] + ε_trb[i]) * up[i].ρaθ_liq * ρa_up_i_inv
            )
        up_src[i].ρaq_tot +=
            up[i].ρaw * (
                (ε_dyn[i] + ε_trb[i]) * q_tot_en -
                (δ_dyn[i] + ε_trb[i]) * up[i].ρaq_tot * ρa_up_i_inv
            )

        # add buoyancy and perturbation pressure in subdomain w equation
        up_src[i].ρaw += up[i].ρa * (up_aux[i].buoyancy - dpdz)
        # microphysics sources should be applied here

        # environment second moments:
        en_src.ρatke += (
            up[i].ρaw *
            δ_dyn[i] *
            (w_up_i - env.w) *
            (w_up_i - env.w) *
            FT(0.5) +
            up[i].ρaw *
            ε_trb[i] *
            (env.w - gm.ρu[3] * ρ_inv) *
            (env.w - w_up_i) - up[i].ρaw * (ε_dyn[i] + ε_trb[i]) * tke_en
        )

        en_src.ρaθ_liq_cv += (
            up[i].ρaw *
            δ_dyn[i] *
            (up[i].ρaθ_liq * ρa_up_i_inv - θ_liq_en) *
            (up[i].ρaθ_liq * ρa_up_i_inv - θ_liq_en) +
            up[i].ρaw *
            ε_trb[i] *
            (θ_liq_en - θ_liq) *
            (θ_liq_en - up[i].ρaθ_liq * ρa_up_i_inv) +
            up[i].ρaw *
            ε_trb[i] *
            (θ_liq_en - θ_liq) *
            (θ_liq_en - up[i].ρaθ_liq * ρa_up_i_inv) -
            up[i].ρaw * (ε_dyn[i] + ε_trb[i]) * en.ρaθ_liq_cv
        )

        en_src.ρaq_tot_cv += (
            up[i].ρaw *
            δ_dyn[i] *
            (up[i].ρaq_tot * ρa_up_i_inv - q_tot_en) *
            (up[i].ρaq_tot * ρa_up_i_inv - q_tot_en) +
            up[i].ρaw *
            ε_trb[i] *
            (q_tot_en - gm.moisture.ρq_tot * ρ_inv) *
            (q_tot_en - up[i].ρaq_tot * ρa_up_i_inv) +
            up[i].ρaw *
            ε_trb[i] *
            (q_tot_en - gm.moisture.ρq_tot * ρ_inv) *
            (q_tot_en - up[i].ρaq_tot * ρa_up_i_inv) -
            up[i].ρaw * (ε_dyn[i] + ε_trb[i]) * en.ρaq_tot_cv
        )

        en_src.ρaθ_liq_q_tot_cv += (
            up[i].ρaw *
            δ_dyn[i] *
            (up[i].ρaθ_liq * ρa_up_i_inv - θ_liq_en) *
            (up[i].ρaq_tot * ρa_up_i_inv - q_tot_en) +
            up[i].ρaw *
            ε_trb[i] *
            (θ_liq_en - θ_liq) *
            (q_tot_en - up[i].ρaq_tot * ρa_up_i_inv) +
            up[i].ρaw *
            ε_trb[i] *
            (q_tot_en - gm.moisture.ρq_tot * ρ_inv) *
            (θ_liq_en - up[i].ρaθ_liq * ρa_up_i_inv) -
            up[i].ρaw * (ε_dyn[i] + ε_trb[i]) * en.ρaθ_liq_q_tot_cv
        )

        # pressure tke source from the i'th updraft
        en_src.ρatke += ρa_up_i * (w_up_i - env.w) * dpdz
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
    K_eddy = m.turbconv.mix_len.c_m * l_mix * sqrt(tke_en)
    Shear² = diffusive.turbconv.S²
    ρa₀ = gm.ρ * env.a
    Diss₀ = m.turbconv.mix_len.c_d * sqrt(tke_en) / l_mix

    # production from mean gradient and Dissipation
    en_src.ρatke += ρa₀ * (K_eddy * Shear² - Diss₀ * tke_en)
    en_src.ρaθ_liq_cv +=
        ρa₀ *
        (K_eddy * en_dif.∇θ_liq[3] * en_dif.∇θ_liq[3] - Diss₀ * en.ρaθ_liq_cv)
    en_src.ρaq_tot_cv +=
        ρa₀ *
        (K_eddy * en_dif.∇q_tot[3] * en_dif.∇q_tot[3] - Diss₀ * en.ρaq_tot_cv)
    en_src.ρaθ_liq_q_tot_cv +=
        ρa₀ * (
            K_eddy * en_dif.∇θ_liq[3] * en_dif.∇q_tot[3] -
            Diss₀ * en.ρaθ_liq_q_tot_cv
        )
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
    up_flx = flux.turbconv.updraft
    N_up = n_updrafts(turbconv)

    ρ_inv = 1 / gm.ρ
    ẑ = vertical_unit_vector(m, aux)
    a_min = turbconv.subdomains.a_min
    a_max = turbconv.subdomains.a_max
    # in future GCM implementations we need to think about grid mean advection

    ρa_up = vuntuple(N_up) do i
        gm.ρ * enforce_unit_bounds(up[i].ρa * ρ_inv, a_min, a_max)
    end

    @unroll_map(N_up) do i
        ρa_i = ρa_up[i]
        up_flx[i].ρa = up[i].ρaw * ẑ
        w_up_i = up[i].ρaw / ρa_i
        up_flx[i].ρaw = up[i].ρaw * w_up_i * ẑ
        up_flx[i].ρaθ_liq = w_up_i * up[i].ρaθ_liq * ẑ
        up_flx[i].ρaq_tot = w_up_i * up[i].ρaq_tot * ẑ
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
    gm_flx = flux
    en_flx = flux.turbconv.environment
    en_dif = diffusive.turbconv.environment

    # Recover thermo states
    ts = recover_thermo_state_all(m, state, aux)

    # Get environment variables
    env = environment_vars(state, aux, N_up)

    ρ_inv = FT(1) / gm.ρ
    _grav::FT = grav(m.param_set)
    z = altitude(m, aux)
    a_min = turbconv.subdomains.a_min
    a_max = turbconv.subdomains.a_max

    εδ_up = ntuple(N_up) do i
        entr_detr(m, m.turbconv.entr_detr, state, aux, t, ts, env, i)
    end

    ε_dyn, δ_dyn, ε_trb = ntuple(i -> map(x -> x[i], εδ_up), 3)

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
    tke_en = enforce_positivity(en.ρatke) / env.a * ρ_inv
    K_eddy = m.turbconv.mix_len.c_m * l_mix * sqrt(tke_en)

    #TotalFlux(ϕ) = Eddy_Diffusivity(ϕ) + MassFlux(ϕ)
    e_int = internal_energy(m, state, aux)

    e_kin = vuntuple(N_up) do i
        FT(1 // 2) * (
            (gm.ρu[1] * ρ_inv)^2 +
            (gm.ρu[2] * ρ_inv)^2 +
            (up[i].ρaw / up[i].ρa)^2
        )
    end
    e_tot_up = ntuple(i -> total_energy(e_kin[i], _grav * z, ts.up[i]), N_up)
    ρa_up = vuntuple(N_up) do i
        gm.ρ * enforce_unit_bounds(up[i].ρa * ρ_inv, a_min, a_max)
    end

    massflux_e = sum(
        vuntuple(N_up) do i
            up[i].ρa *
            ρ_inv *
            (gm.ρe * ρ_inv - e_tot_up[i]) *
            (gm.ρu[3] * ρ_inv - up[i].ρaw / ρa_up[i])
        end,
    )

    massflux_q_tot = sum(
        vuntuple(N_up) do i
            up[i].ρa *
            ρ_inv *
            (gm.moisture.ρq_tot * ρ_inv - up[i].ρaq_tot / up[i].ρa) *
            (gm.ρu[3] * ρ_inv - up[i].ρaw / ρa_up[i])
        end,
    )

    massflux_w = sum(
        vuntuple(N_up) do i
            up[i].ρa *
            ρ_inv *
            (gm.ρu[3] * ρ_inv - up[i].ρaw / up[i].ρa) *
            (gm.ρu[3] * ρ_inv - up[i].ρaw / ρa_up[i])
        end,
    )

    # update grid mean flux_second_order
    ρe_sgs_flux = -gm.ρ * env.a * K_eddy * en_dif.∇e[3] + massflux_e
    ρq_tot_sgs_flux = -gm.ρ * env.a * K_eddy * en_dif.∇q_tot[3] + massflux_q_tot
    ρu_sgs_flux = -gm.ρ * env.a * K_eddy * en_dif.∇w[3] + massflux_w

    # for now the coupling to the dycore is commented out

    # gm_flx.ρe              += SVector{3,FT}(0,0,ρe_sgs_flux)
    # gm_flx.moisture.ρq_tot += SVector{3,FT}(0,0,ρq_tot_sgs_flux)
    # gm_flx.ρu              += SMatrix{3, 3, FT, 9}(
    #     0, 0, 0,
    #     0, 0, 0,
    #     0, 0, ρu_sgs_flux,
    # )

    ẑ = vertical_unit_vector(m, aux)
    # env second moment flux_second_order
    en_flx.ρatke = -gm.ρ * env.a * K_eddy * en_dif.∇tke[3] * ẑ
    en_flx.ρaθ_liq_cv = -gm.ρ * env.a * K_eddy * en_dif.∇θ_liq_cv[3] * ẑ
    en_flx.ρaq_tot_cv = -gm.ρ * env.a * K_eddy * en_dif.∇q_tot_cv[3] * ẑ
    en_flx.ρaθ_liq_q_tot_cv =
        -gm.ρ * env.a * K_eddy * en_dif.∇θ_liq_q_tot_cv[3] * ẑ
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
        a_up_surf,
        θ_liq_up_surf,
        q_tot_up_surf,
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
            up[i].ρa = a_up_surf[i] * gm.ρ
            up[i].ρaθ_liq = up[i].ρa * θ_liq_up_surf[i]
            up[i].ρaq_tot = up[i].ρa * q_tot_up_surf[i]
        end
        a_en = environment_area(gm, gm_a, N_up)
        en.ρatke = gm.ρ * a_en * tke
        en.ρaθ_liq_cv = gm.ρ * a_en * θ_liq_cv
        en.ρaq_tot_cv = gm.ρ * a_en * q_tot_cv
        en.ρaθ_liq_q_tot_cv = gm.ρ * a_en * θ_liq_q_tot_cv
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
    up_flx = fluxᵀn.turbconv.updraft
    en_flx = fluxᵀn.turbconv.environment
    if bctype == 2 # top
        @unroll_map(N_up) do i
            up_flx[i].ρaw = -n⁻ * FT(0)
            up_flx[i].ρa = -n⁻ * FT(0)
            up_flx[i].ρaθ_liq = -n⁻ * FT(0)
            up_flx[i].ρaq_tot = -n⁻ * FT(0)
        end
        en_flx.∇tke = -n⁻ * FT(0)
        en_flx.∇e_int_cv = -n⁻ * FT(0)
        en_flx.∇q_tot_cv = -n⁻ * FT(0)
        en_flx.∇e_int_q_tot_cv = -n⁻ * FT(0)
    end
end;
