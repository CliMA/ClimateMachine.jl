#### EDMF model kernels

using CLIMAParameters.Planet: e_int_v0, grav, day, R_d, R_v, molmass_ratio
using Printf
using ClimateMachine.Atmos: nodal_update_auxiliary_state!, Advect

using ClimateMachine.BalanceLaws

using ClimateMachine.MPIStateArrays: MPIStateArray
using ClimateMachine.DGMethods: LocalGeometry, DGModel

import ClimateMachine.BalanceLaws:
    vars_state,
    prognostic_vars,
    flux,
    precompute,
    source,
    eq_tends,
    update_auxiliary_state!,
    init_state_prognostic!,
    flux_first_order!,
    flux_second_order!,
    source!,
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

vars_state(::Updraft, ::Auxiliary, FT) = @vars(T::FT)
vars_state(::Environment, ::Auxiliary, FT) = @vars(T::FT)

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
        updraft::vars_state(m.updraft, st, FT),
        u::FT,
        v::FT
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
        K_m::FT,
        l_mix::FT,
        shear_prod::FT,
        buoy_prod::FT,
        tke_diss::FT
    )

end

function vars_state(m::EDMF, st::GradientFlux, FT)
    @vars(
        S²::FT, # should be conditionally grabbed from atmos.turbulence
        environment::vars_state(m.environment, st, FT),
        updraft::vars_state(m.updraft, st, FT),
        ∇u::SVector{3, FT},
        ∇v::SVector{3, FT}
    )
end

abstract type EDMFPrognosticVariable <: PrognosticVariable end

abstract type EnvironmentPrognosticVariable <: EDMFPrognosticVariable end
struct en_ρatke <: EnvironmentPrognosticVariable end
struct en_ρaθ_liq_cv <: EnvironmentPrognosticVariable end
struct en_ρaq_tot_cv <: EnvironmentPrognosticVariable end
struct en_ρaθ_liq_q_tot_cv <: EnvironmentPrognosticVariable end

abstract type UpdraftPrognosticVariable{i} <: EDMFPrognosticVariable end
struct up_ρa{i} <: UpdraftPrognosticVariable{i} end
struct up_ρaw{i} <: UpdraftPrognosticVariable{i} end
struct up_ρaθ_liq{i} <: UpdraftPrognosticVariable{i} end
struct up_ρaq_tot{i} <: UpdraftPrognosticVariable{i} end

prognostic_vars(m::EDMF) =
    (prognostic_vars(m.environment)..., prognostic_vars(m.updraft)...)
prognostic_vars(m::Environment) =
    (en_ρatke(), en_ρaθ_liq_cv(), en_ρaq_tot_cv(), en_ρaθ_liq_q_tot_cv())

function prognostic_vars(m::NTuple{N, Updraft}) where {N}
    t_ρa = vuntuple(i -> up_ρa{i}(), N)
    t_ρaw = vuntuple(i -> up_ρaw{i}(), N)
    t_ρaθ_liq = vuntuple(i -> up_ρaθ_liq{i}(), N)
    t_ρaq_tot = vuntuple(i -> up_ρaq_tot{i}(), N)
    t = (t_ρa..., t_ρaw..., t_ρaθ_liq..., t_ρaq_tot...)
    return t
end

struct EntrDetr{PV} <: TendencyDef{Source, PV} end
struct PressSource{PV} <: TendencyDef{Source, PV} end
struct BuoySource{PV} <: TendencyDef{Source, PV} end
struct ShearSource{PV} <: TendencyDef{Source, PV} end
struct DissSource{PV} <: TendencyDef{Source, PV} end
struct GradProdSource{PV} <: TendencyDef{Source, PV} end

EntrDetr(N_up) = (
    vuntuple(i -> EntrDetr{up_ρa{i}}(), N_up)...,
    vuntuple(i -> EntrDetr{up_ρaw{i}}(), N_up)...,
    vuntuple(i -> EntrDetr{up_ρaθ_liq{i}}(), N_up)...,
    vuntuple(i -> EntrDetr{up_ρaq_tot{i}}(), N_up)...,
    EntrDetr{en_ρatke}(),
    EntrDetr{en_ρaθ_liq_cv}(),
    EntrDetr{en_ρaq_tot_cv}(),
    EntrDetr{en_ρaθ_liq_q_tot_cv}(),
)
PressSource(N_up) = vuntuple(i -> PressSource{up_ρaw{i}}(), N_up)
BuoySource(N_up) = vuntuple(i -> BuoySource{up_ρaw{i}}(), N_up)

# Dycore tendencies
eq_tends(
    pv::PV,
    m::EDMF,
    ::Flux{SecondOrder},
) where {PV <: Union{Momentum, Energy, TotalMoisture}} = ()  # do _not_ add SGSFlux back to grid-mean
# (SGSFlux{PV}(),) # add SGSFlux back to grid-mean

# Turbconv tendencies
eq_tends(
    pv::PV,
    m::AtmosModel,
    tt::Flux{O},
) where {O, PV <: EDMFPrognosticVariable} = eq_tends(pv, m.turbconv, tt)

eq_tends(pv::PV, m::EDMF, ::Flux{O}) where {O, PV <: EDMFPrognosticVariable} =
    ()

eq_tends(
    pv::PV,
    m::EDMF,
    ::Flux{SecondOrder},
) where {PV <: EnvironmentPrognosticVariable} = (Diffusion{PV}(),)

eq_tends(
    pv::PV,
    m::EDMF,
    ::Flux{FirstOrder},
) where {PV <: EDMFPrognosticVariable} = (Advect{PV}(),)

eq_tends(pv::PV, m::EDMF, ::Source) where {PV} = ()

eq_tends(pv::PV, m::EDMF, ::Source) where {PV <: EDMFPrognosticVariable} =
    (EntrDetr{PV}(),)

eq_tends(pv::PV, m::EDMF, ::Source) where {PV <: en_ρatke} = (
    EntrDetr{PV}(),
    PressSource{PV}(),
    ShearSource{PV}(),
    BuoySource{PV}(),
    DissSource{PV}(),
)

eq_tends(
    pv::PV,
    m::EDMF,
    ::Source,
) where {PV <: Union{en_ρaθ_liq_cv, en_ρaq_tot_cv, en_ρaθ_liq_q_tot_cv}} =
    (EntrDetr{PV}(), DissSource{PV}(), GradProdSource{PV}())

eq_tends(pv::PV, m::EDMF, ::Source) where {PV <: up_ρaw} = (
    EntrDetr{PV}(),
    PressSource(n_updrafts(m))...,
    BuoySource(n_updrafts(m))...,
)

struct SGSFlux{PV <: Union{Momentum, Energy, TotalMoisture}} <:
       TendencyDef{Flux{SecondOrder}, PV} end

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
) where {FT} end;

function turbconv_nodal_update_auxiliary_state!(
    turbconv::EDMF{FT},
    m::AtmosModel{FT},
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT}
    save_subdomain_temperature!(m, state, aux)
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
    gm_tf = transform.turbconv
    up_tf = transform.turbconv.updraft
    en_tf = transform.turbconv.environment
    gm = state
    up = state.turbconv.updraft
    en = state.turbconv.environment

    # Recover thermo states
    ts = recover_thermo_state_all(m, state, aux)

    # Get environment variables
    env = environment_vars(state, N_up)

    @unroll_map(N_up) do i
        up_tf[i].w = fix_void_up(up[i].ρa, up[i].ρaw / up[i].ρa)
    end
    _grav::FT = grav(m.param_set)

    ρ_inv = 1 / gm.ρ
    θ_liq_en = liquid_ice_pottemp(ts.en)

    if m.moisture isa DryModel
        q_tot_en = FT(0)
    else
        q_tot_en = total_specific_humidity(ts.en)
    end

    # populate gradient arguments
    en_tf.θ_liq = θ_liq_en
    en_tf.q_tot = q_tot_en
    en_tf.w = env.w

    en_tf.tke = enforce_positivity(en.ρatke) / (env.a * gm.ρ)
    en_tf.θ_liq_cv = enforce_positivity(en.ρaθ_liq_cv) / (env.a * gm.ρ)

    if m.moisture isa DryModel
        en_tf.q_tot_cv = FT(0)
        en_tf.θ_liq_q_tot_cv = FT(0)
    else
        en_tf.q_tot_cv = enforce_positivity(en.ρaq_tot_cv) / (env.a * gm.ρ)
        en_tf.θ_liq_q_tot_cv = en.ρaθ_liq_q_tot_cv / (env.a * gm.ρ)
    end

    en_tf.θv = virtual_pottemp(ts.en)
    e_kin = FT(1 // 2) * ((gm.ρu[1] * ρ_inv)^2 + (gm.ρu[2] * ρ_inv)^2 + env.w^2) # TBD: Check
    en_tf.e = total_energy(e_kin, _grav * z, ts.en)

    gm_tf.u = gm.ρu[1] * ρ_inv
    gm_tf.v = gm.ρu[2] * ρ_inv
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
    args = (; diffusive, state, aux, t)
    N_up = n_updrafts(turbconv)

    # Aliases:
    gm = state
    gm_dif = diffusive.turbconv
    gm_∇tf = ∇transform.turbconv
    up_dif = diffusive.turbconv.updraft
    up_∇tf = ∇transform.turbconv.updraft
    en = state.turbconv.environment
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

    gm_dif.∇u = gm_∇tf.u
    gm_dif.∇v = gm_∇tf.v

    gm_dif.S² = ∇transform.u[3, 1]^2 + ∇transform.u[3, 2]^2 + en_dif.∇w[3]^2 # ∇transform.u is Jacobian.T

    # Recompute l_mix, K_m and tke budget terms for output.
    ts = recover_thermo_state_all(m, state, aux)

    env = environment_vars(state, N_up)
    tke_en = enforce_positivity(en.ρatke) * ρ_inv / env.a

    buoy = compute_buoyancy(m, state, env, ts.en, ts.up, aux.ref_state)

    E_dyn, Δ_dyn, E_trb = entr_detr(m, state, aux, ts.up, ts.en, env, buoy)

    en_dif.l_mix, ∂b∂z_env, Pr_t = mixing_length(
        m,
        m.turbconv.mix_len,
        args,
        Δ_dyn,
        E_trb,
        ts.gm,
        ts.en,
        env,
    )

    en_dif.K_m = m.turbconv.mix_len.c_m * en_dif.l_mix * sqrt(tke_en)
    K_h = en_dif.K_m / Pr_t
    ρa₀ = gm.ρ * env.a
    Diss₀ = m.turbconv.mix_len.c_d * sqrt(tke_en) / en_dif.l_mix

    en_dif.shear_prod = ρa₀ * en_dif.K_m * gm_dif.S² # tke Shear source
    en_dif.buoy_prod = -ρa₀ * K_h * ∂b∂z_env   # tke Buoyancy source
    en_dif.tke_diss = -ρa₀ * Diss₀ * tke_en  # tke Dissipation
end;

function source(::EntrDetr{up_ρa{i}}, atmos, args) where {i}
    @unpack E_dyn, Δ_dyn, ρa_up = args.precomputed.turbconv
    return fix_void_up(ρa_up[i], E_dyn[i] - Δ_dyn[i])
end

function source(::EntrDetr{up_ρaw{i}}, atmos, args) where {i}
    @unpack E_dyn, Δ_dyn, E_trb, env, ρa_up, w_up = args.precomputed.turbconv
    up = args.state.turbconv.updraft
    entr = fix_void_up(ρa_up[i], (E_dyn[i] + E_trb[i]) * env.w)
    detr = fix_void_up(ρa_up[i], (Δ_dyn[i] + E_trb[i]) * w_up[i])

    return entr - detr
end

function source(::EntrDetr{up_ρaθ_liq{i}}, atmos, args) where {i}
    @unpack E_dyn, Δ_dyn, E_trb, env, ρa_up, ts_en = args.precomputed.turbconv
    up = args.state.turbconv.updraft
    θ_liq_en = liquid_ice_pottemp(ts_en)
    entr = fix_void_up(ρa_up[i], (E_dyn[i] + E_trb[i]) * θ_liq_en)
    detr =
        fix_void_up(ρa_up[i], (Δ_dyn[i] + E_trb[i]) * up[i].ρaθ_liq / ρa_up[i])

    return entr - detr
end

function source(::EntrDetr{up_ρaq_tot{i}}, atmos, args) where {i}
    @unpack E_dyn, Δ_dyn, E_trb, env, ρa_up, ts_en = args.precomputed.turbconv
    up = args.state.turbconv.updraft
    q_tot_en = total_specific_humidity(ts_en)
    entr = fix_void_up(ρa_up[i], (E_dyn[i] + E_trb[i]) * q_tot_en)
    detr =
        fix_void_up(ρa_up[i], (Δ_dyn[i] + E_trb[i]) * up[i].ρaq_tot / ρa_up[i])

    return entr - detr
end

function source(::EntrDetr{en_ρatke}, atmos, args)
    @unpack E_dyn, Δ_dyn, E_trb, env, ρa_up, w_up = args.precomputed.turbconv
    @unpack state = args
    up = state.turbconv.updraft
    en = state.turbconv.environment
    gm = state
    N_up = n_updrafts(atmos.turbconv)
    ρ_inv = 1 / gm.ρ
    tke_en = enforce_positivity(en.ρatke) * ρ_inv / env.a

    entr_detr = vuntuple(N_up) do i
        fix_void_up(
            ρa_up[i],
            E_trb[i] * (env.w - gm.ρu[3] * ρ_inv) * (env.w - w_up[i]) -
            (E_dyn[i] + E_trb[i]) * tke_en +
            Δ_dyn[i] * (w_up[i] - env.w) * (w_up[i] - env.w) / 2,
        )
    end
    return sum(entr_detr)
end

function source(::EntrDetr{en_ρaθ_liq_cv}, atmos, args)
    @unpack E_dyn, Δ_dyn, E_trb, ρa_up, ts_en = args.precomputed.turbconv
    @unpack state = args
    ts_gm = args.precomputed.ts
    up = state.turbconv.updraft
    en = state.turbconv.environment
    N_up = n_updrafts(atmos.turbconv)
    θ_liq = liquid_ice_pottemp(ts_gm)
    θ_liq_en = liquid_ice_pottemp(ts_en)

    entr_detr = vuntuple(N_up) do i
        fix_void_up(
            ρa_up[i],
            Δ_dyn[i] *
            (up[i].ρaθ_liq / ρa_up[i] - θ_liq_en) *
            (up[i].ρaθ_liq / ρa_up[i] - θ_liq_en) +
            E_trb[i] *
            (θ_liq_en - θ_liq) *
            (θ_liq_en - up[i].ρaθ_liq / ρa_up[i]) +
            E_trb[i] *
            (θ_liq_en - θ_liq) *
            (θ_liq_en - up[i].ρaθ_liq / ρa_up[i]) -
            (E_dyn[i] + E_trb[i]) * en.ρaθ_liq_cv,
        )
    end
    return sum(entr_detr)
end

function source(::EntrDetr{en_ρaq_tot_cv}, atmos, args)
    @unpack E_dyn, Δ_dyn, E_trb, ρa_up, ts_en = args.precomputed.turbconv
    @unpack state = args
    FT = eltype(state)
    up = state.turbconv.updraft
    en = state.turbconv.environment
    gm = state
    N_up = n_updrafts(atmos.turbconv)
    q_tot_en = total_specific_humidity(ts_en)
    ρ_inv = 1 / gm.ρ
    ρq_tot = atmos.moisture isa DryModel ? FT(0) : gm.moisture.ρq_tot

    entr_detr = vuntuple(N_up) do i
        fix_void_up(
            ρa_up[i],
            Δ_dyn[i] *
            (up[i].ρaq_tot / ρa_up[i] - q_tot_en) *
            (up[i].ρaq_tot / ρa_up[i] - q_tot_en) +
            E_trb[i] *
            (q_tot_en - ρq_tot * ρ_inv) *
            (q_tot_en - up[i].ρaq_tot / ρa_up[i]) +
            E_trb[i] *
            (q_tot_en - ρq_tot * ρ_inv) *
            (q_tot_en - up[i].ρaq_tot / ρa_up[i]) -
            (E_dyn[i] + E_trb[i]) * en.ρaq_tot_cv,
        )
    end
    return sum(entr_detr)
end

function source(::EntrDetr{en_ρaθ_liq_q_tot_cv}, atmos, args)
    @unpack E_dyn, Δ_dyn, E_trb, ρa_up, ts_en = args.precomputed.turbconv
    @unpack state = args
    FT = eltype(state)
    ts_gm = args.precomputed.ts
    up = state.turbconv.updraft
    en = state.turbconv.environment
    gm = state
    N_up = n_updrafts(atmos.turbconv)
    q_tot_en = total_specific_humidity(ts_en)
    θ_liq = liquid_ice_pottemp(ts_gm)
    θ_liq_en = liquid_ice_pottemp(ts_en)
    ρ_inv = 1 / gm.ρ
    ρq_tot = atmos.moisture isa DryModel ? FT(0) : gm.moisture.ρq_tot

    entr_detr = vuntuple(N_up) do i
        fix_void_up(
            ρa_up[i],
            Δ_dyn[i] *
            (up[i].ρaθ_liq / ρa_up[i] - θ_liq_en) *
            (up[i].ρaq_tot / ρa_up[i] - q_tot_en) +
            E_trb[i] *
            (θ_liq_en - θ_liq) *
            (q_tot_en - up[i].ρaq_tot / ρa_up[i]) +
            E_trb[i] *
            (q_tot_en - ρq_tot * ρ_inv) *
            (θ_liq_en - up[i].ρaθ_liq / ρa_up[i]) -
            (E_dyn[i] + E_trb[i]) * en.ρaθ_liq_q_tot_cv,
        )
    end
    return sum(entr_detr)
end

function source(::PressSource{en_ρatke}, atmos, args)
    @unpack env, ρa_up, dpdz, w_up = args.precomputed.turbconv
    up = args.state.turbconv.updraft
    N_up = n_updrafts(atmos.turbconv)
    press_tke = vuntuple(N_up) do i
        fix_void_up(ρa_up[i], ρa_up[i] * (w_up[i] - env.w) * dpdz[i])
    end
    return sum(press_tke)
end

function source(::ShearSource{en_ρatke}, atmos, args)
    @unpack env, K_m = args.precomputed.turbconv
    gm = args.state
    Shear² = args.diffusive.turbconv.S²
    ρa₀ = gm.ρ * env.a
    # production from mean gradient and Dissipation
    return ρa₀ * K_m * Shear² # tke Shear source
end

function source(::BuoySource{en_ρatke}, atmos, args)
    @unpack env, K_h, ∂b∂z_env = args.precomputed.turbconv
    gm = args.state
    ρa₀ = gm.ρ * env.a
    return -ρa₀ * K_h * ∂b∂z_env   # tke Buoyancy source
end

function source(::DissSource{en_ρatke}, atmos, args)
    @unpack env, l_mix, Diss₀ = args.precomputed.turbconv
    gm = args.state
    en = args.state.turbconv.environment
    ρa₀ = gm.ρ * env.a
    tke_en = enforce_positivity(en.ρatke) / gm.ρ / env.a
    return -ρa₀ * Diss₀ * tke_en  # tke Dissipation
end

function source(::DissSource{en_ρaθ_liq_cv}, atmos, args)
    @unpack env, K_h, Diss₀ = args.precomputed.turbconv
    gm = args.state
    en = args.state.turbconv.environment
    ρa₀ = gm.ρ * env.a
    return -ρa₀ * Diss₀ * en.ρaθ_liq_cv
end

function source(::DissSource{en_ρaq_tot_cv}, atmos, args)
    @unpack env, K_h, Diss₀ = args.precomputed.turbconv
    gm = args.state
    en = args.state.turbconv.environment
    ρa₀ = gm.ρ * env.a
    return -ρa₀ * Diss₀ * en.ρaq_tot_cv
end

function source(::DissSource{en_ρaθ_liq_q_tot_cv}, atmos, args)
    @unpack env, K_h, Diss₀ = args.precomputed.turbconv
    gm = args.state
    en = args.state.turbconv.environment
    ρa₀ = gm.ρ * env.a
    return -ρa₀ * Diss₀ * en.ρaθ_liq_q_tot_cv
end

function source(::GradProdSource{en_ρaθ_liq_cv}, atmos, args)
    @unpack env, K_h, Diss₀ = args.precomputed.turbconv
    gm = args.state
    en_dif = args.diffusive.turbconv.environment
    ρa₀ = gm.ρ * env.a
    return ρa₀ * (2 * K_h * en_dif.∇θ_liq[3] * en_dif.∇θ_liq[3])
end

function source(::GradProdSource{en_ρaq_tot_cv}, atmos, args)
    @unpack env, K_h, Diss₀ = args.precomputed.turbconv
    gm = args.state
    en_dif = args.diffusive.turbconv.environment
    ρa₀ = gm.ρ * env.a
    return ρa₀ * (2 * K_h * en_dif.∇q_tot[3] * en_dif.∇q_tot[3])
end

function source(::GradProdSource{en_ρaθ_liq_q_tot_cv}, atmos, args)
    @unpack env, K_h, Diss₀ = args.precomputed.turbconv
    gm = args.state
    en_dif = args.diffusive.turbconv.environment
    ρa₀ = gm.ρ * env.a
    return ρa₀ * (2 * K_h * en_dif.∇θ_liq[3] * en_dif.∇q_tot[3])
end

function source(::BuoySource{up_ρaw{i}}, atmos, args) where {i}
    @unpack buoy = args.precomputed.turbconv
    up = args.state.turbconv.updraft
    return up[i].ρa * buoy.up[i]
end

function source(::PressSource{up_ρaw{i}}, atmos, args) where {i}
    @unpack dpdz = args.precomputed.turbconv
    up = args.state.turbconv.updraft
    return -up[i].ρa * dpdz[i]
end

function source!(m::EDMF, src::Vars, atmos, args)
    N_up = n_updrafts(atmos.turbconv)
    # Aliases:
    en_src = src.turbconv.environment
    up_src = src.turbconv.updraft
    tend = Source()
    @unroll_map(N_up) do i
        up_src[i].ρa = Σsources(eq_tends(up_ρa{i}(), atmos, tend), atmos, args)
        up_src[i].ρaw =
            Σsources(eq_tends(up_ρaw{i}(), atmos, tend), atmos, args)
        up_src[i].ρaθ_liq =
            Σsources(eq_tends(up_ρaθ_liq{i}(), atmos, tend), atmos, args)

        if !(atmos.moisture isa DryModel)
            up_src[i].ρaq_tot =
                Σsources(eq_tends(up_ρaq_tot{i}(), atmos, tend), atmos, args)
        end
    end
    en_src.ρatke = Σsources(eq_tends(en_ρatke(), atmos, tend), atmos, args)
    en_src.ρaθ_liq_cv =
        Σsources(eq_tends(en_ρaθ_liq_cv(), atmos, tend), atmos, args)

    if !(atmos.moisture isa DryModel)
        en_src.ρaq_tot_cv =
            Σsources(eq_tends(en_ρaq_tot_cv(), atmos, tend), atmos, args)
        en_src.ρaθ_liq_q_tot_cv =
            Σsources(eq_tends(en_ρaθ_liq_q_tot_cv(), atmos, tend), atmos, args)
    end
end;

function compute_ρa_up(atmos, state, aux)
    # Aliases:
    turbconv = atmos.turbconv
    gm = state
    up = state.turbconv.updraft
    N_up = n_updrafts(turbconv)
    a_min = turbconv.subdomains.a_min
    a_max = turbconv.subdomains.a_max
    # in future GCM implementations we need to think about grid mean advection
    ρa_up = vuntuple(N_up) do i
        gm.ρ * enforce_unit_bounds(up[i].ρa / gm.ρ, a_min, a_max)
    end
    return ρa_up
end

function flux(::Advect{up_ρa{i}}, atmos, args) where {i}
    @unpack state, aux = args
    @unpack ρa_up = args.precomputed.turbconv
    up = state.turbconv.updraft
    ẑ = vertical_unit_vector(atmos, aux)
    return fix_void_up(ρa_up[i], up[i].ρaw) * ẑ
end
function flux(::Advect{up_ρaw{i}}, atmos, args) where {i}
    @unpack state, aux = args
    @unpack ρa_up, w_up = args.precomputed.turbconv
    up = state.turbconv.updraft
    ẑ = vertical_unit_vector(atmos, aux)
    return fix_void_up(ρa_up[i], up[i].ρaw * w_up[i]) * ẑ

end
function flux(::Advect{up_ρaθ_liq{i}}, atmos, args) where {i}
    @unpack state, aux = args
    @unpack ρa_up, w_up = args.precomputed.turbconv
    up = state.turbconv.updraft
    ẑ = vertical_unit_vector(atmos, aux)
    return fix_void_up(ρa_up[i], w_up[i] * up[i].ρaθ_liq) * ẑ

end
function flux(::Advect{up_ρaq_tot{i}}, atmos, args) where {i}
    @unpack state, aux = args
    @unpack ρa_up, w_up = args.precomputed.turbconv
    up = state.turbconv.updraft
    ẑ = vertical_unit_vector(atmos, aux)
    return fix_void_up(ρa_up[i], w_up[i] * up[i].ρaq_tot) * ẑ

end

function flux(::Advect{en_ρatke}, atmos, args)
    @unpack state, aux = args
    @unpack env = args.precomputed.turbconv
    en = state.turbconv.environment
    ẑ = vertical_unit_vector(atmos, aux)
    return en.ρatke * env.w * ẑ
end
function flux(::Advect{en_ρaθ_liq_cv}, atmos, args)
    @unpack state, aux = args
    @unpack env = args.precomputed.turbconv
    en = state.turbconv.environment
    ẑ = vertical_unit_vector(atmos, aux)
    return en.ρaθ_liq_cv * env.w * ẑ
end
function flux(::Advect{en_ρaq_tot_cv}, atmos, args)
    @unpack state, aux = args
    @unpack env = args.precomputed.turbconv
    en = state.turbconv.environment
    ẑ = vertical_unit_vector(atmos, aux)
    return en.ρaq_tot_cv * env.w * ẑ
end
function flux(::Advect{en_ρaθ_liq_q_tot_cv}, atmos, args)
    @unpack state, aux = args
    @unpack env = args.precomputed.turbconv
    en = state.turbconv.environment
    ẑ = vertical_unit_vector(atmos, aux)
    return en.ρaθ_liq_q_tot_cv * env.w * ẑ
end

# # in the EDMF first order (advective) fluxes exist only
# in the grid mean (if <w> is nonzero) and the updrafts
function flux_first_order!(
    turbconv::EDMF{FT},
    atmos::AtmosModel{FT},
    flux::Grad,
    args,
) where {FT}
    # Aliases:
    up_flx = flux.turbconv.updraft
    en_flx = flux.turbconv.environment
    N_up = n_updrafts(turbconv)
    # in future GCM implementations we need to think about grid mean advection
    tend = Flux{FirstOrder}()

    @unroll_map(N_up) do i
        up_flx[i].ρa = Σfluxes(eq_tends(up_ρa{i}(), atmos, tend), atmos, args)
        up_flx[i].ρaw = Σfluxes(eq_tends(up_ρaw{i}(), atmos, tend), atmos, args)
        up_flx[i].ρaθ_liq =
            Σfluxes(eq_tends(up_ρaθ_liq{i}(), atmos, tend), atmos, args)
        if !(atmos.moisture isa DryModel)
            up_flx[i].ρaq_tot =
                Σfluxes(eq_tends(up_ρaq_tot{i}(), atmos, tend), atmos, args)
        end
    end
    en_flx.ρatke = Σfluxes(eq_tends(en_ρatke(), atmos, tend), atmos, args)
    en_flx.ρaθ_liq_cv =
        Σfluxes(eq_tends(en_ρaθ_liq_cv(), atmos, tend), atmos, args)
    if !(atmos.moisture isa DryModel)
        en_flx.ρaq_tot_cv =
            Σfluxes(eq_tends(en_ρaq_tot_cv(), atmos, tend), atmos, args)
        en_flx.ρaθ_liq_q_tot_cv =
            Σfluxes(eq_tends(en_ρaθ_liq_q_tot_cv(), atmos, tend), atmos, args)
    end
end;

function precompute(::EDMF, bl, args, ts, ::Flux{FirstOrder})
    @unpack state, aux = args
    FT = eltype(state)
    env = environment_vars(state, n_updrafts(bl.turbconv))
    ρa_up = compute_ρa_up(bl, state, aux)
    up = state.turbconv.updraft
    gm = state
    N_up = n_updrafts(bl.turbconv)
    ρ_inv = 1 / gm.ρ
    w_up = vuntuple(N_up) do i
        fix_void_up(ρa_up[i], up[i].ρaw / ρa_up[i])
    end

    θ_liq_up = vuntuple(N_up) do i
        fix_void_up(up[i].ρa, up[i].ρaθ_liq / up[i].ρa, liquid_ice_pottemp(ts))
    end
    a_up = vuntuple(N_up) do i
        fix_void_up(up[i].ρa, up[i].ρa * ρ_inv)
    end
    if !(bl.moisture isa DryModel)
        q_tot_up = vuntuple(N_up) do i
            fix_void_up(up[i].ρa, up[i].ρaq_tot / up[i].ρa, gm.moisture.ρq_tot)
        end
    else
        q_tot_up = vuntuple(i -> FT(0), N_up)
    end

    return (; env, a_up, q_tot_up, ρa_up, θ_liq_up, w_up)
end

function precompute(::EDMF, bl, args, ts, ::Flux{SecondOrder})
    @unpack state, aux, diffusive, t = args
    ts_gm = ts
    up = state.turbconv.updraft
    N_up = n_updrafts(bl.turbconv)
    env = environment_vars(state, N_up)
    ts_en = new_thermo_state_en(bl, bl.moisture, state, aux, ts_gm)
    ts_up = new_thermo_state_up(bl, bl.moisture, state, aux, ts_gm)

    buoy = compute_buoyancy(bl, state, env, ts_en, ts_up, aux.ref_state)

    E_dyn, Δ_dyn, E_trb = entr_detr(bl, state, aux, ts_up, ts_en, env, buoy)

    l_mix, ∂b∂z_env, Pr_t = mixing_length(
        bl,
        bl.turbconv.mix_len,
        args,
        Δ_dyn,
        E_trb,
        ts_gm,
        ts_en,
        env,
    )
    ρa_up = compute_ρa_up(bl, state, aux)

    en = state.turbconv.environment
    tke_en = enforce_positivity(en.ρatke) / env.a / state.ρ
    K_m = bl.turbconv.mix_len.c_m * l_mix * sqrt(tke_en)
    K_h = K_m / Pr_t
    ρaw_up = vuntuple(i -> up[i].ρaw, N_up)

    return (;
        env,
        ρa_up,
        ρaw_up,
        ts_en,
        ts_up,
        E_dyn,
        Δ_dyn,
        E_trb,
        l_mix,
        ∂b∂z_env,
        K_h,
        K_m,
        Pr_t,
    )
end

"""
    compute_buoyancy(
        bl::BalanceLaw,
        state::Vars,
        env::NamedTuple,
        ts_en::ThermodynamicState,
        ts_up,
        ref_state::Vars
    )

Compute buoyancies of subdomains
"""
function compute_buoyancy(
    bl::BalanceLaw,
    state::Vars,
    env::NamedTuple,
    ts_en::ThermodynamicState,
    ts_up,
    ref_state::Vars,
)
    FT = eltype(state)
    N_up = n_updrafts(bl.turbconv)
    _grav::FT = grav(bl.param_set)
    gm = state
    ρ_inv = 1 / gm.ρ
    buoyancy_en = -_grav * (air_density(ts_en) - ref_state.ρ) * ρ_inv
    up = state.turbconv.updraft

    a_up = vuntuple(N_up) do i
        fix_void_up(up[i].ρa, up[i].ρa * ρ_inv)
    end

    abs_buoyancy_up = vuntuple(N_up) do i
        -_grav * (air_density(ts_up[i]) - ref_state.ρ) * ρ_inv
    end
    b_gm = grid_mean_b(env, a_up, N_up, abs_buoyancy_up, buoyancy_en)

    # remove the gm_b from all subdomains
    buoyancy_up = vuntuple(N_up) do i
        abs_buoyancy_up[i] - b_gm
    end

    buoyancy_en -= b_gm
    return (; up = buoyancy_up, en = buoyancy_en)
end

function precompute(::EDMF, bl, args, ts, ::Source)
    @unpack state, aux, diffusive, t = args
    ts_gm = ts
    gm = state
    up = state.turbconv.updraft
    N_up = n_updrafts(bl.turbconv)
    # Get environment variables
    env = environment_vars(state, N_up)
    # Recover thermo states
    ts_en = new_thermo_state_en(bl, bl.moisture, state, aux, ts_gm)
    ts_up = new_thermo_state_up(bl, bl.moisture, state, aux, ts_gm)
    ρa_up = compute_ρa_up(bl, state, aux)

    buoy = compute_buoyancy(bl, state, env, ts_en, ts_up, aux.ref_state)
    E_dyn, Δ_dyn, E_trb = entr_detr(bl, state, aux, ts_up, ts_en, env, buoy)

    dpdz = perturbation_pressure(bl, args, env, buoy)

    l_mix, ∂b∂z_env, Pr_t = mixing_length(
        bl,
        bl.turbconv.mix_len,
        args,
        Δ_dyn,
        E_trb,
        ts_gm,
        ts_en,
        env,
    )

    w_up = vuntuple(N_up) do i
        fix_void_up(ρa_up[i], up[i].ρaw / ρa_up[i])
    end

    en = state.turbconv.environment
    tke_en = enforce_positivity(en.ρatke) / env.a / state.ρ
    K_m = bl.turbconv.mix_len.c_m * l_mix * sqrt(tke_en)
    K_h = K_m / Pr_t
    Diss₀ = bl.turbconv.mix_len.c_d * sqrt(tke_en) / l_mix

    return (;
        env,
        Diss₀,
        buoy,
        K_m,
        K_h,
        ρa_up,
        w_up,
        ts_en,
        ts_up,
        E_dyn,
        Δ_dyn,
        E_trb,
        dpdz,
        l_mix,
        ∂b∂z_env,
        Pr_t,
    )
end

function flux(::SGSFlux{Energy}, atmos, args)
    @unpack state, aux, diffusive = args
    @unpack env, K_h, ρa_up, ρaw_up, ts_up = args.precomputed.turbconv
    FT = eltype(state)
    _grav::FT = grav(atmos.param_set)
    z = altitude(atmos, aux)
    en_dif = diffusive.turbconv.environment
    up = state.turbconv.updraft
    gm = state
    ρ_inv = 1 / gm.ρ
    N_up = n_updrafts(atmos.turbconv)
    ρu_gm_tup = Tuple(gm.ρu)

    # TODO: Consider turbulent contribution:
    e_kin =
        FT(1 // 2) *
        ((gm.ρu[1] * ρ_inv)^2 + (gm.ρu[2] * ρ_inv)^2 + (gm.ρu[3] * ρ_inv)^2)
    e_tot_up = ntuple(i -> total_energy(e_kin[i], _grav * z, ts_up[i]), N_up)

    massflux_e = sum(
        ntuple(N_up) do i
            fix_void_up(
                ρa_up[i],
                ρa_up[i] *
                (gm.energy.ρe * ρ_inv - e_tot_up[i]) *
                (gm.ρu[3] * ρ_inv - ρaw_up[i] / ρa_up[i]),
            )
        end,
    )
    ρe_sgs_flux = -gm.ρ * env.a * K_h * en_dif.∇e[3] + massflux_e
    return SVector{3, FT}(0, 0, ρe_sgs_flux)
end

function flux(::SGSFlux{TotalMoisture}, atmos, args)
    @unpack state, diffusive = args
    @unpack env, K_h, ρa_up, ρaw_up = args.precomputed.turbconv
    FT = eltype(state)
    en_dif = diffusive.turbconv.environment
    up = state.turbconv.updraft
    gm = state
    ρ_inv = 1 / gm.ρ
    N_up = n_updrafts(atmos.turbconv)
    ρq_tot = atmos.moisture isa DryModel ? FT(0) : gm.moisture.ρq_tot
    ρaq_tot_up = vuntuple(i -> up[i].ρaq_tot, N_up)

    ρu_gm_tup = Tuple(gm.ρu)

    massflux_q_tot = sum(
        ntuple(N_up) do i
            fix_void_up(
                ρa_up[i],
                ρa_up[i] *
                (ρq_tot * ρ_inv - ρaq_tot_up[i] / ρa_up[i]) *
                (ρu_gm_tup[3] * ρ_inv - ρaw_up[i] / ρa_up[i]),
            )
        end,
    )
    ρq_tot_sgs_flux = -gm.ρ * env.a * K_h * en_dif.∇q_tot[3] + massflux_q_tot
    return SVector{3, FT}(0, 0, ρq_tot_sgs_flux)
end

function flux(::SGSFlux{Momentum}, atmos, args)
    @unpack state, diffusive = args
    @unpack env, K_m, ρa_up, ρaw_up = args.precomputed.turbconv
    FT = eltype(state)
    en_dif = diffusive.turbconv.environment
    gm_dif = diffusive.turbconv
    up = state.turbconv.updraft
    gm = state
    ρ_inv = 1 / gm.ρ
    N_up = n_updrafts(atmos.turbconv)

    ρu_gm_tup = Tuple(gm.ρu)

    massflux_w = sum(
        ntuple(N_up) do i
            fix_void_up(
                ρa_up[i],
                ρa_up[i] *
                (ρu_gm_tup[3] * ρ_inv - ρaw_up[i] / ρa_up[i]) *
                (ρu_gm_tup[3] * ρ_inv - ρaw_up[i] / ρa_up[i]),
            )
        end,
    )
    ρw_sgs_flux = -gm.ρ * env.a * K_m * en_dif.∇w[3] + massflux_w
    ρu_sgs_flux = -gm.ρ * env.a * K_m * gm_dif.∇u[3]
    ρv_sgs_flux = -gm.ρ * env.a * K_m * gm_dif.∇v[3]
    return SMatrix{3, 3, FT, 9}(
        0,
        0,
        ρu_sgs_flux,
        0,
        0,
        ρv_sgs_flux,
        0,
        0,
        ρw_sgs_flux,
    )
end

function flux(::Diffusion{en_ρaθ_liq_cv}, atmos, args)
    @unpack state, aux, diffusive = args
    @unpack env, l_mix, Pr_t, K_h = args.precomputed.turbconv
    en_dif = diffusive.turbconv.environment
    gm = state
    ẑ = vertical_unit_vector(atmos, aux)
    return -gm.ρ * env.a * K_h * en_dif.∇θ_liq_cv[3] * ẑ
end
function flux(::Diffusion{en_ρaq_tot_cv}, atmos, args)
    @unpack state, aux, diffusive = args
    @unpack env, l_mix, Pr_t, K_h = args.precomputed.turbconv
    en_dif = diffusive.turbconv.environment
    gm = state
    ẑ = vertical_unit_vector(atmos, aux)
    return -gm.ρ * env.a * K_h * en_dif.∇q_tot_cv[3] * ẑ
end
function flux(::Diffusion{en_ρaθ_liq_q_tot_cv}, atmos, args)
    @unpack state, aux, diffusive = args
    @unpack env, l_mix, Pr_t, K_h = args.precomputed.turbconv
    en_dif = diffusive.turbconv.environment
    gm = state
    ẑ = vertical_unit_vector(atmos, aux)
    return -gm.ρ * env.a * K_h * en_dif.∇θ_liq_q_tot_cv[3] * ẑ
end
function flux(::Diffusion{en_ρatke}, atmos, args)
    @unpack state, aux, diffusive = args
    @unpack env, K_m = args.precomputed.turbconv
    gm = state
    en_dif = diffusive.turbconv.environment
    ẑ = vertical_unit_vector(atmos, aux)
    return -gm.ρ * env.a * K_m * en_dif.∇tke[3] * ẑ
end

function flux_second_order!(
    turbconv::EDMF{FT},
    flux::Grad,
    atmos::AtmosModel{FT},
    args,
) where {FT}

    # Aliases:
    en_flx = flux.turbconv.environment
    flux_pad = SVector(1, 1, 1)
    # in future GCM implementations we need to think about grid mean advection
    tend = Flux{SecondOrder}()
    en_flx.ρatke =
        Σfluxes(eq_tends(en_ρatke(), atmos, tend), atmos, args) .* flux_pad
    # in the EDMF second order (diffusive) fluxes
    # exist only in the grid mean and the environment
    en_flx.ρaθ_liq_cv =
        Σfluxes(eq_tends(en_ρaθ_liq_cv(), atmos, tend), atmos, args) .* flux_pad
    if !(atmos.moisture isa DryModel)
        en_flx.ρaq_tot_cv =
            Σfluxes(eq_tends(en_ρaq_tot_cv(), atmos, tend), atmos, args) .*
            flux_pad
        en_flx.ρaθ_liq_q_tot_cv =
            Σfluxes(
                eq_tends(en_ρaθ_liq_q_tot_cv(), atmos, tend),
                atmos,
                args,
            ) .* flux_pad
    end
end;

# First order boundary conditions
function turbconv_boundary_state!(
    nf,
    bc::EDMFBottomBC,
    m::AtmosModel{FT},
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    t,
    state_int::Vars,
    aux_int::Vars,
) where {FT}

    turbconv = m.turbconv
    N_up = n_updrafts(turbconv)
    up⁺ = state⁺.turbconv.updraft
    en⁺ = state⁺.turbconv.environment
    gm⁻ = state⁻
    gm_a⁻ = aux⁻

    zLL = altitude(m, aux_int)
    a_up_surf,
    θ_liq_up_surf,
    q_tot_up_surf,
    θ_liq_cv,
    q_tot_cv,
    θ_liq_q_tot_cv,
    tke =
        subdomain_surface_values(turbconv.surface, turbconv, m, gm⁻, gm_a⁻, zLL)

    @unroll_map(N_up) do i
        up⁺[i].ρaw = FT(0)
        up⁺[i].ρa = gm⁻.ρ * a_up_surf[i]
        up⁺[i].ρaθ_liq = gm⁻.ρ * a_up_surf[i] * θ_liq_up_surf[i]
        if !(m.moisture isa DryModel)
            up⁺[i].ρaq_tot = gm⁻.ρ * a_up_surf[i] * q_tot_up_surf[i]
        else
            up⁺[i].ρaq_tot = FT(0)
        end
    end

    a_en = environment_area(gm⁻, N_up)
    en⁺.ρatke = gm⁻.ρ * a_en * tke
    en⁺.ρaθ_liq_cv = gm⁻.ρ * a_en * θ_liq_cv
    if !(m.moisture isa DryModel)
        en⁺.ρaq_tot_cv = gm⁻.ρ * a_en * q_tot_cv
        en⁺.ρaθ_liq_q_tot_cv = gm⁻.ρ * a_en * θ_liq_q_tot_cv
    else
        en⁺.ρaq_tot_cv = FT(0)
        en⁺.ρaθ_liq_q_tot_cv = FT(0)
    end
end;
function turbconv_boundary_state!(
    nf,
    bc::EDMFTopBC,
    m::AtmosModel{FT},
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    t,
    state_int::Vars,
    aux_int::Vars,
) where {FT}
    nothing
end;


# The boundary conditions for second-order unknowns
function turbconv_normal_boundary_flux_second_order!(
    nf,
    bc::EDMFBottomBC,
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
    t,
    _...,
) where {FT}
    nothing
end;
function turbconv_normal_boundary_flux_second_order!(
    nf,
    bc::EDMFTopBC,
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
    t,
    _...,
) where {FT}
    turbconv = m.turbconv
    N_up = n_updrafts(turbconv)
    up_flx = fluxᵀn.turbconv.updraft
    en_flx = fluxᵀn.turbconv.environment
    # @unroll_map(N_up) do i
    #     up_flx[i].ρaw = -n⁻ * FT(0)
    #     up_flx[i].ρa = -n⁻ * FT(0)
    #     up_flx[i].ρaθ_liq = -n⁻ * FT(0)
    #     up_flx[i].ρaq_tot = -n⁻ * FT(0)
    # end
    # en_flx.∇tke = -n⁻ * FT(0)
    # en_flx.∇e_int_cv = -n⁻ * FT(0)
    # en_flx.∇q_tot_cv = -n⁻ * FT(0)
    # en_flx.∇e_int_q_tot_cv = -n⁻ * FT(0)

end;
