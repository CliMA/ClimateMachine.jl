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
    prognostic_to_primitive!,
    primitive_to_prognostic!,
    get_prog_state,
    flux,
    precompute,
    source,
    eq_tends,
    update_auxiliary_state!,
    init_state_prognostic!,
    compute_gradient_argument!,
    compute_gradient_flux!

import ClimateMachine.TurbulenceConvection:
    init_aux_turbconv!,
    turbconv_nodal_update_auxiliary_state!,
    turbconv_boundary_state!,
    turbconv_normal_boundary_flux_second_order!

using ClimateMachine.Thermodynamics: air_pressure, air_density

include(joinpath("helper_funcs", "lamb_smooth_minimum.jl"))
include(joinpath("helper_funcs", "utility_funcs.jl"))
include(joinpath("helper_funcs", "subdomain_statistics.jl"))
include(joinpath("helper_funcs", "diagnose_environment.jl"))
include(joinpath("helper_funcs", "subdomain_thermo_states.jl"))
include(joinpath("helper_funcs", "save_subdomain_temperature.jl"))
include(joinpath("closures", "mixing_length.jl"))
include(joinpath("closures", "turbulence_functions.jl"))
include(joinpath("closures", "surface_functions.jl"))


vars_state(::Environment, ::Auxiliary, FT) = @vars(T::FT)

function vars_state(m::EDMF, st::Auxiliary, FT)
    @vars(
        environment::vars_state(m.environment, st, FT),
    )
end

function vars_state(::Environment, ::Prognostic, FT)
    @vars(ρatke::FT, ρaθ_liq_cv::FT, ρaq_tot_cv::FT, ρaθ_liq_q_tot_cv::FT,)
end

function vars_state(::Environment, ::Primitive, FT)
    @vars(atke::FT, aθ_liq_cv::FT, aq_tot_cv::FT, aθ_liq_q_tot_cv::FT,)
end

function vars_state(m::EDMF, st::Union{Prognostic, Primitive}, FT)
    @vars(
        environment::vars_state(m.environment, st, FT),
    )
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

function vars_state(m::EDMF, st::Gradient, FT)
    @vars(
        environment::vars_state(m.environment, st, FT),
        u::FT,
        v::FT
    )
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

prognostic_vars(m::EDMF) =
    (prognostic_vars(m.environment)...,)
prognostic_vars(m::Environment) =
    (en_ρatke(), en_ρaθ_liq_cv(), en_ρaq_tot_cv(), en_ρaθ_liq_q_tot_cv())

get_prog_state(state, ::en_ρatke) = (state.turbconv.environment, :ρatke)
get_prog_state(state, ::en_ρaθ_liq_cv) =
    (state.turbconv.environment, :ρaθ_liq_cv)
get_prog_state(state, ::en_ρaq_tot_cv) =
    (state.turbconv.environment, :ρaq_tot_cv)
get_prog_state(state, ::en_ρaθ_liq_q_tot_cv) =
    (state.turbconv.environment, :ρaθ_liq_q_tot_cv)

struct EntrDetr{PV} <: TendencyDef{Source, PV} end
struct PressSource{PV} <: TendencyDef{Source, PV} end
struct BuoySource{PV} <: TendencyDef{Source, PV} end
struct ShearSource{PV} <: TendencyDef{Source, PV} end
struct DissSource{PV} <: TendencyDef{Source, PV} end
struct GradProdSource{PV} <: TendencyDef{Source, PV} end

# Dycore tendencies
eq_tends(
    pv::PV,
    m::EDMF,
    ::Flux{SecondOrder},
) where {PV <: Union{Momentum, Energy, TotalMoisture}} =(SGSFlux{PV}(),)  # do _not_ add SGSFlux back to grid-mean
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
) where {PV <: EnvironmentPrognosticVariable} = ()#(Diffusion{PV}(),)

eq_tends(
    pv::PV,
    m::EDMF,
    ::Flux{FirstOrder},
) where {PV <: EDMFPrognosticVariable} = (Advect{PV}(),)

eq_tends(pv::PV, m::EDMF, ::Source) where {PV} = ()

eq_tends(pv::PV, m::EDMF, ::Source) where {PV <: en_ρatke} = (
    # ShearSource{PV}(),
    # BuoySource{PV}(),
    # DissSource{PV}(),
)

eq_tends(
    pv::PV,
    m::EDMF,
    ::Source,
) where {PV <: Union{en_ρaθ_liq_cv, en_ρaq_tot_cv, en_ρaθ_liq_q_tot_cv}} =
    (DissSource{PV}(), GradProdSource{PV}())


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

function prognostic_to_primitive!(
    turbconv::EDMF,
    atmos,
    moist::DryModel,
    prim::Vars,
    prog::Vars,
)
    prim_en = prim.turbconv.environment
    prog_en = prog.turbconv.environment

    ρ_inv = 1 / prog.ρ
    prim_en.atke = prog_en.ρatke * ρ_inv
    prim_en.aθ_liq_cv = prog_en.ρaθ_liq_cv * ρ_inv
    prim_en.aθ_liq_q_tot_cv = prog_en.ρaθ_liq_q_tot_cv * ρ_inv

    if moist isa DryModel
        prim_en.aq_tot_cv = 0
    else
        prim_en.aq_tot_cv = prog_en.ρaq_tot_cv * ρ_inv
    end
end

function primitive_to_prognostic!(
    turbconv::EDMF,
    atmos,
    moist::DryModel,
    prog::Vars,
    prim::Vars,
)

    prim_en = prim.turbconv.environment
    prog_en = prog.turbconv.environment

    ρ_gm = prog.ρ
    prog_en.ρatke = prim_en.atke * ρ_gm
    prog_en.ρaθ_liq_cv = prim_en.aθ_liq_cv * ρ_gm
    prog_en.ρaθ_liq_q_tot_cv = prim_en.aθ_liq_q_tot_cv * ρ_gm

    if moist isa DryModel
        prog_en.ρaq_tot_cv = 0
    else
        prog_en.ρaq_tot_cv = prim_en.aq_tot_cv * ρ_gm
    end

end

function compute_gradient_argument!(
    turbconv::EDMF{FT},
    m::AtmosModel{FT},
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT}
    z = altitude(m, aux)

    # Aliases:
    gm_tf = transform.turbconv
    en_tf = transform.turbconv.environment
    gm = state
    en = state.turbconv.environment

    # Recover thermo states
    ts = recover_thermo_state_all(m, state, aux)

    # Get environment variables
    env = environment_vars(state)

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

    # Aliases:
    gm = state
    gm_dif = diffusive.turbconv
    gm_∇tf = ∇transform.turbconv
    en = state.turbconv.environment
    en_dif = diffusive.turbconv.environment
    en_∇tf = ∇transform.turbconv.environment

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

    env = environment_vars(state)
    tke_en = enforce_positivity(en.ρatke) * ρ_inv / env.a

    buoy = compute_buoyancy(m, state, env, ts.en, aux.ref_state)

    en_dif.l_mix, ∂b∂z_env, Pr_t = mixing_length(
        m,
        m.turbconv.mix_len,
        args,
        ts.gm,
        ts.en,
        env,
    )

    en_dif.K_m = 0.1 #m.turbconv.mix_len.c_m * en_dif.l_mix * sqrt(tke_en)
    K_h = en_dif.K_m / Pr_t
    ρa₀ = gm.ρ * env.a
    Diss₀ = m.turbconv.mix_len.c_d * sqrt(tke_en) / en_dif.l_mix

    en_dif.shear_prod = ρa₀ * en_dif.K_m * gm_dif.S² # tke Shear source
    en_dif.buoy_prod = -ρa₀ * K_h * ∂b∂z_env   # tke Buoyancy source
    en_dif.tke_diss = -ρa₀ * Diss₀ * tke_en  # tke Dissipation
end;


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

function precompute(::EDMF, bl, args, ts, ::Flux{FirstOrder})
    @unpack state, aux = args
    FT = eltype(state)
    env = environment_vars(state)
    gm = state
    ρ_inv = 1 / gm.ρ

    return (; env)
end

function precompute(::EDMF, bl, args, ts, ::Flux{SecondOrder})
    @unpack state, aux, diffusive, t = args
    ts_gm = ts
    env = environment_vars(state)
    ts_en = new_thermo_state_en(bl, bl.moisture, state, aux, ts_gm)

    buoy = compute_buoyancy(bl, state, env, ts_en, aux.ref_state)

    l_mix, ∂b∂z_env, Pr_t = mixing_length(
        bl,
        bl.turbconv.mix_len,
        args,
        ts_gm,
        ts_en,
        env,
    )

    en = state.turbconv.environment
    tke_en = enforce_positivity(en.ρatke) / env.a / state.ρ
    K_m = 0.1 # bl.turbconv.mix_len.c_m * l_mix * sqrt(tke_en)
    K_h = K_m / Pr_t

    return (;
        env,
        ts_en,
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
        ref_state::Vars
    )

Compute buoyancies of subdomains
"""
function compute_buoyancy(
    bl::BalanceLaw,
    state::Vars,
    env::NamedTuple,
    ts_en::ThermodynamicState,
    ref_state::Vars,
)
    FT = eltype(state)
    _grav::FT = grav(bl.param_set)
    gm = state
    ρ_inv = 1 / gm.ρ
    buoyancy_en = -_grav * (air_density(ts_en) - ref_state.ρ) * ρ_inv

    b_gm = buoyancy_en

    # remove the gm_b from all subdomains
    buoyancy_en -= b_gm
    return (;en = buoyancy_en)
end

function precompute(::EDMF, bl, args, ts, ::Source)
    @unpack state, aux, diffusive, t = args
    ts_gm = ts
    gm = state
    # Get environment variables
    env = environment_vars(state)
    # Recover thermo states
    ts_en = new_thermo_state_en(bl, bl.moisture, state, aux, ts_gm)
    buoy = compute_buoyancy(bl, state, env, ts_en, aux.ref_state)

    l_mix, ∂b∂z_env, Pr_t = mixing_length(
        bl,
        bl.turbconv.mix_len,
        args,
        ts_gm,
        ts_en,
        env,
    )

    en = state.turbconv.environment
    tke_en = enforce_positivity(en.ρatke) / env.a / state.ρ
    K_m = 0.1 # bl.turbconv.mix_len.c_m * l_mix * sqrt(tke_en)
    K_h = K_m / Pr_t
    Diss₀ = bl.turbconv.mix_len.c_d * sqrt(tke_en) / l_mix

    return (;
        env,
        Diss₀,
        buoy,
        K_m,
        K_h,
        ts_en,
        l_mix,
        ∂b∂z_env,
        Pr_t,
    )
end

function flux(::SGSFlux{Energy}, atmos, args)
    @unpack state, aux, diffusive = args
    @unpack env, K_h, = args.precomputed.turbconv
    FT = eltype(state)
    _grav::FT = grav(atmos.param_set)
    z = altitude(atmos, aux)
    en_dif = diffusive.turbconv.environment
    gm = state
    ρ_inv = 1 / gm.ρ
    ρu_gm_tup = Tuple(gm.ρu)

    # TODO: Consider turbulent contribution:
    e_kin =
        FT(1 // 2) *
        ((gm.ρu[1] * ρ_inv)^2 + (gm.ρu[2] * ρ_inv)^2 + (gm.ρu[3] * ρ_inv)^2)

    ρe_sgs_flux = -gm.ρ * env.a * K_h * en_dif.∇e[3]
    return SVector{3, FT}(0, 0, ρe_sgs_flux)
end

function flux(::SGSFlux{TotalMoisture}, atmos, args)
    @unpack state, diffusive = args
    @unpack env, K_h = args.precomputed.turbconv
    FT = eltype(state)
    en_dif = diffusive.turbconv.environment
    gm = state
    ρ_inv = 1 / gm.ρ
    ρq_tot = atmos.moisture isa DryModel ? FT(0) : gm.moisture.ρq_tot

    ρu_gm_tup = Tuple(gm.ρu)

    ρq_tot_sgs_flux = -gm.ρ * env.a * K_h * en_dif.∇q_tot[3]
    return SVector{3, FT}(0, 0, ρq_tot_sgs_flux)
end

function flux(::SGSFlux{Momentum}, atmos, args)
    @unpack state, diffusive = args
    @unpack env, K_m = args.precomputed.turbconv
    FT = eltype(state)
    en_dif = diffusive.turbconv.environment
    gm_dif = diffusive.turbconv
    gm = state
    ρ_inv = 1 / gm.ρ

    ρu_gm_tup = Tuple(gm.ρu)

    ρw_sgs_flux = -gm.ρ * env.a * K_m * en_dif.∇w[3]
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

# First order boundary conditions
function turbconv_boundary_state!(
    nf,
    bc::EDMFBottomBC,
    atmos::AtmosModel{FT},
    state⁺::Vars,
    args,
) where {FT}
    @unpack state⁻, aux⁻, aux_int⁻ = args
    turbconv = atmos.turbconv
    en⁺ = state⁺.turbconv.environment
    gm⁻ = state⁻
    gm_a⁻ = aux⁻

    zLL = altitude(atmos, aux_int⁻)
    surf_vals = subdomain_surface_values(atmos, gm⁻, gm_a⁻, zLL)

    a_en = environment_area(gm⁻)
    en⁺.ρatke = gm⁻.ρ * a_en * surf_vals.tke
    en⁺.ρaθ_liq_cv = gm⁻.ρ * a_en * surf_vals.θ_liq_cv
    if !(atmos.moisture isa DryModel)
        en⁺.ρaq_tot_cv = gm⁻.ρ * a_en * surf_vals.q_tot_cv
        en⁺.ρaθ_liq_q_tot_cv = gm⁻.ρ * a_en * surf_vals.θ_liq_q_tot_cv
    else
        en⁺.ρaq_tot_cv = FT(0)
        en⁺.ρaθ_liq_q_tot_cv = FT(0)
    end
end;
function turbconv_boundary_state!(
    nf,
    bc::EDMFTopBC,
    atmos::AtmosModel{FT},
    state⁺::Vars,
    args,
) where {FT}
    nothing
end;


# The boundary conditions for second-order unknowns
function turbconv_normal_boundary_flux_second_order!(
    nf,
    bc::EDMFBottomBC,
    atmos::AtmosModel,
    fluxᵀn::Vars,
    _...,
)
    nothing
end;
function turbconv_normal_boundary_flux_second_order!(
    nf,
    bc::EDMFTopBC,
    atmos::AtmosModel{FT},
    fluxᵀn::Vars,
    args,
) where {FT}

    turbconv = atmos.turbconv
    en_flx = fluxᵀn.turbconv.environment
    en_flx.ρatke = FT(0)
    en_flx.ρaθ_liq_cv = FT(0)
    en_flx.ρaq_tot_cv = FT(0)
    en_flx.ρaθ_liq_q_tot_cv = FT(0)

end;
