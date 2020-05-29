#### EDMF model kernels

include(joinpath("helper_funcs", "nondimensional_exchange_functions.jl"))
include(joinpath("helper_funcs", "lamb_smooth_minimum.jl"))
include(joinpath("helper_funcs", "compute_subdomain_statistics.jl"))
include(joinpath("closures", "entr_detr.jl"))
include(joinpath("closures", "pressure.jl"))
include(joinpath("closures", "mixing_length.jl"))
include(joinpath("closures", "quadrature.jl"))
include(joinpath("closures", "micro_phys.jl"))

function vars_state_auxiliary(m::NTuple{N,Updraft}, FT) where {N}
    return Tuple{ntuple(i->vars_state_auxiliary(m[i], FT), N)...}
end

function vars_state_auxiliary(::Updraft, FT)
    @vars(buoyancy::FT,
          upd_top::FT,
          T::FT
          )
end

function vars_state_auxiliary(::Environment, FT)
    @vars(buoyancy::FT,
          K_eddy::FT,
          cld_frac::FT,
          )
end

function vars_state_auxiliary(m::EDMF, FT)
    @vars(environment::vars_state_auxiliary(m.environment, FT),
          updraft::vars_state_auxiliary(m.updraft, FT)
          );
end

function vars_state_conservative(::Updraft, FT)
    @vars(ρa::FT,
          ρau::SVector{3, FT},
          ρae_int::FT,
          ρaq_tot::FT,
          )
end

function vars_state_conservative(::Environment, FT)
    @vars(ρatke::SVector{3, FT},
          ρae_int_cv::FT,
          ρaq_tot_cv::FT,
          ρae_int_q_tot_cv::FT,
          )
end

function vars_state_conservative(m::NTuple{N,Updraft}, FT) where {N}
    return Tuple{ntuple(i->vars_state_conservative(m[i], FT), N)...}
end


function vars_state_conservative(m::EDMF, FT)
    @vars(environment::vars_state_conservative(m.environment, FT),
          updraft::vars_state_conservative(m.updraft, FT)
          );
end

function vars_state_gradient(::Updraft, FT)
    @vars(u::SVector{3, FT},
          e_int::FT,
          e_int::FT,
          )
end

function vars_state_gradient(::Environment, FT)
    @vars(e_int::FT,
          q_tot::FT,
          u::SVector{3, FT},
          tke::TF,
          e_int_cv::TF,
          q_tot_cv::TF,
          e_int_q_tot_cv::TF,
          )
end


function vars_state_gradient(m::NTuple{N,Updraft}, FT) where {N}
    return Tuple{ntuple(i->vars_state_gradient(m[i], FT), N)...}
end


function vars_state_gradient(m::EDMF, FT)
    @vars(environment::vars_state_gradient(m.environment, FT),
          updraft::vars_state_gradient(m.updraft, FT)
          );
end

function vars_state_gradient_flux(m::NTuple{N,Updraft}, FT) where {N}
    return Tuple{ntuple(i->vars_state_gradient_flux(m[i], FT), N)...}
end

vars_state_gradient_flux(::Updraft, FT) = @vars()

function vars_state_gradient_flux(::Environment, FT)
    @vars(∇e_int::SVector{3, FT},
          ∇q_tot::SVector{3, FT},
          ∇u::SMatrix{3, 3, FT, 9},
          ∇tke::TF,
          ∇e_int_cv::TF,
          ∇q_tot_cv::TF,
          ∇e_int_q_tot_cv::TF,
          ∇θ_ρ::TF, # used in a diagnostic equation for the mixing length
          )
end

function vars_state_gradient_flux(m::EDMF, FT)
    @vars(environment::vars_state_gradient_flux(m.environment, FT),
          updraft::vars_state_gradient_flux(m.updraft, FT)
          );
end

# Specify the initial values in `aux::Vars`, which are available in
# `init_state_conservative!`. Note that
# - this method is only called at `t=0`
# - `aux.z` and `aux.T` are available here because we've specified `z` and `T`
# in `vars_state_auxiliary`
function init_state_auxiliary!(
        m::SingleStack{FT,N},
        edmf::EDMF{FT,N},
        aux::Vars,
        geom::LocalGeometry
    ) where {FT,N}
    # Compute the reference profile ρ_0,p_0 to be stored in grid mean auxiliary vars
    #       consider a flag for SingleStack setting that assigns gm.ρ = ρ_0; gm.p = p_0

    # status:
    # Need to find the right way to integrate the hydrostatic reference profile
    # to obtain both ρ_0 and p_0 by integrating log(p) in z based on dlog(p)/dz = -g/(R_m*T)
    # with constant θ_liq and q_tot
    # it is not clear to in function dynamically assigns p in LiquidIcePotTempSHumEquil_given_pressure
    # at each level

    # Aliases:
    en_a = aux.edmf.environment
    up_a = aux.edmf.updraft

    en_a.buoyancy = 0.0
    en_a.cld_frac = 0.0

    for i in 1:N
        up_a[i].buoyancy = 0.0
        up_a[i].upd_top = 0.0
    end

    en_a.T = aux.T
    en_a.cld_frac = m.cf_initial
end;

# The following two functions should compute the hydrostatic and adiabatic reference state
# this state integrates upwards the equations d(log(p))/dz = -g/(R_m*T)
# with q_tot and θ_liq at each z level equal their respective surface values.

function integral_load_auxiliary_state!(
    m::SingleStack{FT,N},
    edmf::EDMF{FT,N},
    integrand::Vars,
    state::Vars,
    aux::Vars,
)
    # need to define thermo_state with set values of thetali and qt from surface values (moist adiabatic)
    # something like _p = exp(input_logP) where input_logP is log of the p at the lower pressure  level
    ts = LiquidIcePotTempSHumEquil_given_pressure(param_set, m.θ_liq_flux_surf, _p, m.q_tot_flux_surf)
    q = PhasePartition(ts)
    T = air_temperature(ts)
    _R_m = gas_constant_air(param_set, q)
    integrand.a = -g / (Rm * T)
end

function integral_set_auxiliary_state!(
    m::SingleStack{FT,N},
    edmf::EDMF{FT,N},
    aux::Vars,
    integral::Vars,
)
    aux.int.a = integral.a + log(m.P_surf)
    aux.p_0 = exp(aux.int.a)
end

# Specify the initial values in `state::Vars`. Note that
# - this method is only called at `t=0`
# - `state.ρcT` is available here because we've specified `ρcT` in
# `vars_state_conservative`
function init_state_conservative!(
    m::SingleStack{FT,N},
    edmf::EDMF{FT,N},
    state::Vars,
    aux::Vars,
    coords,
    t::Real,
) where {FT,N}

    # Aliases:
    gm = state
    en = state.edmf.environment
    up = state.edmf.updraft

    # gm.ρ = aux.ref_state.ρ # quickly added at end

    # GCM setting - Initialize the grid mean profiles of prognostic variables (ρ,e_int,q_tot,u,v,w)
    z = aux.z

    # SCM setting - need to have separate cases coded and called from a folder - see what LES does
    # a moist_thermo state is used here to convert the input θ,q_tot to e_int, q_tot profile
    ts = LiquidIcePotTempSHumEquil_given_pressure(param_set, θ_liq, P, q_tot)
    T = air_temperature(ts)
    ρ = air_density(ts)

    a_up = m.a_updraft_initial/FT(N)
    for i in 1:N
        up[i].ρa = ρ * a_up
        up[i].ρau = gm.ρu * a_up
        up[i].ρae_int = gm.ρe_int * a_up
        up[i].ρaq_tot = gm.ρq_tot * a_up
    end

    # initialize environment covariance
    en.ρae_int_cv =
    en.ρaq_tot_cv =
    en.ρae_int_q_tot_cv =

end;

# The remaining methods, defined in this section, are called at every
# time-step in the solver by the [`BalanceLaw`](@ref
# ClimateMachine.DGMethods.BalanceLaw) framework.

# Overload `update_auxiliary_state!` to call `single_stack_nodal_update_aux!`, or
# any other auxiliary methods
function update_auxiliary_state!(
    dg::DGModel,
    m::SingleStack,
    edmf::EDMF,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    nodal_update_auxiliary_state!(edmf_stack_nodal_update_aux!, dg, m, Q, t, elems)
end;

# Compute/update all auxiliary variables at each node. Note that
# - `aux.T` is available here because we've specified `T` in
# `vars_state_auxiliary`
function edmf_stack_nodal_update_aux!(
    m::SingleStack{FT,N},
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT, N}

    en_a = aux.edmf.environment
    up_a = aux.edmf.updraft
    gm = state
    en = state.edmf.environment
    up = state.edmf.updraft

    #   -------------  Compute buoyancies of subdomains
    ρinv = 1/gm.ρ
    b_upds = 0
    a_upds = 0
    for i in 1:N
        # computing buoyancy with PhaseEquil (i.e. qv_star) that uses gm.ρ instead of ρ_i that is unknown
        ts = PhaseEquil(param_set ,up[i].e_int, gm.ρ, up[i].q_tot)

        ρ_i = air_density(ts)
        up_a[i].buoyancy = -grav*(ρ_i-aux.ref_state.ρ)*ρinv
        b_upds += up_a[i].buoyancy
        a_upds += up_a[i].ρa*ρinv
    end
    # compute the buoyancy of the environment
    env_e_int = (gm.e_int - up[i].e_int*up[i].ρa*ρinv)/(1-up[i].ρa*ρinv)
    env_q_tot = (gm.q_tot - up[i].q_tot*up[i].ρa*ρinv)/(1-up[i].ρa*ρinv)
    ts = PhaseEquil(param_set ,env_e_int, gm.ρ, env_q_tot)
    env_ρ = air_density(ts)
    env_q_liq = PhasePartition(ts).liq
    env_q_ice = PhasePartition(ts).ice
    b_env = -grav*(env_ρ-aux.ref_state.ρ)*ρinv
    # subtract the grid mean
    b_gm = (1 - a_ups)*b_env
    for i in 1:N
        b_gm += up_a[i].buoayncy*up_a[i].ρa*ρinv
    end
    up_a[i].buoyancy -= b_gm

    #   -------------  Compute upd_top
    for i in 1:N
        # YAIR - check this with Charlie 
        J=1
        for j in length(up[i].ρa)
            if up[i].ρa*ρinv>FT(0)
                up[i].updraft_top = aux.z[J]
            end
        end
    end
end;

# Since we have second-order fluxes, we must tell `ClimateMachine` to compute
# the gradient of `ρcT`. Here, we specify how `ρcT` is computed. Note that
#  - `transform.ρcT` is available here because we've specified `ρcT` in
#  `vars_state_gradient`
function compute_gradient_argument!(
    m::SingleStack{FT,N},
    edmf::EDMF{FT,N},
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT, N}
    # Aliases:
    up_t = transform.edmf.updraft
    en_t = transform.edmf.environment
    gm = state
    up = state.edmf.updraft
    en = state.edmf.environment

    ρ = gm.ρ
    env_u = (gm.u - up[i].u*up[i].ρa*ρinv)/(1-up[i].ρa*ρinv)
    env_e_int = (gm.e_int - up[i].e_int*up[i].ρa*ρinv)/(1-up[i].ρa*ρinv)
    env_q_tot = (gm.q_tot - up[i].q_tot*up[i].ρa*ρinv)/(1-up[i].ρa*ρinv)
    en_t.u     = env_u
    en_t.q_tot = env_q_tot
    en_t.e_int = env_e_int

    ts = thermo_state(SingleStack, state, aux)
    en_t.θ_ρ = virtual_pottemp(ts)

    # for i in 1:N
    #     up_t[i].u = up[i].ρau/up[i].ρa
    # end
end;

# Specify where in `diffusive::Vars` to store the computed gradient from
# `compute_gradient_argument!`. Note that:
#  - `diffusive.α∇ρcT` is available here because we've specified `α∇ρcT` in
#  `vars_state_gradient_flux`
#  - `∇transform.ρcT` is available here because we've specified `ρcT`  in
#  `vars_state_gradient`
function compute_gradient_flux!(
    m::SingleStack{FT,N},
    edmf::EDMF{FT,N},
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT,N}
    # Aliases:
    gm_d = diffusive
    up_d = diffusive.edmf.updraft
    en_d = diffusive.edmf.environment
    gm_∇t = ∇transform
    up_∇t = ∇transform.edmf.updraft
    en_∇t = ∇transform.edmf.environment
    gm = state
    up = state.edmf.updraft
    # gm_d.α∇ρcT = -m.α * gm_∇t.ρcT
    # gm_d.μ∇u = -m.μ * gm_∇t.u
    ρinv = 1/gm.ρ
    en_d.∇θ_ρ = en_∇t.θ_ρ

    for i in 1:N
        up_d[i].∇u = up_∇t[i].u
    end
end;

# We have no sources, nor non-diffusive fluxes.
function source!(
    m::SingleStack{FT, N},
    edmf::EDMF{FT, N},
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
) where {FT, N}

    # Aliases:
    gm = state
    en = state
    up = state.edmf.updraft
    en_s = source
    up_s = source.edmf.updraft

    # should be conditioned on updraft_area > minval
    a_env = 1-sum([up[i].ρa for i in 1:N])*ρinv
    ρinv = 1/gm.ρ
    for i in 1:N
        # get environment values for e_int, q_tot , u[3]
        env_u = (gm.u - up[i].u*up[i].ρa*ρinv)/(1-up[i].ρa*ρinv)
        env_e_int = (gm.e_int - up[i].e_int*up[i].ρa*ρinv)/(1-up[i].ρa*ρinv)
        env_q_tot = (gm.q_tot - up[i].q_tot*up[i].ρa*ρinv)/(1-up[i].ρa*ρinv)

        # first moment sources
        εt, ε, δ = entr_detr(m, m.edmf.entr_detr, state, diffusive, aux, t, direction, i)
        l = mixing_length(m, m.edmf.mix_len, source, state, diffusive, aux, t, direction, δ, εt)
        K_eddy = m.c_k*l*sqrt(en.tke)
        dpdz, dpdz_tke_i = perturbation_pressure(m, m.edmf.pressure, source, state, diffusive, aux, t, direction, i)

           # entrainment and detrainment
        w_i = up[i].ρu[3]*ρinv
        up_s[i].ρa      += up[i].ρa * w_i * (ε - δ)
        up_s[i].ρau     += up[i].ρa * w_i * ((ε+εt)*up_s[i].ρau     - (δ+εt)*env_u)
        up_s[i].ρae_int += up[i].ρa * w_i * ((ε+εt)*up_s[i].ρae_int - (δ+εt)*env_e_int)
        up_s[i].ρaq_tot += up[i].ρa * w_i * ((ε+εt)*up_s[i].ρaq_tot - (δ+εt)*env_q_tot)

           # perturbation pressure
        up_s[i].ρau[3]  += up[i].ρa * dpdz

        # second moment sources
        en.ρatke += dpdz_tke_i

        # sources  for the grid mean
    end
end;

function flux_first_order!(
    m::SingleStack{FT,N},
    edmf::EDMF{FT, N}
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT,N}

    # Aliases:
    up = state.edmf.updraft
    up_f = flux.edmf.updraft

    # up
    for i in 1:N
        up_f[i].ρa = up[i].ρau
        u = up[i].ρau / up[i].ρa
        up_f[i].ρau = up[i].ρau * u'
        up_f[i].ρacT = u * up[i].ρacT
    end

end;

function flux_second_order!(
    m::SingleStack{FT,N},
    edmf::EDMF{FT,N},
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
) where {FT,N}
    # Aliases:
    gm   = state
    up   = state.edmf.updraft
    en   = state.edmf.environment

    # compute the mixing length and eddy diffusivity
    # (I am repeating this after doing in sources assuming that it is
    # better to compute this twice than to add mixing length as a aux.variable)
    l = mixing_length(m, m.edmf.mix_len, source, state, diffusive, aux, t, direction, δ, εt)
    K_eddy = m.c_k*l*sqrt(en.tke)
    # flux_second_order in the grid mean is the environment turbulent diffusion
    en_ρa = gm.ρ-sum([up[i].ρa for i in 1:N])
end;

# ### Boundary conditions

# Second-order terms in our equations, ``∇⋅(G)`` where ``G = α∇ρcT``, are
# internally reformulated to first-order unknowns.
# Boundary conditions must be specified for all unknowns, both first-order and
# second-order unknowns which have been reformulated.

# The boundary conditions for `ρcT` (first order unknown)
function boundary_state!(
    nf,
    m::SingleStack{FT,N},
    edmf::EDMF{FT,N},
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
) where {FT,N}
    gm = state⁺
    up = state⁺.edmf.updraft
    if bctype == 1 # bottom
        # placeholder to add a function for updraft surface value
        # this function should use surface covariance in the grid mean from a corresponding function
        for i in 1:N
            upd_a_surf, upd_e_int_surf, upd_q_tot_surf  = compute_updraft_surface_BC(i)
            up[i].ρau = SVector(0,0,0)
            up[i].ρa = upd_a_surf
            up[i].ρae_int = upd_e_int_surf
            up[i].ρaq_tot = upd_q_tot_surf
        end

    elseif bctype == 2 # top
        # if yes not BC on upd are needed at the top (currently set to GM)
        # if not many issues might  come up with area fraction at the upd top

        for i in 1:N
            up[i].ρau = SVector(0,0,0)
            up[i].ρa = 0.0
            up[i].ρae_int = gm.ρe_int*up[i].ρa*ρinv
            up[i].ρaq_tot = gm.ρq_tot*up[i].ρa*ρinv
        end

    end
end;

# The boundary conditions for `ρcT` are specified here for second-order
# unknowns
function boundary_state!(
    nf,
    m::SingleStack{FT,N},
    edmf::EDMF{FT,N},
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
) where {FT,N}
    gm = state⁺
    up = state⁺.edmf.updraft
    gm_d = diff⁺
    up_d = diff⁺.edmf.updraft
    if bctype == 1 # bottom
        # YAIR - I need to pass the SurfaceModel into BC and into env_surface_covariances
        tke, e_int_cv ,q_tot_cv ,e_int_q_tot_cv = env_surface_covariances(ss, m, edmf, source, state)
        en_d.ρatke = gm.ρ * area_en * tke
        en_d.ρae_int_cv = gm.ρ * area_en * e_int_cv
        en_d.ρaq_tot_cv = gm.ρ * area_en * q_tot_cv
        en_d.ρae_int_q_tot_cv = gm.ρ * area_en * e_int_q_tot_cv
    elseif bctype == 2 # top
        # for now zero flux at the top
        en_d.ρatke = -n⁻ * 0.0
        en_d.ρae_int_cv = -n⁻ * 0.0
        en_d.ρaq_tot_cv = -n⁻ * 0.0
        en_d.ρae_int_q_tot_cv = -n⁻ * 0.0
    end
end;
