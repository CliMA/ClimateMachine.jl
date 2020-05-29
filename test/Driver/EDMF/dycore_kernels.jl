#### Dynamical core kernels

# Specify auxiliary variables for `SingleStack`

function vars_state_auxiliary(m::SingleStack, FT)
    @vars(z::FT,
          buoyancy::FT,
          ρ_0::FT,
          p_0::FT,
          edmf::vars_state_auxiliary(m.edmf, FT),
          # ref_state::vars_state_auxiliary(m.ref_state, FT) # quickly added at end
          );
end

function vars_state_conservative(m::SingleStack, FT)
    @vars(ρ::FT,
          ρu::SVector{3, FT},
          ρe_int::FT,
          ρq_tot::FT,
          edmf::vars_state_conservative(m.edmf, FT));
end

function vars_state_gradient(m::SingleStack, FT)
    @vars(edmf::vars_state_gradient(m.edmf, FT));
end

function vars_state_gradient_flux(m::SingleStack, FT)
    @vars(edmf::vars_state_gradient_flux(m.edmf, FT));
end
# ## Define the compute kernels

# Specify the initial values in `aux::Vars`, which are available in
# `init_state_conservative!`. Note that
# - this method is only called at `t=0`
# - `aux.z` and `aux.T` are available here because we've specified `z` and `T`
# in `vars_state_auxiliary`
function init_state_auxiliary!(m::SingleStack{FT,N}, aux::Vars, geom::LocalGeometry) where {FT,N}
    aux.z = geom.coord[3]

    # Compute the reference profile ρ_0, p_0 
    #       consider a flag for SingleStack setting that assigns gm.ρ = ρ_0; gm.p = p_0

    # status:
    # Need to find the right way to integrate the hydrostatic reference profile
    # to obtain both ρ_0 and p_0 by integrating log(p) in z based on dlog(p)/dz = -g/(R_m*T)
    # with constant θ_liq and q_tot
    # it is not clear to me if the function dynamically assigns p in LiquidIcePotTempSHumEquil_given_pressure
    # at each level


    # aux.ref_state.p = p_0
    # aux.ref_state.ρ = p_0 / (R_m * T)
    # R_m = R_m()

    aux.buoyancy = 0.0
    aux.ρ_0 = hydrostatic_ref()  # yair - this needs to be computed consistently by the  functions below
    aux.p_0 = hydrostatic_ref()  # yair - this needs to be computed consistently by the  functions below
    init_state_auxiliary!(m, m.edmf, aux, geom)
end;

# The following two functions should compute the hydrostatic and adiabatic reference state
# this state integrates upwards the equations d(log(p))/dz = -g/(R_m*T)
# with q_tot and θ_liq at each z level equal their respective surface values.

function integral_load_auxiliary_state!(
    m::SingleStack{FT,N},
    integrand::Vars,
    state::Vars,
    aux::Vars,
)
    integral_load_auxiliary_state!(m, m.edmf, integrand, state, aux)
end

function integral_set_auxiliary_state!(
    m::SingleStack{FT,N},
    aux::Vars,
    integral::Vars,
)
    integral_set_auxiliary_state!(m, m.edmf, aux, integral)
end

# Specify the initial values in `state::Vars`. Note that
# - this method is only called at `t=0`
# - `state.ρcT` is available here because we've specified `ρcT` in
# `vars_state_conservative`
function init_state_conservative!(
    m::SingleStack{FT,N},
    state::Vars,
    aux::Vars,
    coords,
    t::Real,
) where {FT,N}

    state.ρ = 1
    state.ρu = 0
    state.ρe_int = 0
    state.ρq_tot = 0
    init_state_conservative!(m, m.edmf, state, aux, coords, t)

end;

# The remaining methods, defined in this section, are called at every
# time-step in the solver by the [`BalanceLaw`](@ref
# ClimateMachine.DGMethods.BalanceLaw) framework.

# Overload `update_auxiliary_state!` to call `single_stack_nodal_update_aux!`, or
# any other auxiliary methods
function update_auxiliary_state!(
    dg::DGModel,
    m::SingleStack,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    nodal_update_auxiliary_state!(single_stack_nodal_update_aux!, dg, m, Q, t, elems)
    update_auxiliary_state!(dg, m, m.edmf, Q, t, elems)
end;

# Compute/update all auxiliary variables at each node. Note that
# - `aux.T` is available here because we've specified `T` in
# `vars_state_auxiliary`
function single_stack_nodal_update_aux!(
    m::SingleStack{FT,N},
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT, N}
    aux.buoyancy = 0
end;

# Since we have second-order fluxes, we must tell `ClimateMachine` to compute
# the gradient of `ρcT`. Here, we specify how `ρcT` is computed. Note that
#  - `transform.ρcT` is available here because we've specified `ρcT` in
#  `vars_state_gradient`
function compute_gradient_argument!(
    m::SingleStack{FT,N},
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT, N}
    compute_gradient_argument!(m, m.edmf, state, aux, t)
end;

# Specify where in `diffusive::Vars` to store the computed gradient from
# `compute_gradient_argument!`. Note that:
#  - `diffusive.α∇ρcT` is available here because we've specified `α∇ρcT` in
#  `vars_state_gradient_flux`
#  - `∇transform.ρcT` is available here because we've specified `ρcT`  in
#  `vars_state_gradient`
function compute_gradient_flux!(
    m::SingleStack{FT,N},
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT,N}
    compute_gradient_flux!(m, m.edmf, diffusive, ∇transform, state, aux, t)
end;

# We have no sources, nor non-diffusive fluxes.
function source!(
    m::SingleStack{FT, N},
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
) where {FT, N}
    source!(m, m.edmf, state, diffusive, aux, t, direction)
end;

function flux_first_order!(
    m::SingleStack,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)

    ρinv = 1/state.ρ
    flux.ρ = state.ρu
    u = state.ρu * ρinv
    flux.ρu = state.ρu * u'
    flux.ρe_int = u * state.ρe_int
    flux.ρq_tot = u * state.ρq_tot
    flux_first_order!(m, m.edmf, flux, state, aux, t)
end;

function flux_second_order!(
    m::SingleStack{FT,N},
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
) where {FT,N}
    # TODO: This should be in sync with `flux_second_order!(::AtmosModel)`, plus
    # the call to `flux_second_order!(m, m.edmf, flux, state, diffusive, hyperdiffusive, aux, t)`

    # Aliases:
    up   = state.edmf.updraft
    en   = state.edmf.environment
    en_d = diffusive.edmf.environment

    # compute the mixing length and eddy diffusivity
    # (I am repeating this after doing in sources assuming that it is better to compute this twice than to add mixing length as a aux.variable)
    l = mixing_length(m, m.edmf.mix_len, source, state, diffusive, aux, t, direction, δ, εt)
    K_eddy = m.c_k*l*sqrt(en.tke)
    # flux_second_order in the grid mean is the environment turbulent diffusion
    en_ρa = state.ρ-sum([up[i].ρa for i in 1:N])
    flux.ρe_int += en_ρa*K_eddy*en_d.∇e_int # check Prantl number here
    flux.ρq_tot += en_ρa*K_eddy*en_d.∇q_tot # check Prantl number here
    flux.ρu     += en_ρa*K_eddy*en_d.∇u     # check Prantl number here
    flux_second_order!(m, m.edmf, flux, state, diffusive, hyperdiffusive, aux, t)
end;

# ### Boundary conditions

# Second-order terms in our equations, ``∇⋅(G)`` where ``G = α∇ρe_int``, are
# internally reformulated to first-order unknowns.
# Boundary conditions must be specified for all unknowns, both first-order and
# second-order unknowns which have been reformulated.

# The boundary conditions for `ρe_int` (first order unknown)
function boundary_state!(
    nf,
    m::SingleStack{FT,N},
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    args...,
) where {FT,N}
    if bctype == 1 # bottom
        state⁺.ρ = m.surface_ρ # find out how is the density at the surface computed in the LES?
        state⁺.ρu = SVector(0,0,0)
        state⁺.ρe_int = state⁺.ρ * m.surface_e_int
        state⁺.ρq_tot = state⁺.ρ * m.surface_q_tot
    elseif bctype == 2 # top
        state⁺.ρ = # placeholder to find out how density at the top is computed in the LES?
        state⁺.ρu = SVector(0,0,0)
    end
    boundary_state!(nf, m, m.edmf, state⁺, aux⁺, n⁻, state⁻, aux⁻, bctype, t, args...)
end;

# The boundary conditions for `ρe_int` are specified here for second-order
# unknowns
function boundary_state!(
    nf,
    m::SingleStack{FT,N},
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    args...,
) where {FT,N}
    if bctype == 1 # bottom
        diff⁺.ρ∇e_int = state⁺.ρ * m.e_int_surface_flux # e_int_surface_flux has units of w'e_int'
        diff⁺.ρ∇q_tot = state⁺.ρ * m.q_tot_surface_flux # q_tot_surface_flux has units of w'q_tot'
        # diff⁺.ρ∇u = SVector(...) # probably need one of these...
        # diff⁺.ρ∇u = SMatrix(...) # probably need one of these...
        diff⁺.ρ∇u[1] = state⁺.ρ * m.u_surface_flux # u_surface_flux has units of w'u'
        diff⁺.ρ∇u[2] = state⁺.ρ * m.v_surface_flux # v_surface_flux has units of w'v'

        diff⁺.ρ∇e_int = state⁺.ρ * m.e_int_surface_flux # e_int_surface_flux has units of w'e_int'
        diff⁺.ρ∇q_tot = state⁺.ρ * m.q_tot_surface_flux # q_tot_surface_flux has units of w'q_tot'

    elseif bctype == 2 # top
        diff⁺.ρ∇u[1] = -n⁻ * FT(0)
        diff⁺.ρ∇u[2] = -n⁻ * FT(0)
        diff⁺.ρ∇e_int = -n⁻ * FT(0)
        diff⁺.ρ∇q_tot = -n⁻ * FT(0)
    end
    boundary_state!(nf, m, m.edmf, state⁺, diff⁺, aux⁺, n⁻, state⁻, diff⁻, aux⁻, bctype, t, args)
end;
