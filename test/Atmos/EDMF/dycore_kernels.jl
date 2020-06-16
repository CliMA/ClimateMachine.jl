#### Dynamical core kernels

# Reference state
abstract type ReferenceState end
struct HydrostaticState{P, FT} <: ReferenceState
    virtual_temperature_profile::P
    relative_humidity::FT
end
function HydrostaticState(
    virtual_temperature_profile::TemperatureProfile{FT},
) where {FT}
    return HydrostaticState{typeof(virtual_temperature_profile), FT}(
        virtual_temperature_profile,
        FT(0),
    )
end

# Specify auxiliary variables for `SingleStack`

function vars_state_auxiliary(m::SingleStack, FT)
    @vars(z::FT,
          buoyancy::FT,
          ρ0::FT,
          p0::FT,
          edmf::vars_state_auxiliary(m.edmf, FT),
          # ref_state::vars_state_auxiliary(m.ref_state, FT) # quickly added at end
          );
end

function vars_state_conservative(m::SingleStack, FT)
    @vars(ρ::FT,
          ρu::SVector{3, FT},
          ρe::FT,
          ρq_tot::FT, # We could add moisture model (similar to atmos model)
          edmf::vars_state_conservative(m.edmf, FT));
end

function vars_state_gradient(m::SingleStack, FT)
    @vars(p0::FT,
          e::FT,
          q_tot::FT, # We could add moisture model (similar to atmos model)
          u::FT,
          edmf::vars_state_gradient(m.edmf, FT));
end

function vars_state_gradient_flux(m::SingleStack, FT)
    @vars(∇e::SVector{3, FT},
          ∇q_tot::SVector{3, FT},
          ∇u::SMatrix{3, 3, FT, 9},
          ∇p0::SVector{3, FT},
          edmf::vars_state_gradient_flux(m.edmf, FT));
end
# ## Define the compute kernels

# Specify the initial values in `aux::Vars`, which are available in
# `init_state_conservative!`. Note that
# - this method is only called at `t=0`
# - `aux.z` and `aux.T` are available here because we've specified `z` and `T`
# in `vars_state_auxiliary`

# Overload `update_auxiliary_state!` to call `single_stack_nodal_update_aux!`, or
# any other auxiliary methods
function init_state_auxiliary!(m::SingleStack{FT,N}, aux::Vars, geom::LocalGeometry) where {FT,N}
    aux.z = geom.coord[3]
    T_profile = DecayingTemperatureProfile{FT}(m.param_set)
    ref_state = HydrostaticState(T_profile)
    T_virt, p = ref_state.virtual_temperature_profile(m.param_set, aux.z)
    _R_d = FT(R_d(m.param_set))

    aux.buoyancy = 0
    aux.ρ0 = p / (_R_d * T_virt)
    aux.p0 = p

    init_state_auxiliary!(m, m.edmf, aux, geom)
end;

function init_state_conservative!(
    m::SingleStack{FT,N},
    state::Vars,
    aux::Vars,
    coords,
    t::Real,
) where {FT,N}

    state.ρ = aux.ρ0
    state.ρu = SVector(0,0,0)
    state.ρe     = FT(300000) # need to add intial state here
    state.ρq_tot = eps(FT) # need to add intial state here
    init_state_conservative!(m, m.edmf, state, aux, coords, t)

end;

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
function single_stack_nodal_update_aux!(
    m::SingleStack{FT,N},
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT, N}
    _grav::FT = grav(m.param_set)
    aux.buoyancy = _grav*(state.ρ-aux.ρ0)/state.ρ
end;

# Since we have second-order fluxes, we must tell `ClimateMachine` to compute
function compute_gradient_argument!(
    m::SingleStack{FT,N},
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT, N}
    transform.p0 = aux.p0
    compute_gradient_argument!(m, m.edmf, state, aux, t)
end;

# Specify where in `diffusive::Vars` to store the computed gradient from
function compute_gradient_flux!(
    m::SingleStack{FT,N},
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT,N}
    diffusive.∇p0 = ∇transform.p0
    compute_gradient_flux!(m, m.edmf, diffusive, ∇transform, state, aux, t)
end;

# We have no sources, nor non-diffusive fluxes.
function source!(
    m::SingleStack{FT, N},
    state::Vars,
    source::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
) where {FT, N}
    source!(m, m.edmf, state, source, diffusive, aux, t, direction)
end;

function flux_first_order!(
    m::SingleStack,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    # in single column setting ⟨w⟩=0
    ρinv = 1/state.ρ
    flux.ρ = state.ρu
    u = state.ρu * ρinv
    flux.ρu = state.ρu * u'
    flux.ρe = u * state.ρe
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
    # SGS contributions to flux_second_order for dycore are updated in EDMF
    flux_second_order!(m, m.edmf, flux, state, diffusive, hyperdiffusive, aux, t)
end;

# ### Boundary conditions

# Second-order terms in our equations, ``∇⋅(G)`` where ``G = α∇ρe``, are
# internally reformulated to first-order unknowns.
# Boundary conditions must be specified for all unknowns, both first-order and
# second-order unknowns which have been reformulated.

# The boundary conditions for `ρe` (first order unknown)
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
        state⁺.ρ = FT(1) # m.surface_ρ # find out how is the density at the surface computed in the LES?
        state⁺.ρu = SVector(0,0,0)
        state⁺.ρe = state⁺.ρ #* m.surface_e
        state⁺.ρq_tot = state⁺.ρ #* m.surface_q_tot
    elseif bctype == 2 # top
        state⁺.ρ = FT(1) # placeholder to find out how density at the top is computed in the LES?
        state⁺.ρu = SVector(0,0,0)
    end
    boundary_state!(nf, m, m.edmf, state⁺, aux⁺, n⁻, state⁻, aux⁻, bctype, t, args...)
end;

# The boundary conditions for `ρe` are specified here for second-order
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
        diff⁺.∇e     = n⁻ * state⁺.ρ * m.edmf.surface.surface_shf # e_surface_flux has units of w'e'
        diff⁺.∇q_tot = n⁻ * state⁺.ρ * m.edmf.surface.surface_lhf # q_tot_surface_flux has units of w'q_tot'
        # diff⁺.ρ∇u = SVector(...) # probably need one of these...
        # diff⁺.ρ∇u = SMatrix(...) # probably need one of these...
    
    elseif bctype == 2 # top
        # diff⁺.∇u[1]  = -n⁻ * eps(FT)
        # diff⁺.∇u[2]  = -n⁻ * eps(FT)
        diff⁺.∇e     = -n⁻ * eps(FT)
        diff⁺.∇q_tot = -n⁻ * eps(FT)
    end
    boundary_state!(nf, m, m.edmf, state⁺, diff⁺, aux⁺, n⁻, state⁻, diff⁻, aux⁻, bctype, t, args)
end;
