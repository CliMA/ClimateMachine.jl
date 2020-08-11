#### Balance Law Interface

"""
    BalanceLaw

An abstract type representing a PDE balance law of the form

elements for balance laws of the form

```math
q_{,t} + Σ_{i=1,...d} F_{i,i} = s
```

Subtypes `L` should define the methods below
"""
abstract type BalanceLaw end # PDE part

"""
    vars_state(::L, ::AbstractStateType, FT)

a tuple of symbols containing the state variables
given a float type `FT`.
"""
function vars_state end

# Fallback: no variables
vars_state(::BalanceLaw, ::AbstractStateType, FT) = @vars()

"""
    init_state_prognostic!(
      ::L,
      state_prognostic::Vars,
      state_auxiliary::Vars,
      coords,
      args...)

Initialize the prognostic state variables at ``t = 0``
"""
function init_state_prognostic! end

"""
    init_state_auxiliary!(
      ::L,
      state_auxiliary::MPIStateArray,
      grid)

Initialize the auxiliary state, at ``t = 0``
"""
function init_state_auxiliary! end

"""
    flux_first_order!(
        ::L,
        flux::Grad,
        state_prognostic::Vars,
        state_auxiliary::Vars,
        t::Real,
        directions
    )

Compute first-order flux terms in balance law equation
"""
function flux_first_order! end

"""
    flux_second_order!(
        ::L,
        flux::Grad,
        state_prognostic::Vars,
        state_gradient_flux::Vars,
        hyperdiffusive::Vars,
        state_auxiliary::Vars,
        t::Real
    )

Compute second-order flux terms in balance law equation
"""
function flux_second_order! end

"""
    source!(
        ::L,
        source::Vars,
        state_prognostic::Vars,
        diffusive::Vars,
        state_auxiliary::Vars,
        t::Real
    )

Compute non-conservative source terms in balance law equation
"""
function source! end

"""
    compute_gradient_argument!(
        ::L,
        transformstate::Vars,
        state_prognostic::Vars,
        state_auxiliary::Vars,
        t::Real
    )

transformation of state variables to variables of which gradients are
computed
"""
compute_gradient_argument!(::BalanceLaw, args...) = nothing

"""
    compute_gradient_flux!(
        ::L,
        state_gradient_flux::Vars,
        ∇transformstate::Grad,
        state_auxiliary::Vars,
        t::Real
    )

transformation of gradients to the diffusive variables
"""
compute_gradient_flux!(::BalanceLaw, args...) = nothing

"""
    transform_post_gradient_laplacian!(
        ::L,
        Qhypervisc_div::Vars,
        ∇Δtransformstate::Grad,
        state_auxiliary::Vars,
        t::Real
    )

transformation of laplacian gradients to the hyperdiffusive variables
"""
function transform_post_gradient_laplacian! end

"""
    wavespeed(
        ::L,
        n⁻,
        state_prognostic::Vars,
        state_auxiliary::Vars,
        t::Real,
        direction
    )

wavespeed
"""
function wavespeed end

"""
    boundary_state!(
        ::NumericalFluxGradient,
        ::L,
        state_prognostic⁺::Vars,
        state_auxiliary⁺::Vars,
        normal⁻,
        state_prognostic⁻::Vars,
        state_auxiliary⁻::Vars,
        bctype,
        t
    )
    boundary_state!(
        ::NumericalFluxFirstOrder,
        ::L,
        state_prognostic⁺::Vars,
        state_auxiliary⁺::Vars,
        normal⁻,
        state_prognostic⁻::Vars,
        state_auxiliary⁻::Vars,
        bctype,
        t
    )
    boundary_state!(
        ::NumericalFluxSecondOrder,
        ::L,
        state_prognostic⁺::Vars,
        state_gradient_flux⁺::Vars,
        state_auxiliary⁺:
        Vars, normal⁻,
        state_prognostic⁻::Vars,
        state_gradient_flux⁻::Vars,
        state_auxiliary⁻::Vars,
        bctype,
        t
    )

Apply boundary conditions for

 - `NumericalFluxGradient` numerical flux (internal method)
 - `NumericalFluxFirstOrder` first-order unknowns
 - `NumericalFluxSecondOrder` second-order unknowns

"""
function boundary_state! end

"""
    update_auxiliary_state!(
        dg::DGModel,
        m::BalanceLaw,
        Q::MPIStateArray,
        t::Real,
        elems::UnitRange,
    )

Update the auxiliary state variables with global scope.
"""
function update_auxiliary_state! end

"""
    nodal_update_auxiliary_state!()

Update the auxiliary state variables at each location in space.
"""
function nodal_update_auxiliary_state! end

"""
    update_auxiliary_state_gradient!


Update the auxiliary state gradient variables
"""
function update_auxiliary_state_gradient! end

"""
    integral_load_auxiliary_state!

Specify how to compute integrands. Can be functions of the prognostic
state and auxiliary variables.
"""
function integral_load_auxiliary_state! end

"""
    integral_set_auxiliary_state!

Specify which auxiliary variables are used to store the output of the
integrals.
"""
function integral_set_auxiliary_state! end

"""
    indefinite_stack_integral!

Compute indefinite integral along stack.
"""
function indefinite_stack_integral! end

"""
    reverse_integral_load_auxiliary_state!

Specify auxiliary variables need their integrals reversed.
"""
function reverse_integral_load_auxiliary_state! end

"""
    reverse_integral_set_auxiliary_state!

Specify which auxiliary variables are used to store the output of the
reversed integrals.
"""
function reverse_integral_set_auxiliary_state! end

"""
    reverse_indefinite_stack_integral!

Compute reverse indefinite integral along stack.
"""
function reverse_indefinite_stack_integral! end

# Internal methods
number_states(m::BalanceLaw, st::AbstractStateType, FT = Int) =
    varsize(vars_state(m, st, FT))

### split explicit functions
function initialize_states! end
function tendency_from_slow_to_fast! end
function cummulate_fast_solution! end
function reconcile_from_fast_to_slow! end
