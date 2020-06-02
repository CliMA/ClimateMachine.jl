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
    vars_state_conservative(::L, FT)

a tuple of symbols containing the state variables
given a float type `FT`.
"""
function vars_state_conservative end

"""
    vars_state_auxiliary(::L, FT)

a tuple of symbols containing the auxiliary variables
given a float type `FT`.
"""
function vars_state_auxiliary end

"""
    vars_state_gradient(::L, FT)

a tuple of symbols containing the transformed variables
of which gradients are computed given a float type `FT`.
"""
function vars_state_gradient end

"""
    vars_state_gradient_flux(::L, FT)

a tuple of symbols containing the diffusive variables
given a float type `FT`.
"""
function vars_state_gradient_flux end

"""
    vars_gradient_laplacian(::L, FT)

a tuple of symbols containing the transformed variables
of which gradients of laplacian are computed, they must
be a subset of `vars_state_gradient`, given a float type `FT`.
"""
vars_gradient_laplacian(::BalanceLaw, FT) = @vars()

"""
    vars_hyperdiffusive(::L, FT)

a tuple of symbols containing the hyperdiffusive variables
given a float type `FT`.
"""
vars_hyperdiffusive(::BalanceLaw, FT) = @vars()

"""
    vars_integrals(::L, FT)

a tuple of symbols containing variables to be integrated
along a vertical stack, given a float type `FT`.
"""
vars_integrals(::BalanceLaw, FT) = @vars()

"""
    vars_reverse_integrals(::L, FT)

a tuple of symbols containing variables to be integrated
along a vertical stack, in reverse, given a float type `FT`.
"""
vars_reverse_integrals(::BalanceLaw, FT) = @vars()

"""
    init_state_conservative!(
      ::L,
      state_conservative::Vars,
      state_auxiliary::Vars,
      coords,
      args...)

Initialize the conservative state variables at ``t = 0``
"""
function init_state_conservative! end

"""
    init_state_auxiliary!(
      ::L,
      state_auxiliary::Vars,
      coords,
      args...)

Initialize the auxiliary state, at ``t = 0``
"""
function init_state_auxiliary! end

"""
    flux_first_order!(
        ::L,
        flux::Grad,
        state_conservative::Vars,
        state_auxiliary::Vars,
        t::Real
    )

Compute first-order flux terms in balance law equation
"""
function flux_first_order! end

"""
    flux_second_order!(
        ::L,
        flux::Grad,
        state_conservative::Vars,
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
        state_conservative::Vars,
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
        state_conservative::Vars,
        state_auxiliary::Vars,
        t::Real
    )

transformation of state variables to variables of which gradients are computed
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
        state_conservative::Vars,
        state_auxiliary::Vars,
        t::Real
    )

wavespeed
"""
function wavespeed end

"""
    boundary_state!(
        ::NumericalFluxGradient,
        ::L,
        state_conservative⁺::Vars,
        state_auxiliary⁺::Vars,
        normal⁻,
        state_conservative⁻::Vars,
        state_auxiliary⁻::Vars,
        bctype,
        t
    )
    boundary_state!(
        ::NumericalFluxFirstOrder,
        ::L,
        state_conservative⁺::Vars,
        state_auxiliary⁺::Vars,
        normal⁻,
        state_conservative⁻::Vars,
        state_auxiliary⁻::Vars,
        bctype,
        t
    )
    boundary_state!(
        ::NumericalFluxSecondOrder,
        ::L,
        state_conservative⁺::Vars,
        state_gradient_flux⁺::Vars,
        state_auxiliary⁺:
        Vars, normal⁻,
        state_conservative⁻::Vars,
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
        m::HBModel,
        Q::MPIStateArray,
        t::Real,
        elems::UnitRange,
    )

Update the auxiliary state variables
"""
function update_auxiliary_state! end

"""
    update_auxiliary_state_gradient!


Update the auxiliary state gradient variables
"""
function update_auxiliary_state_gradient! end

"""
    integral_load_auxiliary_state!

Specify how to compute integrands. Can be functions of the conservative state and auxiliary variables.
"""
function integral_load_auxiliary_state! end

"""
    integral_set_auxiliary_state!

Specify which auxiliary variables are used to store the output of the integrals.
"""
function integral_set_auxiliary_state! end

"""
    reverse_integral_load_auxiliary_state!

Specify auxiliary variables need their integrals reversed.
"""
function reverse_integral_load_auxiliary_state! end

"""
    reverse_integral_set_auxiliary_state!

Specify which auxiliary variables are used to store the output of the reversed integrals.
"""
function reverse_integral_set_auxiliary_state! end

# Internal methods
number_state_conservative(m::BalanceLaw, FT) =
    varsize(vars_state_conservative(m, FT))
number_state_auxiliary(m::BalanceLaw, FT) = varsize(vars_state_auxiliary(m, FT))
number_state_gradient(m::BalanceLaw, FT) = varsize(vars_state_gradient(m, FT))
number_state_gradient_flux(m::BalanceLaw, FT) =
    varsize(vars_state_gradient_flux(m, FT))
num_gradient_laplacian(m::BalanceLaw, FT) =
    varsize(vars_gradient_laplacian(m, FT))
num_hyperdiffusive(m::BalanceLaw, FT) = varsize(vars_hyperdiffusive(m, FT))
num_integrals(m::BalanceLaw, FT) = varsize(vars_integrals(m, FT))
num_reverse_integrals(m::BalanceLaw, FT) =
    varsize(vars_reverse_integrals(m, FT))
