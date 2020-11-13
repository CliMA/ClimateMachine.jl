# eventually boundary conditions should be a subtype of this
# we don't enforce it currently to make the transition easier
abstract type BoundaryCondition end


"""
    boundary_conditions(::BL)

Define the set of boundary conditions for the balance law `BL`. This should
return a tuple, where a boundary tagged with the integer `i` will use the `i`th
element of the tuple.
"""
function boundary_conditions end


"""
    boundary_state!(
        ::NumericalFluxGradient,
        ::BC
        ::BL,
        state_prognostic⁺::Vars,
        state_auxiliary⁺::Vars,
        normal⁻,
        state_prognostic⁻::Vars,
        state_auxiliary⁻::Vars,
        t
    )
    boundary_state!(
        ::NumericalFluxFirstOrder,
        ::BC
        ::BL,
        state_prognostic⁺::Vars,
        state_auxiliary⁺::Vars,
        normal⁻,
        state_prognostic⁻::Vars,
        state_auxiliary⁻::Vars,
        t
    )
    boundary_state!(
        ::NumericalFluxSecondOrder,
        ::BC
        ::BL,
        state_prognostic⁺::Vars,
        state_gradient_flux⁺::Vars,
        state_auxiliary⁺:
        Vars, normal⁻,
        state_prognostic⁻::Vars,
        state_gradient_flux⁻::Vars,
        state_auxiliary⁻::Vars,
        t
    )

Specify the opposite (+ side) face for the boundary condition type `BC` with balance law `BL`.

 - `NumericalFluxGradient` numerical flux (internal method)
 - `NumericalFluxFirstOrder` first-order unknowns
 - `NumericalFluxSecondOrder` second-order unknowns

"""
function boundary_state! end
