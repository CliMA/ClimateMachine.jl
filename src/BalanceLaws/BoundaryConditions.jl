"""
    boundary_conditions(::BalanceLaw)::Tuple

Return a tuple of `BoundaryCondition` objects: grid boundaries tagged with integer `i` will use the `i`th entry of the tuple.

Default is an empty tuple `()`.
"""
function boundary_conditions(::BalanceLaw)
    ()
end


"""
boundary_state!(
    ::NumericalFluxGradient,
    bctag,
    ::L,
    state_prognostic⁺::Vars,
    state_auxiliary⁺::Vars,
    normal⁻,
    state_prognostic⁻::Vars,
    state_auxiliary⁻::Vars,
    t
)
boundary_state!(
    ::NumericalFluxFirstOrder,
    bctag,
    ::L,
    state_prognostic⁺::Vars,
    state_auxiliary⁺::Vars,
    normal⁻,
    state_prognostic⁻::Vars,
    state_auxiliary⁻::Vars,
    t
)
boundary_state!(
    ::NumericalFluxSecondOrder,
    bctag,
    ::L,
    state_prognostic⁺::Vars,
    state_gradient_flux⁺::Vars,
    state_auxiliary⁺:
    Vars, normal⁻,
    state_prognostic⁻::Vars,
    state_gradient_flux⁻::Vars,
    state_auxiliary⁻::Vars,
    t
)

Apply boundary conditions for

- `NumericalFluxGradient` numerical flux (internal method)
- `NumericalFluxFirstOrder` first-order unknowns
- `NumericalFluxSecondOrder` second-order unknowns

"""
function boundary_state! end

function boundary_state!(nf, bctag::Integer, balance_law::BalanceLaw, args...)
    bcs = boundary_conditions(balance_law)

    _boundary_state!(nf, bctag, bcs, balance_law, args...)
end

@generated function _boundary_state!(
    nf,
    bctag,
    bcs::Tuple,
    balance_law::BalanceLaw,
    args...,
)
    N = fieldcount(bcs)
    return quote
        Base.Cartesian.@nif(
            $(N + 1),
            i -> bctag == i, # conditionexpr
            i -> boundary_state!(
                nf,
                bcs[i],
                balance_law,
                args...,
            ), # expr
            i -> error("Invalid boundary tag")
        ) # elseexpr
        return nothing
    end
end

"""
dumb hack for now
"""
function normal_boundary_flux_second_order end

function normal_boundary_flux_second_order!(nf, bctag::Integer, balance_law::BalanceLaw, args...)
    bcs = boundary_conditions(balance_law)

    _normal_boundary_flux_second_order!(nf, bctag, bcs, balance_law, args...)
end

@generated function _normal_boundary_flux_second_order!(
    nf,
    bctag,
    bcs::Tuple,
    balance_law::BalanceLaw,
    args...,
)
    N = fieldcount(tup)
    return quote
        Base.Cartesian.@nif(
            $(N + 1),
            i -> bctag == i, # conditionexpr
            i -> normal_boundary_flux_second_order!(
                nf,
                bcs[i],
                balance_law,
                args...,
            ), #expr
            i -> error("Invalid boundary tag")
        ) # elseexpr
        return nothing
    end
end