"""
    InitStateBC

Set the value at the boundary to match the `init_state_prognostic!` function. This is
mainly useful for cases where the problem has an explicit solution.

# TODO: This should be fixed later once BCs are figured out (likely want
# different things here?)
"""
struct InitStateBC <: BoundaryCondition end


function boundary_state!(
    ::Union{NumericalFluxFirstOrder,NumericalFluxGradient},
    bc::InitStateBC,
    m::AtmosModel,
    state⁺,
    aux⁺,
    n⁻,
    state⁻,
    aux⁻,
    t,
    args...,
)
    init_state_prognostic!(m, state⁺, aux⁺, aux⁺.coord, t)
end

function boundary_state!(
    ::NumericalFluxSecondOrder,
    bc::InitStateBC,
    m::AtmosModel,
    state⁺,
    diff⁺,
    aux⁺,
    n⁻,
    state⁻,
    diff⁻,
    aux⁻,
    t,
    args...,
)
    init_state_prognostic!(m, state⁺, aux⁺, aux⁺.coord, t)
end
