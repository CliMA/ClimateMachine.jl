using ..Mesh.Grids: _x1, _x2, _x3
"""
    InitStateBC

Set the value at the boundary to match the `init_state_prognostic!` function. This is
mainly useful for cases where the problem has an explicit solution.

# TODO: This should be fixed later once BCs are figured out (likely want
# different things here?)
"""
struct InitStateBC <: AbstractAtmosBC end
function boundary_state!(
    ::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    bc::InitStateBC,
    m::AtmosModel,
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)
    # Put cood in a NamedTuple to mimmic LocalGeometry
    init_state_prognostic!(m, state⁺, aux⁺, (coord = aux⁺.coord,), t)
end

function boundary_state!(
    ::NumericalFluxSecondOrder,
    bc::InitStateBC,
    m::AtmosModel,
    state⁺::Vars,
    diff⁺::Vars,
    hyperdiff⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    diff⁻::Vars,
    hyperdiff⁻::Vars,
    aux⁻::Vars,
    t,
    args...,
)
    # Put coord in a NamedTuple to mimmic LocalGeometry
    init_state_prognostic!(m, state⁺, aux⁺, (coord = aux⁺.coord,), t)
end

function boundary_state!(
    nf::CentralNumericalFluxHigherOrder,
    bc::InitStateBC,
    m::AtmosModel,
    x...,
)
    nothing
end
