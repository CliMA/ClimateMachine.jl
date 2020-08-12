"""
    InitStateBC

Set the value at the boundary to match the `init_state_prognostic!` function. This is
mainly useful for cases where the problem has an explicit solution.

# TODO: This should be fixed later once BCs are figured out (likely want
# different things here?)
"""
struct InitStateBC <: BoundaryCondition end
function atmos_boundary_state!(
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
    init_state_prognostic!(m, state⁺, aux⁺, aux⁺.coord, t)
end

function atmos_normal_boundary_flux_second_order!(
    nf,
    bc::InitStateBC,
    atmos,
    args...,
)

    normal_boundary_flux_second_order!(
        nf,
        bc,
        atmos,
        args...,
    )

end


function boundary_state!(
    ::NumericalFluxSecondOrder,
    bc::InitStateBC,
    m::AtmosModel,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    t,
    args...,
)
    init_state_prognostic!(m, state⁺, aux⁺, aux⁺.coord, t)
end
