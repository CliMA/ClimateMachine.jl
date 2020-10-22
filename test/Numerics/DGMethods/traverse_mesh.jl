using OrderedCollections
using StaticArrays
using Test

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.Grids
using ClimateMachine.MPIStateArrays
using ClimateMachine.VariableTemplates
using ClimateMachine.DGMethods: LocalGeometry

using ClimateMachine.BalanceLaws
using ClimateMachine.BalanceLaws:
    BalanceLaw, Auxiliary, Prognostic, Gradient, GradientFlux
import ClimateMachine.BalanceLaws: vars_state, init_state_prognostic!

struct EmptyBalLaw{FT, PS} <: BalanceLaw
    "Parameters"
    param_set::PS
    "Domain height"
    zmax::FT
end
EmptyBalLaw(param_set, zmax) =
    EmptyBalLaw{typeof(zmax), typeof(param_set)}(param_set, zmax)

vars_state(::EmptyBalLaw, ::Auxiliary, FT) = @vars(x::FT, y::FT, z::FT)
vars_state(::EmptyBalLaw, ::Prognostic, FT) = @vars(ρ::FT)
vars_state(::EmptyBalLaw, ::Gradient, FT) = @vars()
vars_state(::EmptyBalLaw, ::GradientFlux, FT) = @vars()

function init_state_prognostic!(
    m::EmptyBalLaw,
    state::Vars,
    aux::Vars,
    localgeo,
    t::Real,
)
    state.ρ = 0
end

function traverse_mesh_kernel!(bl, state, aux)
    aux.x = 1
    aux.y = 1
    aux.z = 1
    state.ρ = 1
    @show state.ρ
end

function main()
    FT = Float64
    ClimateMachine.init()

    m = EmptyBalLaw(param_set, FT(1))

    # Prescribe polynomial order of basis functions in finite elements
    N_poly = 1
    # Specify the number of vertical elements
    nelem_vert = 20
    # Specify the domain height
    zmax = m.zmax
    # Initial and final times
    t0 = 0.0
    timeend = 1.0
    dt = 0.1
    # Establish a `ClimateMachine` single stack configuration
    driver_config = ClimateMachine.SingleStackConfiguration(
        "TraverseMeshTest",
        N_poly,
        nelem_vert,
        zmax,
        param_set,
        m,
    )

    dg = DGModel(
        m,
        driver_config.grid,
        driver_config.numerical_flux_first_order,
        driver_config.numerical_flux_second_order,
        driver_config.numerical_flux_gradient)

    Q = init_ode_state(dg, FT(0))
    Q .= 0
    dg.state_auxiliary .= 0
    states = (
        (Prognostic(), Q.data),
        (Auxiliary(), dg.state_auxiliary.data),
        )
    traverse_mesh(traverse_mesh_kernel!, Pointwise(), driver_config.grid, m, states...)
    vs = vars_state(m, Prognostic(), FT)
    i_ρ = varsindex(vs, :ρ)
    @show [Q[:,i_ρ,:]...]
    @show [dg.state_auxiliary...]
    @show BalanceLaws.number_states(m, Auxiliary())
    @show BalanceLaws.number_states(m, Prognostic())
    @test all(Q[:,i_ρ,:] .≈ 1)

end
main()
