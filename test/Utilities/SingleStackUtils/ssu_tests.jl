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
using ClimateMachine.SingleStackUtils
using ClimateMachine.DGMethods: LocalGeometry

using ClimateMachine.BalanceLaws:
    BalanceLaw, Auxiliary, Prognostic, Gradient, GradientFlux
import ClimateMachine.BalanceLaws:
    vars_state, nodal_init_state_auxiliary!, init_state_prognostic!

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

function nodal_init_state_auxiliary!(
    m::EmptyBalLaw,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    aux.x = geom.coord[1]
    aux.y = geom.coord[2]
    aux.z = geom.coord[3]
end

function init_state_prognostic!(
    m::EmptyBalLaw,
    state::Vars,
    aux::Vars,
    coords,
    t::Real,
)
    z = aux.z
    x = aux.x
    y = aux.y
    state.ρ = (1 - 4 * (z - m.zmax / 2)^2) * (2 - x - y)
end

function test_hmean(
    grid::DiscontinuousSpectralElementGrid{T, dim, N},
    Q::MPIStateArray,
    vars,
) where {T, dim, N}
    state_vars_avg = get_horizontal_mean(grid, Q, vars)
    target = target_meanprof(grid)
    @test state_vars_avg["ρ"] ≈ target
end

function test_hvar(
    grid::DiscontinuousSpectralElementGrid{T, dim, N},
    Q::MPIStateArray,
    vars,
) where {T, dim, N}
    state_vars_var = get_horizontal_variance(grid, Q, vars)
    target = target_varprof(grid)
    @test state_vars_var["ρ"] ≈ target
end

function target_meanprof(
    grid::DiscontinuousSpectralElementGrid{T, dim, N},
) where {T, dim, N}
    Nq = N + 1
    Ntot = Nq * grid.topology.stacksize
    z = Array(get_z(grid))
    target =
        SVector{Ntot, T}([1.0 - 4.0 * (z_i - z[Ntot] / 2.0)^2 for z_i in z])
    return target
end

function target_varprof(
    grid::DiscontinuousSpectralElementGrid{T, dim, N},
) where {T, dim, N}
    Nq = N + 1
    nvertelem = grid.topology.stacksize
    Ntot = Nq * nvertelem
    z = Array(get_z(grid))
    x = z[1:Nq] * nvertelem
    scaled_var = 0.0
    for i in 1:Nq
        for j in 1:Nq
            scaled_var = scaled_var + (2 - x[i] - x[j]) * (2 - x[i] - x[j])
        end
    end
    target = SVector{Ntot, Float64}([
        (1.0 - 4.0 * (z_i - z[Ntot] / 2.0)^2) *
        (1.0 - 4.0 * (z_i - z[Ntot] / 2.0)^2) *
        (scaled_var / Nq / Nq - 1) for z_i in z
    ])
    return target
end

function main()
    FT = Float64
    ClimateMachine.init()

    m = EmptyBalLaw(param_set, FT(1))

    # Prescribe polynomial order of basis functions in finite elements
    N_poly = 5
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
        "SingleStackUtilsTest",
        N_poly,
        nelem_vert,
        zmax,
        param_set,
        m,
    )
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )

    # tests
    test_hmean(
        driver_config.grid,
        solver_config.Q,
        vars_state(m, Prognostic(), FT),
    )
    test_hvar(
        driver_config.grid,
        solver_config.Q,
        vars_state(m, Prognostic(), FT),
    )

    r1, z1 = reduce_nodal_stack(
        max,
        solver_config.dg.grid,
        solver_config.Q,
        vars_state(m, Prognostic(), FT),
        "ρ",
        i = 6,
        j = 6,
    )
    @test r1 ≈ 8.880558532968455e-16 && z1 == 10
    r2, z2 = reduce_nodal_stack(
        +,
        solver_config.dg.grid,
        solver_config.Q,
        vars_state(m, Prognostic(), FT),
        "ρ",
        i = 3,
        j = 3,
    )
    @test r2 ≈ 102.73283921735293 && z2 == 20
    ns = reduce_element_stack(
        +,
        solver_config.dg.grid,
        solver_config.Q,
        vars_state(m, Prognostic(), FT),
        "ρ",
    )
    (r3, z3) = let
        f(a, b) = (a[1] + b[1], b[2])
        reduce(f, ns)
    end
    @test r3 ≈ FT(2877.6) && z3 == 20
end
main()
