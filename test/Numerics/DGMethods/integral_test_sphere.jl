using MPI
using StaticArrays
using ClimateMachine
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.MPIStateArrays
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods: nodal_init_state_auxiliary!
using ClimateMachine.DGMethods.NumericalFluxes
using Printf
using LinearAlgebra
using Logging

using ClimateMachine.BalanceLaws:
    BalanceLaw,
    Prognostic,
    Auxiliary,
    GradientFlux,
    UpwardIntegrals,
    DownwardIntegrals

import ClimateMachine.BalanceLaws:
    vars_state,
    integral_load_auxiliary_state!,
    flux_first_order!,
    flux_second_order!,
    source!,
    wavespeed,
    update_auxiliary_state!,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    boundary_state!,
    compute_gradient_argument!,
    init_state_auxiliary!,
    init_state_prognostic!,
    integral_set_auxiliary_state!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!

import ClimateMachine.DGMethods: init_ode_state
using ClimateMachine.Mesh.Geometry: LocalGeometry

struct IntegralTestSphereModel{T} <: BalanceLaw
    Rinner::T
    Router::T
end

function update_auxiliary_state!(
    dg::DGModel,
    m::IntegralTestSphereModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    indefinite_stack_integral!(dg, m, Q, dg.state_auxiliary, t, elems)
    reverse_indefinite_stack_integral!(dg, m, Q, dg.state_auxiliary, t, elems)

    return true
end

vars_state(::IntegralTestSphereModel, ::UpwardIntegrals, T) = @vars(v::T)
vars_state(::IntegralTestSphereModel, ::DownwardIntegrals, T) = @vars(v::T)
vars_state(m::IntegralTestSphereModel, ::Auxiliary, T) = @vars(
    int::vars_state(m, UpwardIntegrals(), T),
    rev_int::vars_state(m, UpwardIntegrals(), T),
    r::T,
    a::T
)

vars_state(::IntegralTestSphereModel, ::Prognostic, T) = @vars()
vars_state(::IntegralTestSphereModel, ::GradientFlux, T) = @vars()

flux_first_order!(::IntegralTestSphereModel, _...) = nothing
flux_second_order!(::IntegralTestSphereModel, _...) = nothing
source!(::IntegralTestSphereModel, _...) = nothing
boundary_state!(_, ::IntegralTestSphereModel, _...) = nothing
init_state_prognostic!(::IntegralTestSphereModel, _...) = nothing
wavespeed(::IntegralTestSphereModel, _...) = 1

function integral_nodal_init_state_auxiliary!(
    m::IntegralTestSphereModel,
    aux::Vars,
    tmp::Vars,
    g::LocalGeometry,
)

    x, y, z = g.coord
    aux.r = hypot(x, y, z)
    θ = atan(y, x)
    ϕ = asin(z / aux.r)
    # Exact integral
    aux.a = 1 + cos(ϕ)^2 * sin(θ)^2 + sin(ϕ)^2
    aux.int.v = exp(-aux.a * aux.r^2) - exp(-aux.a * m.Rinner^2)
    aux.rev_int.v = exp(-aux.a * m.Router^2) - exp(-aux.a * aux.r^2)
end

function init_state_auxiliary!(
    m::IntegralTestSphereModel,
    state_auxiliary::MPIStateArray,
    grid,
)
    nodal_init_state_auxiliary!(
        m,
        integral_nodal_init_state_auxiliary!,
        state_auxiliary,
        grid,
    )
end

@inline function integral_load_auxiliary_state!(
    m::IntegralTestSphereModel,
    integrand::Vars,
    state::Vars,
    aux::Vars,
)
    integrand.v = -2 * aux.r * aux.a * exp(-aux.a * aux.r^2)
end

@inline function integral_set_auxiliary_state!(
    m::IntegralTestSphereModel,
    aux::Vars,
    integral::Vars,
)
    aux.int.v = integral.v
end

@inline function reverse_integral_load_auxiliary_state!(
    m::IntegralTestSphereModel,
    integral::Vars,
    state::Vars,
    aux::Vars,
)
    integral.v = aux.int.v
end

@inline function reverse_integral_set_auxiliary_state!(
    m::IntegralTestSphereModel,
    aux::Vars,
    integral::Vars,
)
    aux.rev_int.v = integral.v
end

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

using Test
function run(mpicomm, topl, ArrayType, N, FT, Rinner, Router)
    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
        meshwarp = Topologies.cubedshellwarp,
    )
    dg = DGModel(
        IntegralTestSphereModel(Rinner, Router),
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q = init_ode_state(dg, FT(0))
    dQdt = similar(Q)

    exact_aux = copy(dg.state_auxiliary)
    dg(dQdt, Q, nothing, 0.0)
    euclidean_distance(exact_aux, dg.state_auxiliary)
end

let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 4

    base_Nhorz = 4
    base_Nvert = 2
    Rinner = 1 // 2
    Router = 1

    polynomialorder = 4

    expected_result = [
        4.662884229467401e-7,
        7.218989778540723e-9,
        1.1258613174916711e-10,
        1.7587739986848968e-12,
    ]
    lvls = integration_testing ? length(expected_result) : 1

    for FT in (Float64,) # Float32)
        err = zeros(FT, lvls)
        for l in 1:lvls
            @info (ArrayType, FT, "sphere", l)
            Nhorz = 2^(l - 1) * base_Nhorz
            Nvert = 2^(l - 1) * base_Nvert
            Rrange = grid1d(FT(Rinner), FT(Router); nelem = Nvert)
            topl = StackedCubedSphereTopology(mpicomm, Nhorz, Rrange)
            err[l] = run(
                mpicomm,
                topl,
                ArrayType,
                polynomialorder,
                FT,
                FT(Rinner),
                FT(Router),
            )
            @test expected_result[l] ≈ err[l] rtol = 1e-3 atol = eps(FT)
        end
        if integration_testing
            @info begin
                msg = ""
                for l in 1:(lvls - 1)
                    rate = log2(err[l]) - log2(err[l + 1])
                    msg *= @sprintf("\n  rate for level %d = %e\n", l, rate)
                end
                msg
            end
        end
    end
end

nothing
