using MPI
using StaticArrays
using ClimateMachine
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.MPIStateArrays
using ClimateMachine.DGmethods
using ClimateMachine.DGmethods.NumericalFluxes
using Printf
using LinearAlgebra
using Logging

import ClimateMachine.DGmethods:
    BalanceLaw,
    vars_state_auxiliary,
    vars_state_conservative,
    vars_state_gradient,
    vars_state_gradient_flux,
    flux_first_order!,
    flux_second_order!,
    source!,
    wavespeed,
    LocalGeometry,
    boundary_state!,
    init_state_auxiliary!,
    init_state_conservative!,
    init_ode_state,
    update_auxiliary_state!,
    vars_integrals,
    vars_reverse_integrals,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!


struct IntegralTestModel{dim} <: BalanceLaw end

vars_reverse_integrals(::IntegralTestModel, T) = @vars(a::T, b::T)
vars_integrals(::IntegralTestModel, T) = @vars(a::T, b::T)
vars_state_auxiliary(m::IntegralTestModel, T) = @vars(
    int::vars_integrals(m, T),
    rev_int::vars_reverse_integrals(m, T),
    coord::SVector{3, T},
    a::T,
    b::T,
    rev_a::T,
    rev_b::T
)

vars_state_conservative(::IntegralTestModel, T) = @vars()
vars_state_gradient_flux(::IntegralTestModel, T) = @vars()

flux_first_order!(::IntegralTestModel, _...) = nothing
flux_second_order!(::IntegralTestModel, _...) = nothing
source!(::IntegralTestModel, _...) = nothing
boundary_state!(_, ::IntegralTestModel, _...) = nothing
init_state_conservative!(::IntegralTestModel, _...) = nothing
wavespeed(::IntegralTestModel, _...) = 1

function init_state_auxiliary!(
    ::IntegralTestModel{dim},
    aux::Vars,
    g::LocalGeometry,
) where {dim}
    x, y, z = aux.coord = g.coord
    if dim == 2
        aux.a = x * y + z * y
        aux.b = 2 * x * y + sin(x) * y^2 / 2 - (z - 1)^2 * y^3 / 3
        y_top = 3
        a_top = x * y_top + z * y_top
        b_top = 2 * x * y_top + sin(x) * y_top^2 / 2 - (z - 1)^2 * y_top^3 / 3
        aux.rev_a = a_top - aux.a
        aux.rev_b = b_top - aux.b
    else
        aux.a = x * z + z^2 / 2
        aux.b = 2 * x * z + sin(x) * y * z - (1 + (z - 1)^3) * y^2 / 3
        zz_top = 3
        a_top = x * zz_top + zz_top^2 / 2
        b_top =
            2 * x * zz_top + sin(x) * y * zz_top -
            (1 + (zz_top - 1)^3) * y^2 / 3
        aux.rev_a = a_top - aux.a
        aux.rev_b = b_top - aux.b
    end
end

function update_auxiliary_state!(
    dg::DGModel,
    m::IntegralTestModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    indefinite_stack_integral!(dg, m, Q, dg.state_auxiliary, t, elems)
    reverse_indefinite_stack_integral!(dg, m, Q, dg.state_auxiliary, t, elems)

    return true
end

@inline function integral_load_auxiliary_state!(
    m::IntegralTestModel,
    integrand::Vars,
    state::Vars,
    aux::Vars,
)
    x, y, z = aux.coord
    integrand.a = x + z
    integrand.b = 2 * x + sin(x) * y - (z - 1)^2 * y^2
end

@inline function integral_set_auxiliary_state!(
    m::IntegralTestModel,
    aux::Vars,
    integral::Vars,
)
    aux.int.a = integral.a
    aux.int.b = integral.b
end

@inline function reverse_integral_load_auxiliary_state!(
    m::IntegralTestModel,
    integral::Vars,
    state::Vars,
    aux::Vars,
)
    integral.a = aux.int.a
    integral.b = aux.int.b
end

@inline function reverse_integral_set_auxiliary_state!(
    m::IntegralTestModel,
    aux::Vars,
    integral::Vars,
)
    aux.rev_int.a = integral.a
    aux.rev_int.b = integral.b
end

using Test
function run(mpicomm, dim, Ne, N, FT, ArrayType)

    brickrange = ntuple(j -> range(FT(0); length = Ne[j] + 1, stop = 3), dim)
    topl = StackedBrickTopology(
        mpicomm,
        brickrange,
        periodicity = ntuple(j -> true, dim),
    )

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )
    dg = DGModel(
        IntegralTestModel{dim}(),
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q = init_ode_state(dg, FT(0))
    dQdt = similar(Q)

    dg(dQdt, Q, nothing, 0.0)

    # Wrapping in Array ensure both GPU and CPU code use same approx
    @test Array(dg.state_auxiliary.data[:, 1, :]) ≈
          Array(dg.state_auxiliary.data[:, 8, :])
    @test Array(dg.state_auxiliary.data[:, 2, :]) ≈
          Array(dg.state_auxiliary.data[:, 9, :])
    @test Array(dg.state_auxiliary.data[:, 3, :]) ≈
          Array(dg.state_auxiliary.data[:, 10, :])
    @test Array(dg.state_auxiliary.data[:, 4, :]) ≈
          Array(dg.state_auxiliary.data[:, 11, :])
end

let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    numelem = (5, 5, 5)
    lvls = 1

    polynomialorder = 4

    for FT in (Float64,) #Float32)
        for dim in 2:3
            err = zeros(FT, lvls)
            for l in 1:lvls
                @info (ArrayType, FT, dim)
                run(
                    mpicomm,
                    dim,
                    ntuple(j -> 2^(l - 1) * numelem[j], dim),
                    polynomialorder,
                    FT,
                    ArrayType,
                )
            end
        end
    end
end

nothing
