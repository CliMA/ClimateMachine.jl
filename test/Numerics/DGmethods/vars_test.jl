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
    vars_integrals,
    flux_first_order!,
    flux_second_order!,
    source!,
    wavespeed,
    update_auxiliary_state!,
    boundary_state!,
    init_state_auxiliary!,
    init_state_conservative!,
    init_ode_state,
    LocalGeometry


struct VarsTestModel{dim} <: BalanceLaw end

vars_state_conservative(::VarsTestModel, T) = @vars(x::T, coord::SVector{3, T})
vars_state_auxiliary(m::VarsTestModel, T) =
    @vars(coord::SVector{3, T}, polynomial::T)
vars_state_gradient_flux(m::VarsTestModel, T) = @vars()

flux_first_order!(::VarsTestModel, _...) = nothing
flux_second_order!(::VarsTestModel, _...) = nothing
source!(::VarsTestModel, _...) = nothing
boundary_state!(_, ::VarsTestModel, _...) = nothing
wavespeed(::VarsTestModel, _...) = 1

function init_state_conservative!(
    m::VarsTestModel,
    state::Vars,
    aux::Vars,
    coord,
    t::Real,
)
    @inbounds state.x = coord[1]
    state.coord = coord
end

function init_state_auxiliary!(
    ::VarsTestModel{dim},
    aux::Vars,
    g::LocalGeometry,
) where {dim}
    x, y, z = aux.coord = g.coord
    aux.polynomial = x * y + x * z + y * z
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
        VarsTestModel{dim}(),
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q = init_ode_state(dg, FT(0))
    @test Array(Q.x)[:, 1, :] == Array(Q.coord)[:, 1, :]
    @test Array(dg.state_auxiliary.coord) == Array(Q.coord)
    x = Array(Q.coord)[:, 1, :]
    y = Array(Q.coord)[:, 2, :]
    z = Array(Q.coord)[:, 3, :]
    @test Array(dg.state_auxiliary.polynomial)[:, 1, :] ≈
          x .* y + x .* z + y .* z
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
