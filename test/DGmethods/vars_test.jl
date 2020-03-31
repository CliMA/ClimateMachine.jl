using MPI
using StaticArrays
using CLIMA
using CLIMA.VariableTemplates
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.MPIStateArrays
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using Printf
using LinearAlgebra
using Logging
using GPUifyLoops

import CLIMA.DGmethods:
    BalanceLaw,
    vars_aux,
    vars_state,
    vars_gradient,
    vars_diffusive,
    vars_integrals,
    integrate_aux!,
    flux_nondiffusive!,
    flux_diffusive!,
    source!,
    wavespeed,
    update_aux!,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    boundary_state!,
    init_aux!,
    init_state!,
    init_ode_state,
    LocalGeometry


struct VarsTestModel{dim} <: BalanceLaw end

vars_state(::VarsTestModel, T) = @vars(x::T, coord::SVector{3, T})
vars_aux(m::VarsTestModel, T) = @vars(coord::SVector{3, T}, polynomial::T)
vars_diffusive(m::VarsTestModel, T) = @vars()

flux_nondiffusive!(::VarsTestModel, _...) = nothing
flux_diffusive!(::VarsTestModel, _...) = nothing
source!(::VarsTestModel, _...) = nothing
boundary_state!(_, ::VarsTestModel, _...) = nothing
wavespeed(::VarsTestModel, _...) = 1

function init_state!(m::VarsTestModel, state::Vars, aux::Vars, coord, t::Real)
    @inbounds state.x = coord[1]
    state.coord = coord
end

function init_aux!(
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
        Rusanov(),
        CentralNumericalFluxDiffusive(),
        CentralNumericalFluxGradient(),
    )

    Q = init_ode_state(dg, FT(0))
    @test Array(Q.x)[:, 1, :] == Array(Q.coord)[:, 1, :]
    @test Array(dg.auxstate.coord) == Array(Q.coord)
    x = Array(Q.coord)[:, 1, :]
    y = Array(Q.coord)[:, 2, :]
    z = Array(Q.coord)[:, 3, :]
    @test Array(dg.auxstate.polynomial)[:, 1, :] ≈ x .* y + x .* z + y .* z
end

let
    CLIMA.init()
    ArrayType = CLIMA.array_type()

    mpicomm = MPI.COMM_WORLD
    ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
    loglevel = ll == "DEBUG" ? Logging.Debug :
        ll == "WARN" ? Logging.Warn :
        ll == "ERROR" ? Logging.Error : Logging.Info
    logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
    global_logger(ConsoleLogger(logger_stream, loglevel))

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
