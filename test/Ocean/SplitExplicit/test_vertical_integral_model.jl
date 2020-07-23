#!/usr/bin/env julia --project
using Test
using ClimateMachine
ClimateMachine.init()
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Grids: polynomialorder
using ClimateMachine.Ocean
using ClimateMachine.Ocean.HydrostaticBoussinesq
using ClimateMachine.Ocean.ShallowWater
using ClimateMachine.Ocean.SplitExplicit: VerticalIntegralModel

using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.BalanceLaws
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.VTK

using MPI
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates

import ClimateMachine.Ocean.ShallowWater: shallow_init_state!, shallow_init_aux!

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

struct GyreInABox{T} <: ShallowWaterProblem
    τₒ::T
    fₒ::T # value includes τₒ, g, and ρ
    β::T
    Lˣ::T
    Lʸ::T
    H::T
end

function shallow_init_state!(
    m::ShallowWaterModel,
    p::GyreInABox,
    Q,
    A,
    coords,
    t,
)
    @inbounds x = coords[1]

    Lˣ = p.Lˣ
    H = p.H

    kˣ = 2π / Lˣ
    νʰ = m.turbulence.ν

    M = @SMatrix [-νʰ * kˣ^2 grav(m.param_set) * H * kˣ; -kˣ 0]
    A = exp(M * t) * @SVector [1, 1]

    U = A[1] * sin(kˣ * x)

    Q.U = @SVector [U, -0]
    Q.η = A[2] * cos(kˣ * x)

    return nothing
end

function shallow_init_aux!(m::ShallowWaterModel, p::GyreInABox, aux, geom)
    @inbounds y = geom.coord[2]

    Lʸ = p.Lʸ
    τₒ = p.τₒ
    fₒ = p.fₒ
    β = p.β

    aux.τ = @SVector [-τₒ * cos(π * y / Lʸ), 0]
    aux.f = fₒ + β * (y - Lʸ / 2)

    return nothing
end

function test_vertical_integral_model(time; refDat = ())
    mpicomm = MPI.COMM_WORLD
    ArrayType = ClimateMachine.array_type()


    brickrange_2D = (xrange, yrange)
    topl_2D = BrickTopology(
        mpicomm,
        brickrange_2D,
        periodicity = (true, true),
        boundary = ((0, 0), (0, 0)),
    )
    grid_2D = DiscontinuousSpectralElementGrid(
        topl_2D,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    brickrange_3D = (xrange, yrange, zrange)
    topl_3D = StackedBrickTopology(
        mpicomm,
        brickrange_3D;
        periodicity = (true, true, false),
        boundary = ((0, 0), (0, 0), (1, 2)),
    )
    grid_3D = DiscontinuousSpectralElementGrid(
        topl_3D,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    prob_3D = SimpleBox{FT}(Lˣ, Lʸ, H)
    prob_2D = GyreInABox{FT}(-0, -0, -0, Lˣ, Lʸ, H)

    model_3D = HydrostaticBoussinesqModel{FT}(
        param_set,
        prob_3D;
        cʰ = FT(1),
        αᵀ = FT(0),
        κʰ = FT(0),
        κᶻ = FT(0),
        fₒ = FT(0),
        β = FT(0),
    )

    model_2D = ShallowWaterModel(
        param_set,
        prob_2D,
        Uncoupled(),
        ShallowWater.ConstantViscosity{FT}(model_3D.νʰ),
        nothing,
        FT(1),
    )

    integral_bl = VerticalIntegralModel(model_3D)

    integral_model = DGModel(
        integral_bl,
        grid_3D,
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    dg_3D = DGModel(
        model_3D,
        grid_3D,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    dg_2D = DGModel(
        model_2D,
        grid_2D,
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q_3D = init_ode_state(dg_3D, FT(time); init_on_cpu = true)
    Q_2D = init_ode_state(dg_2D, FT(time); init_on_cpu = true)
    Q_int = integral_model.state_auxiliary

    state_check = ClimateMachine.StateCheck.sccreate(
        [
            (Q_int, "∫u")
            (Q_2D, "U")
        ],
        1;
        prec = 12,
    )

    update_auxiliary_state!(integral_model, integral_bl, Q_3D, time)
    GenericCallbacks.call!(state_check, nothing, nothing, nothing, nothing)

    ClimateMachine.StateCheck.scprintref(state_check)
    if length(refDat) > 0
        @test ClimateMachine.StateCheck.scdocheck(state_check, refDat)
    end

    return nothing
end

#################
# RUN THE TESTS #
#################
FT = Float64

const N = 4
const Nˣ = 5
const Nʸ = 5
const Nᶻ = 8
const Lˣ = 1e6  # m
const Lʸ = 1e6  # m
const H = 400  # m

xrange = range(FT(0); length = Nˣ + 1, stop = Lˣ)
yrange = range(FT(0); length = Nʸ + 1, stop = Lʸ)
zrange = range(FT(-H); length = Nᶻ + 1, stop = 0)

#const cʰ = sqrt(grav * H)
const cʰ = 1  # typical of ocean internal-wave speed
const cᶻ = 0

@testset "$(@__FILE__)" begin
    include("../refvals/test_vertical_integral_model_refvals.jl")

    times =
        [0, 86400, 30 * 86400, 365 * 86400, 10 * 365 * 86400, 100 * 365 * 86400]
    for (index, time) in enumerate(times)
        @testset "$(time)" begin
            test_vertical_integral_model(time, refDat = refVals[index])
        end
    end
end
