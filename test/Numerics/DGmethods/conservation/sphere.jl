#=
Here we solve the equation:
```math
 q + dot(∇, uq) = 0
 p - dot(∇, up) = 0
```
on a sphere to test the conservation of the numerics

The boundary conditions are `p = q` when `dot(n, u) > 0` and
`q = p` when `dot(n, u) < 0` (i.e., `p` flows into `q` and vice-sersa).
=#

using MPI
using ClimateMachine
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGmethods
using ClimateMachine.DGmethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.GenericCallbacks
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using Random

using ClimateMachine.VariableTemplates
import ClimateMachine.DGmethods:
    BalanceLaw,
    vars_state_auxiliary,
    vars_state_conservative,
    vars_state_gradient,
    vars_state_gradient_flux,
    flux_first_order!,
    flux_second_order!,
    source!,
    boundary_state!,
    init_state_auxiliary!,
    init_state_conservative!,
    init_ode_state,
    LocalGeometry
import ClimateMachine.DGmethods.NumericalFluxes:
    NumericalFluxFirstOrder,
    numerical_flux_first_order!,
    numerical_boundary_flux_first_order!

struct ConservationTestModel <: BalanceLaw end

vars_state_auxiliary(::ConservationTestModel, T) = @vars(vel::SVector{3, T})
vars_state_conservative(::ConservationTestModel, T) = @vars(q::T, p::T)

vars_state_gradient(::ConservationTestModel, T) = @vars()
vars_state_gradient_flux(::ConservationTestModel, T) = @vars()

function init_state_auxiliary!(
    ::ConservationTestModel,
    aux::Vars,
    g::LocalGeometry,
)
    x, y, z = g.coord
    r = x^2 + y^2 + z^2
    aux.vel = SVector(
        cos(10 * π * x) * sin(10 * π * y) + cos(20 * π * z),
        exp(sin(π * r)),
        sin(π * (x + y + z)),
    )
end

function init_state_conservative!(
    ::ConservationTestModel,
    state::Vars,
    aux::Vars,
    coord,
    t,
)
    state.q = rand()
    state.p = rand()
end

function flux_first_order!(
    ::ConservationTestModel,
    flux::Grad,
    state::Vars,
    auxstate::Vars,
    t::Real,
)
    vel = auxstate.vel
    flux.q = state.q .* vel
    flux.p = -state.p .* vel
end

flux_second_order!(::ConservationTestModel, _...) = nothing

source!(::ConservationTestModel, _...) = nothing

struct ConservationTestModelNumFlux <: NumericalFluxFirstOrder end

boundary_state!(
    ::CentralNumericalFluxSecondOrder,
    ::ConservationTestModel,
    _...,
) = nothing

function numerical_flux_first_order!(
    ::ConservationTestModelNumFlux,
    bl::BalanceLaw,
    fluxᵀn::Vars{S},
    n::SVector,
    state⁻::Vars{S},
    aux⁻::Vars{A},
    state⁺::Vars{S},
    aux⁺::Vars{A},
    t,
) where {S, A}
    un⁻ = dot(n, aux⁻.vel)
    un⁺ = dot(n, aux⁺.vel)
    un = (un⁺ + un⁻) / 2

    if un > 0
        fluxᵀn.q = un * state⁻.q
        fluxᵀn.p = -un * state⁺.p
    else
        fluxᵀn.q = un * state⁺.q
        fluxᵀn.p = -un * state⁻.p
    end
end

function numerical_boundary_flux_first_order!(
    ::ConservationTestModelNumFlux,
    bl::BalanceLaw,
    fluxᵀn::Vars{S},
    n::SVector,
    state⁻::Vars{S},
    aux⁻::Vars{A},
    state⁺::Vars{S},
    aux⁺::Vars{A},
    bctype,
    t,
    state1⁻::Vars{S},
    aux1⁻::Vars{A},
) where {S, A}
    un = dot(n, aux⁻.vel)

    if un > 0
        fluxᵀn.q = un * state⁻.q
        fluxᵀn.p = -un * state⁻.q
    else
        fluxᵀn.q = un * state⁻.p
        fluxᵀn.p = -un * state⁻.p
    end
end

function run(mpicomm, ArrayType, N, Nhorz, Rrange, timeend, FT, dt)

    topl = StackedCubedSphereTopology(mpicomm, Nhorz, Rrange)

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
        meshwarp = Topologies.cubedshellwarp,
    )
    dg = DGModel(
        ConservationTestModel(),
        grid,
        ConservationTestModelNumFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q = init_ode_state(dg, FT(0); init_on_cpu = true)

    lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    sum0 = weightedsum(Q)
    @info @sprintf """Starting
    norm(Q₀) = %.16e
    sum(Q₀) = %.16e""" eng0 sum0

    max_mass_loss = FT(0)
    max_mass_gain = FT(0)
    cbmass = GenericCallbacks.EveryXSimulationSteps(1) do
        cbsum = weightedsum(Q)
        max_mass_loss = max(max_mass_loss, sum0 - cbsum)
        max_mass_gain = max(max_mass_gain, cbsum - sum0)
    end
    solve!(Q, lsrk; timeend = timeend, callbacks = (cbmass,))

    # Print some end of the simulation information
    engf = norm(Q)
    sumf = weightedsum(Q)
    @info @sprintf """Finished
    norm(Q)            = %.16e
    norm(Q) / norm(Q₀) = %.16e
    norm(Q) - norm(Q₀) = %.16e
    max mass loss      = %.16e
    max mass gain      = %.16e
    initial mass       = %.16e
    """ engf engf / eng0 engf - eng0 max_mass_loss max_mass_gain sum0
    max(max_mass_loss, max_mass_gain) / sum0
end

using Test
let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()
    mpicomm = MPI.COMM_WORLD

    polynomialorder = 4

    Nhorz = 4

    tolerance = Dict(Float64 => 1e-15, Float32 => 1e-7)

    @testset "$(@__FILE__)" for FT in (Float64, Float32)
        dt = FT(1e-4)
        timeend = 100 * dt
        Rrange = range(FT(1), stop = FT(2), step = FT(1 // 4))

        Random.seed!(0)
        @info (ArrayType, FT)
        delta_mass = run(
            mpicomm,
            ArrayType,
            polynomialorder,
            Nhorz,
            Rrange,
            timeend,
            FT,
            dt,
        )
        @test abs(delta_mass) < tolerance[FT]
    end
end
