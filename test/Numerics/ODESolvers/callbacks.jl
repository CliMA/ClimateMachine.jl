using MPI
using Test
using ClimateMachine.ODESolvers: AbstractODESolver
using ClimateMachine.GenericCallbacks

mutable struct PseudoSolver <: AbstractODESolver
    t::Float64
    step::Int
    PseudoSolver() = new(42.0, 0)
end
gettime(ps::PseudoSolver) = ps.t


mutable struct MyCallback
    initialized::Bool
    calls::Int
end
MyCallback() = MyCallback(false, 0)

GenericCallbacks.init!(cb::MyCallback, _...) = cb.initialized = true
GenericCallbacks.call!(cb::MyCallback, _...) = (cb.calls += 1; nothing)

MPI.Init()
mpicomm = MPI.COMM_WORLD
ps = PseudoSolver()

wtcb = GenericCallbacks.EveryXWallTimeSeconds(MyCallback(), 2, mpicomm)
stcb = GenericCallbacks.EveryXSimulationTime(MyCallback(), 0.5)
sscb = GenericCallbacks.EveryXSimulationSteps(MyCallback(), 10)

fn_calls = 0
fncb = () -> (global fn_calls += 1)

wtfn_calls = 0
wtfncb = GenericCallbacks.EveryXWallTimeSeconds(
    AtStart(() -> global wtfn_calls += 1),
    2,
    mpicomm,
)

stfn_calls = 0
stfncb = GenericCallbacks.EveryXSimulationTime(
    AtStart(() -> global stfn_calls += 1),
    0.5,
)

ssfn_calls = 0
ssfncb = GenericCallbacks.EveryXSimulationSteps(
    AtStart(() -> global ssfn_calls += 1),
    5,
)

callbacks = ((wtcb, stcb, sscb), fncb, (wtfncb, stfncb, ssfncb))

@testset "GenericCallbacks" begin
    GenericCallbacks.init!(callbacks, ps, nothing, nothing, ps.t)

    @test wtcb.callback.initialized
    @test stcb.callback.initialized
    @test sscb.callback.initialized

    @test wtcb.callback.calls == 0
    @test stcb.callback.calls == 0
    @test sscb.callback.calls == 0
    @test fn_calls == 0
    @test wtfn_calls == 1
    @test stfn_calls == 1
    @test ssfn_calls == 1

    @test GenericCallbacks.call!(callbacks, ps, nothing, nothing, ps.t) in
          (0, nothing)

    @test wtcb.callback.calls == 0
    @test stcb.callback.calls == 0
    @test sscb.callback.calls == 0
    @test fn_calls == 1
    @test wtfn_calls == 1
    @test stfn_calls == 1
    @test ssfn_calls == 1

    ps.t += 0.5
    @test GenericCallbacks.call!(callbacks, ps, nothing, nothing, ps.t) in
          (0, nothing)

    @test wtcb.callback.calls == 0
    @test stcb.callback.calls == 1
    @test sscb.callback.calls == 0
    @test fn_calls == 2
    @test wtfn_calls == 1
    @test stfn_calls == 2
    @test ssfn_calls == 1

    sleep(max((2.1 + wtcb.lastcbtime_ns / 1e9 - time_ns() / 1e9), 0.0))
    ps.t += 0.5
    @test GenericCallbacks.call!(callbacks, ps, nothing, nothing, ps.t) in
          (0, nothing)

    @test wtcb.callback.calls == 1
    @test stcb.callback.calls == 2
    @test sscb.callback.calls == 0
    @test fn_calls == 3
    @test wtfn_calls == 2
    @test stfn_calls == 3
    @test ssfn_calls == 1

    for i in 1:7
        @test GenericCallbacks.call!(callbacks, ps, nothing, nothing, ps.t) in
              (0, nothing)
    end

    @test wtcb.callback.calls == 1
    @test stcb.callback.calls == 2
    @test sscb.callback.calls == 1
    @test fn_calls == 10
    @test wtfn_calls == 2
    @test stfn_calls == 3
    @test ssfn_calls == 3
end
