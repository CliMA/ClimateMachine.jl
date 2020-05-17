using MPI
using Test
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers: AbstractODESolver

mutable struct PseudoSolver <: AbstractODESolver
    t::UInt64
    step::Int
    PseudoSolver() = new(time_ns(), 0)
end
gettime(ps::PseudoSolver) = ps.t

function call_cbs(cbs, init = false)
    foreach(cbs) do cb
        cb(init)
    end
end

function main()
    MPI.Init()
    mpicomm = MPI.COMM_WORLD
    ps = PseudoSolver()

    wtcb_initialized = false
    wtcb_calls = 0
    wtcb = GenericCallbacks.EveryXWallTimeSeconds(1, mpicomm) do (init = false)
        if init
            wtcb_initialized = true
        else
            wtcb_calls = wtcb_calls + 1
        end
    end
    stcb_initialized = false
    stcb_calls = 0
    stcb = GenericCallbacks.EveryXSimulationTime(0.5, ps) do (init = false)
        if init
            stcb_initialized = true
        else
            stcb_calls = stcb_calls + 1
        end
    end
    sscb_initialized = false
    sscb_calls = 0
    sscb = GenericCallbacks.EveryXSimulationSteps(10) do (init = false)
        if init
            sscb_initialized = true
        else
            sscb_calls = sscb_calls + 1
        end
    end
    callbacks = (wtcb, stcb, sscb)

    @testset "Generic Callbacks" begin
        call_cbs(callbacks, true)
        @test wtcb_initialized
        @test stcb_initialized
        @test sscb_initialized
        call_cbs(callbacks)
        @test wtcb_calls == 0
        @test stcb_calls == 0
        @test sscb_calls == 0
        sleep(0.5)
        ps.t = time_ns()
        call_cbs(callbacks)
        @test wtcb_calls == 0
        @test stcb_calls == 1
        @test sscb_calls == 0
        sleep(0.5)
        ps.t = time_ns()
        call_cbs(callbacks)
        @test wtcb_calls == 1
        @test stcb_calls == 2
        @test sscb_calls == 0
        for i in 1:7
            call_cbs(callbacks)
        end
        @test wtcb_calls == 1
        @test stcb_calls == 2
        @test sscb_calls == 1
    end
end

main()
