using MPI, Test

include("../testhelpers.jl")

@testset "Driver" begin
    runmpi(joinpath(@__DIR__, "cr_unit_tests.jl"), ntasks = 1)
    runmpi(joinpath(@__DIR__, "driver_test.jl"), ntasks = 1)

    #TODO: these take too long on Azure CI -- need to go in weekly
    # runmpi(`$(joinpath(@__DIR__, mms3.jl)) --checkpoint-walltime 60 --checkpoint-keep-all`; ntasks=1)
    # runmpi(`$(joinpath(@__DIR__, mms3.jl)) --restart-from-num 3`; ntasks=1)
end
