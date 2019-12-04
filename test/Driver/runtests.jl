using MPI, Test

include("../testhelpers.jl")

@testset "Driver" begin
    tests = [
        (1, "cr_unit_tests.jl"),
        (1, "driver_test.jl"),
        #TODO: these take too long on Azure CI -- need to go in weekly
        #(1, "mms3.jl --checkpoint-walltime 60 --checkpoint-keep-all"),
        #(1, "mms3.jl --restart-from-num 3"),
    ]

    runmpi(tests, @__FILE__)
end
