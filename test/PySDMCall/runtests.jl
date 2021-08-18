"""

julia --project test/PySDMCall/runtests.jl will run the experiment from the main ClimateMachine.jl directory.

"""

using Test

@testset "PySDMCall tests" begin
    include(joinpath("PyCall_invocation.jl"))
    include(joinpath("PySDM_presence.jl"))
    include(joinpath("PySDMCallback_invocation.jl")) 
end