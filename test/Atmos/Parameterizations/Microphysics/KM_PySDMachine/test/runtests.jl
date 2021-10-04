"""

julia --project="test/Atmos/Parameterizations/Microphysics/KM_PySDMachine" test/Atmos/Parameterizations/Microphysics/KM_PySDMachine/test/runtests.jl will run tests from the main ClimateMachine.jl directory.

"""

using Test, Pkg, Conda


begin
    root_folder_index = findlast("ClimateMachine.jl", pwd())
    tmp_path = pwd()[root_folder_index[1]:end]
    n = length(splitpath(tmp_path)) - 1

    if n == 0
        path = "."
    else
        path = repeat("../", n)
    end

    Pkg.add(url = path)

    Conda.pip_interop(true)
    Conda.pip("install", "PySDM==1.16")
end

@testset "PySDMCall tests" begin
    @testset "PyCall invocation" begin
        include(joinpath("PyCall_invocation.jl"))
    end
    @testset "PySDM presence" begin
        include(joinpath("PySDM_presence.jl"))
    end
    @testset "PySDMCallback invocation" begin
        include(joinpath("PySDMCallback_invocation.jl"))
    end
    @testset "Constant cloud base height" begin
        include(joinpath("KM_constant_cloud_base_height.jl"))
    end
    @testset "PySDM Constants" begin
        include(joinpath("PySDM_constants.jl"))
    end
end
