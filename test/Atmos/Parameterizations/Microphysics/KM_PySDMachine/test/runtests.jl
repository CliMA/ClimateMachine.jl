"""

julia --project="test/Atmos/Parameterizations/Microphysics/KM_PySDMachine" test/Atmos/Parameterizations/Microphysics/KM_PySDMachine/test/runtests.jl will run tests from the main ClimateMachine.jl directory.

"""

include("./utils/ci.jl")

using Pkg

begin
    path = find_path_to_climatemachine_project()

    Pkg.add(url = path)
end

using Test, Conda

begin
    if is_buildkite_pipeline()
        ENV["PYTHON"] = string(Conda.PYTHONDIR, "/python")
        Pkg.build("PyCall")
    end

    Conda.pip_interop(true)
    Conda.pip("install", "PySDM==1.19")
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
