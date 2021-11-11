#=

julia --project="test/Atmos/Parameterizations/Microphysics/KM_PySDMachine" test/Atmos/Parameterizations/Microphysics/KM_PySDMachine/test/runtests.jl will run tests from the main ClimateMachine.jl directory.

=#

using Pkg, Conda
using Test

begin
    if "BUILDKITE" in keys(ENV) && ENV["BUILDKITE"] == "true"
        ENV["PYTHON"] = string(Conda.PYTHONDIR, "/python")
        Pkg.build("PyCall")
    end

    Conda.pip_interop(true)
    Conda.pip("install", "PySDM==1.19")
end

include(joinpath("..", "..", "KinematicModel.jl"))
include(joinpath("PySDMCall", "PySDMCall.jl"))
include(joinpath("PySDMCall", "PySDMCallback.jl"))
include(joinpath("utils", "KM_CliMA_no_saturation_adjustment.jl"))
include(joinpath("utils", "KM_PySDM.jl"))

@testset "PySDMCall tests" begin
    @testset "PyCall invocation" begin
        include(joinpath("PyCall_invocation.jl"))
    end
    @testset "PySDM presence" begin
        include(joinpath("PySDM_presence.jl"))
    end
    @testset "Constant cloud base height" begin
        include(joinpath("KM_constant_cloud_base_height.jl"))
    end
    @testset "PySDM Constants" begin
        include(joinpath("PySDM_constants.jl"))
    end
    @testset "PySDMCallback invocation" begin
        include(joinpath("PySDMCallback_invocation.jl"))
    end
end
