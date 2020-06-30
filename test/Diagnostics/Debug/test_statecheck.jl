using Test

# # State debug statistics
#
# Set up a basic environment
using MPI
using StaticArrays
using Random
using ClimateMachine
using ClimateMachine.VariableTemplates
using ClimateMachine.MPIStateArrays
using ClimateMachine.StateCheck
using ClimateMachine.GenericCallbacks

@testset "$(@__FILE__)" begin

    ClimateMachine.init()
    FT = Float64

    # Define some dummy vector and tensor abstract variables with associated types
    # and dimensions
    F1 = @vars begin
        ν∇u::SMatrix{3, 2, FT, 6}
        κ∇θ::SVector{3, FT}
    end
    F2 = @vars begin
        u::SVector{2, FT}
        θ::SVector{1, FT}
    end

    # Create ```MPIStateArray``` variables with arrays to hold elements of the 
    # vectors and tensors
    Q1 = MPIStateArray{Float32, F1}(
        MPI.COMM_WORLD,
        ClimateMachine.array_type(),
        4,
        9,
        8,
    )
    Q2 = MPIStateArray{Float64, F2}(
        MPI.COMM_WORLD,
        ClimateMachine.array_type(),
        4,
        6,
        8,
    )

    # ### Create a call-back
    cb = ClimateMachine.StateCheck.sccreate(
        [(Q1, "My gradients"), (Q2, "My fields")],
        1;
        prec = 15,
    )

    # ### Invoke the call-back
    #     Compare on local "realdata", fill via broadcast to keep GPU happy.
    Q1.data = rand(MersenneTwister(0), Float32, size(Q1.data))
    Q2.data = rand(MersenneTwister(0), Float64, size(Q2.data))
    Q1.realdata .= Q1.data
    Q2.realdata .= Q2.data
    GenericCallbacks.init!(cb, nothing, nothing, nothing, nothing)
    GenericCallbacks.call!(cb, nothing, nothing, nothing, nothing)

    # ### Check with reference values
    include("test_statecheck_refvals.jl")

    # This should return true (MPI rank != 0 always returns true)
    test_stat_01 = ClimateMachine.StateCheck.scdocheck(cb, (varr, parr))

    # This should return false (MPI rank != 0 always returns true)
    varr[1][3] = varr[1][3] * 10.0
    test_stat_02 = ClimateMachine.StateCheck.scdocheck(cb, (varr, parr))
    if MPI.Comm_rank(MPI.COMM_WORLD) != 0
        test_stat_02 = false
    end

    test_stat = test_stat_01 && !test_stat_02

    @test test_stat

end
