using Test, MPI

using ClimateMachine
using ClimateMachine.MPIStateArrays
using ClimateMachine.MPIStateArrays: getstateview
using ClimateMachine.VariableTemplates: @vars, varsindex, varsindices
using StaticArrays

ClimateMachine.init()
const ArrayType = ClimateMachine.array_type()
const mpicomm = MPI.COMM_WORLD

const V = @vars begin
    a::Float32
    b::SVector{3, Float32}
    c::SMatrix{3, 8, Float32}
    d::Float32
    e::@vars begin
        a::Float32
        b::SVector{3, Float32}
        d::Float32
    end
end

const VNT = @vars begin
    a::Float32
    e::Tuple{ntuple(3) do i
        @vars(b::SVector{3, Float32})
    end...}
end

@testset "MPIStateArray varsindex" begin
    # check with invalid vars size
    @test_throws ErrorException MPIStateArray{Float32, V}(
        mpicomm,
        ArrayType,
        4,
        1,
        8,
    )

    Q = MPIStateArray{Float32, V}(mpicomm, ArrayType, 4, 34, 8)
    @test Q.a === view(MPIStateArrays.realview(Q), :, 1:1, :)
    @test Q.b === view(MPIStateArrays.realview(Q), :, 2:4, :)
    @test Q.c === view(MPIStateArrays.realview(Q), :, 5:28, :)
    @test Q.d === view(MPIStateArrays.realview(Q), :, 29:29, :)
    @test Q.e === view(MPIStateArrays.realview(Q), :, 30:34, :)

    @test getstateview(Q, "a") === view(MPIStateArrays.realview(Q), :, 1:1, :)
    @test getstateview(Q, "b") === view(MPIStateArrays.realview(Q), :, 2:4, :)
    @test getstateview(Q, "c") === view(MPIStateArrays.realview(Q), :, 5:28, :)
    @test getstateview(Q, "d") === view(MPIStateArrays.realview(Q), :, 29:29, :)
    @test getstateview(Q, "e") === view(MPIStateArrays.realview(Q), :, 30:34, :)
    @test getstateview(Q, "e.a") ===
          view(MPIStateArrays.realview(Q), :, 30:30, :)
    @test getstateview(Q, "e.b") ===
          view(MPIStateArrays.realview(Q), :, 31:33, :)
    @test getstateview(Q, "e.d") ===
          view(MPIStateArrays.realview(Q), :, 34:34, :)

    @test getstateview(Q, :(a)) === view(MPIStateArrays.realview(Q), :, 1:1, :)
    @test getstateview(Q, :(b)) === view(MPIStateArrays.realview(Q), :, 2:4, :)
    @test getstateview(Q, :(c)) === view(MPIStateArrays.realview(Q), :, 5:28, :)
    @test getstateview(Q, :(d)) ===
          view(MPIStateArrays.realview(Q), :, 29:29, :)
    @test getstateview(Q, :(e)) ===
          view(MPIStateArrays.realview(Q), :, 30:34, :)
    @test getstateview(Q, :(e.a)) ===
          view(MPIStateArrays.realview(Q), :, 30:30, :)
    @test getstateview(Q, :(e.b)) ===
          view(MPIStateArrays.realview(Q), :, 31:33, :)
    @test getstateview(Q, :(e.d)) ===
          view(MPIStateArrays.realview(Q), :, 34:34, :)

    @test_throws ErrorException Q.aa
    @test_throws ErrorException getstateview(Q, "aa")

    P = similar(Q)
    @test P.a === view(MPIStateArrays.realview(P), :, 1:1, :)
    @test P.b === view(MPIStateArrays.realview(P), :, 2:4, :)
    @test P.c === view(MPIStateArrays.realview(P), :, 5:28, :)
    @test P.d === view(MPIStateArrays.realview(P), :, 29:29, :)
    @test P.e === view(MPIStateArrays.realview(P), :, 30:34, :)

    @test getstateview(P, "a") === view(MPIStateArrays.realview(P), :, 1:1, :)
    @test getstateview(P, "b") === view(MPIStateArrays.realview(P), :, 2:4, :)
    @test getstateview(P, "c") === view(MPIStateArrays.realview(P), :, 5:28, :)
    @test getstateview(P, "d") === view(MPIStateArrays.realview(P), :, 29:29, :)
    @test getstateview(P, "e") === view(MPIStateArrays.realview(P), :, 30:34, :)
    @test getstateview(P, "e.a") ===
          view(MPIStateArrays.realview(P), :, 30:30, :)
    @test getstateview(P, "e.b") ===
          view(MPIStateArrays.realview(P), :, 31:33, :)
    @test getstateview(P, "e.d") ===
          view(MPIStateArrays.realview(P), :, 34:34, :)

    @test getstateview(P, :(a)) === view(MPIStateArrays.realview(P), :, 1:1, :)
    @test getstateview(P, :(b)) === view(MPIStateArrays.realview(P), :, 2:4, :)
    @test getstateview(P, :(c)) === view(MPIStateArrays.realview(P), :, 5:28, :)
    @test getstateview(P, :(d)) ===
          view(MPIStateArrays.realview(P), :, 29:29, :)
    @test getstateview(P, :(e)) ===
          view(MPIStateArrays.realview(P), :, 30:34, :)
    @test getstateview(P, :(e.a)) ===
          view(MPIStateArrays.realview(P), :, 30:30, :)
    @test getstateview(P, :(e.b)) ===
          view(MPIStateArrays.realview(P), :, 31:33, :)
    @test getstateview(P, :(e.d)) ===
          view(MPIStateArrays.realview(P), :, 34:34, :)

    A = MPIStateArray{Float32}(mpicomm, ArrayType, 4, 29, 8)
    @test_throws ErrorException A.a
    @test_throws ErrorException getstateview(A, "a")
end

@testset "MPIStateArray show_not_finite_fields" begin
    post_msg = "are not finite (has NaNs or Inf)"
    Q = MPIStateArray{Float32, V}(mpicomm, ArrayType, 4, 34, 8)
    Q .= 1
    Qv = view(MPIStateArrays.realview(Q), :, 1:1, :)
    Qv .= NaN
    msg = "Field(s) (a) " * post_msg
    @test_logs (:warn, msg) show_not_finite_fields(Q)

    Qv = view(MPIStateArrays.realview(Q), :, 2:2, :)
    Qv .= NaN
    msg = "Field(s) (a, and b[1]) " * post_msg
    @test_logs (:warn, msg) show_not_finite_fields(Q)

    Qv = view(MPIStateArrays.realview(Q), :, 31:31, :)
    Qv .= NaN
    msg = "Field(s) (a, b[1], and e.b[1]) " * post_msg
    @test_logs (:warn, msg) show_not_finite_fields(Q)

    Q .= 1
    @test show_not_finite_fields(Q) == nothing
end

@testset "MPIStateArray show_not_finite_fields - ntuple vars" begin
    post_msg = "are not finite (has NaNs or Inf)"
    Q = MPIStateArray{Float32, VNT}(mpicomm, ArrayType, 4, 10, 8)
    Q .= 1
    Qv = view(MPIStateArrays.realview(Q), :, 1:1, :)
    Qv .= NaN
    msg = "Field(s) (a) " * post_msg
    @test_logs (:warn, msg) show_not_finite_fields(Q)

    Qv = view(MPIStateArrays.realview(Q), :, 2:2, :)
    Qv .= NaN
    msg = "Field(s) (a, and e.1.b[1]) " * post_msg
    @test_logs (:warn, msg) show_not_finite_fields(Q)

    Qv = view(MPIStateArrays.realview(Q), :, 6:6, :)
    Qv .= NaN
    msg = "Field(s) (a, e.1.b[1], and e.2.b[2]) " * post_msg
    @test_logs (:warn, msg) show_not_finite_fields(Q)
end
