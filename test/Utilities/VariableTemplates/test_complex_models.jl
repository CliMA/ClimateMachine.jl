using Test
using StaticArrays
using ClimateMachine.VariableTemplates

@testset "Complex models" begin

    abstract type OneLayerModel end

    struct EmptyModel <: OneLayerModel end
    vars_state(m::EmptyModel, T) = @vars()

    struct ScalarModel <: OneLayerModel end
    vars_state(m::ScalarModel, T) = @vars(x::T)

    struct VectorModel{N} <: OneLayerModel end
    vars_state(m::VectorModel{N}, T) where {N} = @vars(x::SVector{N, T})

    struct MatrixModel{N, M} <: OneLayerModel end
    vars_state(m::MatrixModel{N, M}, T) where {N, M} =
        @vars(x::SHermitianCompact{N, T, M})

    abstract type TwoLayerModel end

    Base.@kwdef struct CompositModel{Nv, N, M} <: TwoLayerModel
        empty_model = EmptyModel()
        scalar_model = ScalarModel()
        vector_model = VectorModel{Nv}()
        matrix_model = MatrixModel{N, M}()
    end
    function vars_state(m::CompositModel, T)
        @vars begin
            empty_model::vars_state(m.empty_model, T)
            scalar_model::vars_state(m.scalar_model, T)
            vector_model::vars_state(m.vector_model, T)
            matrix_model::vars_state(m.matrix_model, T)
        end
    end

    Base.@kwdef struct NTupleModel <: OneLayerModel
        scalar_model = ScalarModel()
    end

    function vars_state(m::NTupleModel, T)
        @vars begin
            scalar_model::vars_state(m.scalar_model, T)
        end
    end

    vars_state(m::NTuple{N, NTupleModel}, FT) where {N} =
        Tuple{ntuple(i -> vars_state(m[i], FT), N)...}

    Base.@kwdef struct NTupleContainingModel{N, Nv} <: TwoLayerModel
        ntuple_model = ntuple(i -> NTupleModel(), N)
        vector_model = VectorModel{Nv}()
        scalar_model = ScalarModel()
    end

    function vars_state(m::NTupleContainingModel, T)
        @vars begin
            ntuple_model::vars_state(m.ntuple_model, T)
            vector_model::vars_state(m.vector_model, T)
            scalar_model::vars_state(m.scalar_model, T)
        end
    end

    FT = Float32

    # ------------------------------- Test getproperty
    m = ScalarModel()
    st = vars_state(m, FT)
    vs = varsize(st)
    a_global = collect(1:vs)
    v = Vars{st}(a_global)
    @test v.x == FT(1)

    Nv = 3
    m = VectorModel{Nv}()
    st = vars_state(m, FT)
    vs = varsize(st)
    a_global = collect(1:vs)
    v = Vars{st}(a_global)
    @test v.x == SVector{Nv, FT}(FT[1, 2, 3])

    N = 3
    M = 6
    m = MatrixModel{N, M}()
    st = vars_state(m, FT)
    vs = varsize(st)
    a_global = collect(1:vs)
    v = Vars{st}(a_global)
    @test v.x == SHermitianCompact{N, FT, M}(collect(1:(1 + M - 1)))

    Nv = 3
    N = 3
    M = 6
    m = CompositModel{Nv, N, M}()
    st = vars_state(m, FT)
    vs = varsize(st)
    a_global = collect(1:vs)
    v = Vars{st}(a_global)

    scalar_model = v.scalar_model
    @test v.scalar_model.x == FT(1)

    vector_model = v.vector_model
    @test v.vector_model.x == SVector{Nv, FT}([2, 3, 4])

    matrix_model = v.matrix_model
    @test v.matrix_model.x ==
          SHermitianCompact{N, FT, M}(collect(5:(5 + M - 1)))

    Nv = 3
    N = 3
    m = NTupleContainingModel{N, Nv}()
    st = vars_state(m, FT)
    vs = varsize(st)
    a_global = collect(1:vs)
    v = Vars{st}(a_global)

    @test v.vector_model.x == SVector{Nv, FT}([4, 5, 6])
    @test v.scalar_model.x == FT(7)

    for i in 1:N
        @test m.ntuple_model[i] isa NTupleModel
        @test m.ntuple_model[i].scalar_model isa ScalarModel
        @test v.ntuple_model[i].scalar_model.x == FT(i)
        @test v.vector_model.x == SVector{Nv, FT}((N + 1):(N + Nv))
        @test v.scalar_model.x == FT(N + Nv + 1)
    end

    fn = flattenednames(st)
    @test fn[1] === "ntuple_model[1].scalar_model.x"
    @test fn[2] === "ntuple_model[2].scalar_model.x"
    @test fn[3] === "ntuple_model[3].scalar_model.x"
    @test fn[4] === "vector_model.x[1]"
    @test fn[5] === "vector_model.x[2]"
    @test fn[6] === "vector_model.x[3]"
    @test fn[7] === "scalar_model.x"

    ftc = flattened_tup_chain(st)
    @test ftc[1] === (:ntuple_model, 1, :scalar_model, :x)
    @test ftc[2] === (:ntuple_model, 2, :scalar_model, :x)
    @test ftc[3] === (:ntuple_model, 3, :scalar_model, :x)
    @test ftc[4] === (:vector_model, :x)
    @test ftc[5] === (:scalar_model, :x)

    # getproperty with tup-chain
    for i in 1:N
        @test v.scalar_model.x == getproperty(v, (:scalar_model, :x))
        @test v.vector_model.x == getproperty(v, (:vector_model, :x))
        @test v.ntuple_model[i] == getproperty(v, (:ntuple_model, i))
        @test v.ntuple_model[i].scalar_model ==
              getproperty(v, (:ntuple_model, i, :scalar_model))
        @test v.ntuple_model[i].scalar_model.x ==
              getproperty(v, (:ntuple_model, i, :scalar_model, :x))
    end

end
