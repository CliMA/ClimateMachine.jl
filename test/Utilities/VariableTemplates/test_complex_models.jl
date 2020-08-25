
@testset "Test complex models" begin

    include("complex_models.jl")

    FT = Float32

    # ------------------------------- Test getproperty
    m = ScalarModel()
    st = state(m, FT)
    vs = varsize(st)
    a_global = collect(1:vs)
    v = Vars{st}(a_global)
    @test v.x == FT(1)

    Nv = 3
    m = VectorModel{Nv}()
    st = state(m, FT)
    vs = varsize(st)
    a_global = collect(1:vs)
    v = Vars{st}(a_global)
    @test v.x == SVector{Nv, FT}(FT[1, 2, 3])

    N = 3
    M = 6
    m = MatrixModel{N, M}()
    st = state(m, FT)
    vs = varsize(st)
    a_global = collect(1:vs)
    v = Vars{st}(a_global)
    @test v.x == SHermitianCompact{N, FT, M}(collect(1:(1 + M - 1)))

    Nv = 3
    N = 3
    M = 6
    m = CompositModel(Nv, N, M)
    st = state(m, FT)
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
    m = NTupleContainingModel(N, Nv)
    st = state(m, FT)
    vs = varsize(st)
    a_global = collect(1:vs)
    v = Vars{st}(a_global)

    @test v.vector_model.x == SVector{Nv, FT}([4, 5, 6])
    @test v.scalar_model.x == FT(7)

    unval(::Val{i}) where {i} = i
    @unroll_map(N) do i
        @test m.ntuple_model[i] isa NTupleModel
        @test m.ntuple_model[i].scalar_model isa ScalarModel
        @test v.ntuple_model[i].scalar_model.x == FT(unval(i))
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
    @unroll_map(N) do i
        @test v.scalar_model.x == getproperty(v, (:scalar_model, :x))
        @test v.vector_model.x == getproperty(v, (:vector_model, :x))
        @test v.ntuple_model[i] == getproperty(v, (:ntuple_model, unval(i)))
        @test v.ntuple_model[i].scalar_model ==
              getproperty(v, (:ntuple_model, unval(i), :scalar_model))
        @test v.ntuple_model[i].scalar_model.x ==
              getproperty(v, (:ntuple_model, unval(i), :scalar_model, :x))
    end

end
