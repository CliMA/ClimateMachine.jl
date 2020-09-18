using Test
using StaticArrays
using ClimateMachine.VariableTemplates

@testset "Test complex models" begin

    include("complex_models.jl")

    FT = Float32

    # test getproperty
    m = ScalarModel()
    st = state(m, FT)
    vs = varsize(st)
    a_global = collect(1:vs)
    v = Vars{st}(a_global)
    @test v.x == FT(1)

    Nv = 4
    m = VectorModel{Nv}()
    st = state(m, FT)
    vs = varsize(st)
    a_global = collect(1:vs)
    v = Vars{st}(a_global)
    @test v.x == SVector{Nv, FT}(1:Nv)

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
    N = 5
    m = NTupleContainingModel(N, Nv)
    st = state(m, FT)
    vs = varsize(st)
    a_global = collect(1:vs)
    v = Vars{st}(a_global)

    offset = (Nv + 1) * N
    @test v.vector_model.x == SVector{Nv, FT}(collect(1:Nv) .+ offset)
    @test v.scalar_model.x == FT(1 + Nv) + offset

    unval(::Val{i}) where {i} = i
    @unroll_map(N) do i
        @test m.ntuple_model[i] isa NTupleModel
        @test m.ntuple_model[i].scalar_model isa ScalarModel
        @test v.ntuple_model[i].scalar_model.x ==
              FT(unval(i)) + (Nv) * (unval(i) - 1)
        @test v.vector_model.x == SVector{Nv, FT}(1:Nv) .+ offset
        @test v.scalar_model.x == FT(Nv + 1) + offset
    end

    # test flattenednames
    fn = flattenednames(st)
    j = 1
    for i in 1:N
        @test fn[j] === "ntuple_model[$i].scalar_model.x"
        j += 1
        for k in 1:Nv
            @test fn[j] === "ntuple_model[$i].vector_model.x[$k]"
            j += 1
        end
    end
    for k in 1:Nv
        @test fn[j] === "vector_model.x[$k]"
        j += 1
    end
    @test fn[j] === "scalar_model.x"

    # test flattened_tup_chain
    ftc = flattened_tup_chain(st)
    j = 1
    for i in 1:N
        @test ftc[j] === (:ntuple_model, i, :scalar_model, :x)
        j += 1
        @test ftc[j] === (:ntuple_model, i, :vector_model, :x)
        j += 1
    end
    @test ftc[j] === (:vector_model, :x)
    j += 1
    @test ftc[j] === (:scalar_model, :x)

    # test varsindex
    ntuple(N) do i
        i_val = Val(i)
        i_sm = varsindex(st, :ntuple_model, i_val, :scalar_model, :x)
        i_vm = varsindex(st, :ntuple_model, i_val, :vector_model, :x)
        nt_offset = (Nv + 1) - 1

        i_sm_correct = (i + nt_offset * (i - 1)):(i + nt_offset * (i - 1))
        @test i_sm == i_sm_correct

        offset = 1
        i_start = i + nt_offset * (i - 1) + offset
        i_vm_correct = (i_start):(i_start + Nv - 1)
        @test i_vm == i_vm_correct
    end

    convert_to_val(sym) = sym
    convert_to_val(i::Int) = Val(i)
    # test that getproperty matches varsindex
    ntuple(N) do i
        i_ϕ = varsindex(st, convert_to_val.(ftc[i])...)
        ϕ = getproperty(v, ftc[i])
        @test all(parent(v)[i_ϕ] .≈ ϕ)
    end

    # test getproperty with tup-chain
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
