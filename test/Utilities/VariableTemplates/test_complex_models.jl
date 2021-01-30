using Test
using StaticArrays
using ClimateMachine.VariableTemplates
using ClimateMachine.VariableTemplates: wrap_val
import ClimateMachine.VariableTemplates
VT = VariableTemplates

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

    @test vuntuple(x -> x, 5) == ntuple(i -> Val(i), Val(5))

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

    # flattened_tup_chain - empty/generic cases
    struct Foo end
    @test flattened_tup_chain(NamedTuple{(), Tuple{}}) == ()
    @test flattened_tup_chain(Foo, RetainArr()) == ((Symbol(),),)
    @test flattened_tup_chain(Foo, FlattenArr()) == ((Symbol(),),)

    # flattened_tup_chain - SHermitianCompact
    Nv, M = 3, 6
    A = SHermitianCompact{Nv, FT, M}(collect(1:(1 + M - 1)))

    ftc = flattened_tup_chain(typeof(A), FlattenArr())
    @test ftc == ntuple(i -> (Symbol(), i), M)

    ftc = flattened_tup_chain(typeof(A), RetainArr())
    @test ftc == ((Symbol(),),)

    # flattened_tup_chain - Retain arrays

    ftc = flattened_tup_chain(st, RetainArr())
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

    # test that getproperty matches varsindex
    ntuple(N) do i
        i_ϕ = varsindex(st, wrap_val.(ftc[i])...)
        ϕ = getproperty(v, wrap_val.(ftc[i]))
        @test all(parent(v)[i_ϕ] .≈ ϕ)
    end

    # test getproperty with tup-chain
    @unroll_map(N) do i
        @test v.scalar_model.x == getproperty(v, (:scalar_model, :x))
        @test v.vector_model.x == getproperty(v, (:vector_model, :x))
        @test v.ntuple_model[i] == getproperty(v, (:ntuple_model, i))
        @test v.ntuple_model[i].scalar_model ==
              getproperty(v, (:ntuple_model, i, :scalar_model))
        @test v.ntuple_model[i].scalar_model.x ==
              getproperty(v, (:ntuple_model, i, :scalar_model, :x))
    end

    # Test converting to flattened NamedTuple
    fnt = flattened_named_tuple(v, RetainArr())
    @test fnt.ntuple_model_1_scalar_model_x == 1.0f0
    @test fnt.ntuple_model_1_vector_model_x == Float32[2.0, 3.0, 4.0]
    @test fnt.ntuple_model_2_scalar_model_x == 5.0f0
    @test fnt.ntuple_model_2_vector_model_x == Float32[6.0, 7.0, 8.0]
    @test fnt.ntuple_model_3_scalar_model_x == 9.0f0
    @test fnt.ntuple_model_3_vector_model_x == Float32[10.0, 11.0, 12.0]
    @test fnt.ntuple_model_4_scalar_model_x == 13.0f0
    @test fnt.ntuple_model_4_vector_model_x == Float32[14.0, 15.0, 16.0]
    @test fnt.ntuple_model_5_scalar_model_x == 17.0f0
    @test fnt.ntuple_model_5_vector_model_x == Float32[18.0, 19.0, 20.0]
    @test fnt.vector_model_x == Float32[21.0, 22.0, 23.0]
    @test fnt.scalar_model_x == 24.0f0

    # flattened_tup_chain - Flatten arrays

    ftc = flattened_tup_chain(st, FlattenArr())
    j = 1
    for i in 1:N
        @test ftc[j] === (:ntuple_model, i, :scalar_model, :x)
        j += 1
        for k in 1:Nv
            @test ftc[j] === (:ntuple_model, i, :vector_model, :x, k)
            j += 1
        end
    end
    for i in 1:Nv
        @test ftc[j] === (:vector_model, :x, i)
        j += 1
    end
    @test ftc[j] === (:scalar_model, :x)

    # test varsindex (flatten arrays)
    ntuple(N) do i
        i_val = Val(i)
        i_sm = varsindex(st, :ntuple_model, i_val, :scalar_model, :x)
        nt_offset = (Nv + 1) - 1

        i_sm_correct = (i + nt_offset * (i - 1)):(i + nt_offset * (i - 1))
        @test i_sm == i_sm_correct

        for j in 1:Nv
            i_vm =
                varsindex(st, :ntuple_model, i_val, :vector_model, :x, Val(j))
            offset = 1
            i_start = i + nt_offset * (i - 1) + offset
            i_vm_correct = i_start + j - 1
            @test i_vm == i_vm_correct:i_vm_correct
        end
    end

    # test that getproperty matches varsindex
    ntuple(N) do i
        i_ϕ = varsindex(st, wrap_val.(ftc[i])...)
        ϕ = getproperty(v, wrap_val.(ftc[i]))
        @test all(parent(v)[i_ϕ] .≈ ϕ)
    end

    # test getproperty with tup-chain
    for k in 1:Nv
        @test v.vector_model.x[k] == getproperty(v, (:vector_model, :x, Val(k)))
    end

    # test getindex with Val
    @test getindex((1, 2), Val(1)) == 1
    @test getindex((1, 2), Val(2)) == 2
    @test getindex(SVector(1, 2), Val(1)) == 1
    @test getindex(SVector(1, 2), Val(2)) == 2

    nt = (; a = ((; x = 1), (; x = 2)))
    fnt = VT.flattened_nt_vals(FlattenArr(), nt)
    vg = Grad{typeof(nt)}(zeros(MMatrix{3, length(fnt), FT}))
    parent(vg)[1, :] .= fnt
    parent(vg)[2, :] .= fnt
    parent(vg)[3, :] .= fnt
    for i in 1:2
        @test getindex(vg.a, Val(i)).x[1] == i
        @test getindex(vg.a, Val(i)).x[2] == i
        @test getindex(vg.a, Val(i)).x[3] == i
    end

    # getpropertyorindex
    @test VT.getpropertyorindex((1, 2), Val(1)) == 1
    @test VT.getpropertyorindex((1, 2), Val(2)) == 2
    @test VT.getpropertyorindex([1, 2], Val(1)) == 1
    @test VT.getpropertyorindex([1, 2], Val(2)) == 2
    @test VT.getpropertyorindex(v, :scalar_model) == v.scalar_model
    for i in 1:N
        @test VT.getpropertyorindex(v.ntuple_model, Val(i)) ==
              v.ntuple_model[Val(i)]
        @test VT.getpropertyorindex(v.ntuple_model, (Val(i),)) ==
              v.ntuple_model[Val(i)]
        @test getindex(v.ntuple_model, (Val(i),)) ==
              VT.getpropertyorindex(v.ntuple_model, (Val(i),))
    end

    # Test converting to flattened NamedTuple
    fnt = flattened_named_tuple(v, FlattenArr())
    @test fnt.ntuple_model_1_scalar_model_x == 1.0f0
    @test fnt.ntuple_model_1_vector_model_x_1 == 2.0
    @test fnt.ntuple_model_1_vector_model_x_2 == 3.0
    @test fnt.ntuple_model_1_vector_model_x_3 == 4.0
    @test fnt.ntuple_model_2_scalar_model_x == 5.0f0
    @test fnt.ntuple_model_2_vector_model_x_1 == 6.0
    @test fnt.ntuple_model_2_vector_model_x_2 == 7.0
    @test fnt.ntuple_model_2_vector_model_x_3 == 8.0
    @test fnt.ntuple_model_3_scalar_model_x == 9.0f0
    @test fnt.ntuple_model_3_vector_model_x_1 == 10.0
    @test fnt.ntuple_model_3_vector_model_x_2 == 11.0
    @test fnt.ntuple_model_3_vector_model_x_3 == 12.0
    @test fnt.ntuple_model_4_scalar_model_x == 13.0f0
    @test fnt.ntuple_model_4_vector_model_x_1 == 14.0
    @test fnt.ntuple_model_4_vector_model_x_2 == 15.0
    @test fnt.ntuple_model_4_vector_model_x_3 == 16.0
    @test fnt.ntuple_model_5_scalar_model_x == 17.0f0
    @test fnt.ntuple_model_5_vector_model_x_1 == 18.0
    @test fnt.ntuple_model_5_vector_model_x_2 == 19.0
    @test fnt.ntuple_model_5_vector_model_x_3 == 20.0
    @test fnt.vector_model_x_1 == 21.0
    @test fnt.vector_model_x_2 == 22.0
    @test fnt.vector_model_x_3 == 23.0
    @test fnt.scalar_model_x == 24.0f0

    struct Foo end
    nt = (;
        nest = (;
            v = SVector(1, 2, 3),
            nt = (;
                shc = SHermitianCompact{3, FT, 6}(collect(1:6)),
                f = FT(1.0),
            ),
            d = SDiagonal(collect(1:3)...),
            tt = (Foo(), Foo()),
            t = Foo(),
        ),
    )
    # Test flattened_nt_vals:

    @test VT.flattened_nt_vals(RetainArr(), NamedTuple()) == ()
    @test VT.flattened_nt_vals(FlattenArr(), NamedTuple()) == ()
    @test VT.flattened_nt_vals(RetainArr(), Tuple(NamedTuple())) == ()
    @test VT.flattened_nt_vals(FlattenArr(), Tuple(NamedTuple())) == ()

    ft = FlattenArr()
    @test VT.flattened_nt_vals(ft, nt.nest.nt.f) == (1.0f0,)
    @test VT.flattened_nt_vals(ft, nt.nest.nt) ==
          (1.0f0, 2.0f0, 3.0f0, 4.0f0, 5.0f0, 6.0f0, 1.0f0)
    @test VT.flattened_nt_vals(ft, nt.nest.d) == (1, 2, 3)
    @test VT.flattened_nt_vals(ft, nt.nest.t) == (Foo(),)
    @test VT.flattened_nt_vals(ft, nt.nest.tt) == (Foo(), Foo())

    ft = RetainArr()
    @test VT.flattened_nt_vals(ft, nt.nest.nt.f) == (1.0f0,)
    @test VT.flattened_nt_vals(ft, nt.nest.nt)[1] ==
          nt.nest.nt.shc.lowertriangle
    @test VT.flattened_nt_vals(ft, nt.nest.nt)[2] == 1.0f0
    @test VT.flattened_nt_vals(ft, nt.nest.d) == (nt.nest.d.diag,)
    @test VT.flattened_nt_vals(ft, nt.nest.t) == (Foo(),)
    @test VT.flattened_nt_vals(ft, nt.nest.tt) == (Foo(), Foo())

    # Test flattened_named_tuple for NamedTuples
    fnt = flattened_named_tuple(nt, FlattenArr())
    @test fnt.nest_v_1 == 1
    @test fnt.nest_v_2 == 2
    @test fnt.nest_v_3 == 3
    @test fnt.nest_nt_shc_1 == 1.0
    @test fnt.nest_nt_shc_2 == 2.0
    @test fnt.nest_nt_shc_3 == 3.0
    @test fnt.nest_nt_shc_4 == 4.0
    @test fnt.nest_nt_shc_5 == 5.0
    @test fnt.nest_nt_shc_6 == 6.0
    @test fnt.nest_nt_f == 1.0
    @test fnt.nest_tt_1 == Foo()
    @test fnt.nest_tt_2 == Foo()
    @test fnt.nest_t == Foo()

    fnt = flattened_named_tuple(nt, RetainArr())
    @test fnt.nest_v == SVector(1, 2, 3)
    @test fnt.nest_nt_shc == nt.nest.nt.shc.lowertriangle
    @test fnt.nest_nt_f == 1.0
    @test fnt.nest_tt_1 == Foo()
    @test fnt.nest_tt_2 == Foo()
    @test fnt.nest_t == Foo()

end
