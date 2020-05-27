using Test, StaticArrays
using ClimateMachine.VariableTemplates

@testset "NTuple type" begin

    struct SingleType{FT} end
    struct NTupleType{FT} end
    Base.@kwdef struct Container{FT, N}
        ntuple_type::NTuple{N, NTupleType{FT}} =
            ntuple(i -> NTupleType{FT}(), N)
        single_type::SingleType{FT} = SingleType{FT}()
    end

    vars_state(m::NTupleType, FT) = @vars(a::FT, b::SVector{3, FT})
    vars_state(m::NTuple{N, NTupleType}, FT) where {N} =
        Tuple{ntuple(i -> vars_state(m[i], FT), N)...}
    vars_state(::SingleType, FT) = @vars(a::FT)

    function vars_state(m::Container, FT)
        @vars(
            single_type::vars_state(m.single_type, FT),
            ntuple_type::vars_state(m.ntuple_type, FT)
        )
    end

    FT = Float32
    N = 5
    container = Container{FT, N}()
    st = vars_state(container, FT)
    num_state = varsize(vars_state(container, FT))

    # Vars
    local_state = MArray{Tuple{num_state}, FT}(undef)
    fill!(local_state, 0)
    v = Vars{st}(local_state)
    v.single_type.a = 2
    @test v.single_type.a == 2
    v.ntuple_type[1].a = 2
    @test v.ntuple_type[1].a == 2
    v.ntuple_type[2].b = SVector(FT(1), FT(2), FT(3))
    @test v.ntuple_type[2].b == SVector(FT(1), FT(2), FT(3))
    @test v.ntuple_type[1].b == SVector(FT(0), FT(0), FT(0))
    v.ntuple_type[3].a = 3
    @test v.ntuple_type[3].a == 3
    v.ntuple_type[N].a = 3
    @test v.ntuple_type[N].a == 3

    # Grad
    local_state = MArray{Tuple{3, num_state}, FT}(undef)
    fill!(local_state, 0)
    v = Grad{st}(local_state)
    v.single_type.a = SVector(2, 2, 2)
    @test v.single_type.a == SVector(2, 2, 2)
    v.ntuple_type[1].a = SVector(1, 2, 3)
    @test v.ntuple_type[1].a == SVector(1, 2, 3)

    v.ntuple_type[2].b = SArray{Tuple{3, 3}, FT}([1, 2, 3, 4, 5, 6, 7, 8, 9])
    @test v.ntuple_type[2].b ==
          SArray{Tuple{3, 3}, FT}([1, 2, 3, 4, 5, 6, 7, 8, 9])
    @test v.ntuple_type[3].b ==
          SArray{Tuple{3, 3}, FT}([0, 0, 0, 0, 0, 0, 0, 0, 0])
    @test v.ntuple_type[1].b ==
          SArray{Tuple{3, 3}, FT}([0, 0, 0, 0, 0, 0, 0, 0, 0])

    # flattenednames
    fn = flattenednames(st)
    @test fn == [
        "single_type.a",
        "ntuple_type[1].a",
        "ntuple_type[1].b[1]",
        "ntuple_type[1].b[2]",
        "ntuple_type[1].b[3]",
        "ntuple_type[2].a",
        "ntuple_type[2].b[1]",
        "ntuple_type[2].b[2]",
        "ntuple_type[2].b[3]",
        "ntuple_type[3].a",
        "ntuple_type[3].b[1]",
        "ntuple_type[3].b[2]",
        "ntuple_type[3].b[3]",
        "ntuple_type[4].a",
        "ntuple_type[4].b[1]",
        "ntuple_type[4].b[2]",
        "ntuple_type[4].b[3]",
        "ntuple_type[5].a",
        "ntuple_type[5].b[1]",
        "ntuple_type[5].b[2]",
        "ntuple_type[5].b[3]",
    ]

end
