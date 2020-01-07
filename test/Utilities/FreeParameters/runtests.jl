using Test, CLIMA.FreeParameters


@testset "FreeParameters" begin

  mutable struct Bar{FT}
    c::FreeParameter{FT}
    d::FT
  end

  mutable struct Foo{FT}
    a::FreeParameter{FT}
    b::Bar{FT}
  end

  FT = Float64
  b = Bar{FT}(FreeParameter{FT}(5.0, 0.0, 10.0),
              10.0)
  f = Foo(FreeParameter{FT}(2.0, 0.0, 10.0),
          b)

  fp = extract_free_parameters(f)
  fp[1].val = 3.0
  fp[2].val = 30.0

  # f is now updated
  # inject_free_parameters!(f, fp)
  @show f


end


end