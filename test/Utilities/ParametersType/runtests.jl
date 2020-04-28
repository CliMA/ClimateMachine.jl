using Test
using CLIMA.ParametersType
@parameter a π "parameter a"
@parameter b π "parameter b"
@parameter c 2a+b "parameter c"
@parameter d 2//3 "parameter d"
function bar()
  sin(a / 3)
end
@parameter e bar() "parameter d"

module Foo
  using CLIMA.ParametersType
  @parameter a1 π "parameter a1"
  @parameter b1 π "parameter b1"
  @parameter c1 2a1+b1 "parameter c1"
  @exportparameter a2 π "parameter a2"
  @exportparameter b2 π "parameter b2"
  @exportparameter c2 2a2+b2 "parameter c2"
end

using Main.Foo

@testset begin
  # Test comparisons
  @test a==a
  @test a==b
  @test a!=c
  @test !(a==c)
  @test !(a!=a)

  @test a<=b
  @test a<=a

  @test !(b<a)
  @test !(a<a)

  @test  a<c
  @test  a<=c
  @test !(a<a)
  @test !(a>=c)
  @test !(a>c)

  # Check some more advanced equalities
  @test ParametersType.getval(a) == π
  @test a == π
  @test ParametersType.getval(a) != Float64(π)
  @test ParametersType.getval(d) == 2//3
  @test d == 2//3
  @test ParametersType.getval(d) != Float64(2//3)
  @test e == bar()

  # check types
  @test typeof(2*a) == Float64
  x = Float32(1)
  @test typeof(a+x) == Float32

  # check modules based parameters
  @test !@isdefined a1
  @test !@isdefined b1
  @test !@isdefined c1

  @test @isdefined a2
  @test @isdefined b2
  @test @isdefined c2

  @test try
    Foo.a1
    Foo.b1
    Foo.c1
    true
  catch
    false
  end
end
