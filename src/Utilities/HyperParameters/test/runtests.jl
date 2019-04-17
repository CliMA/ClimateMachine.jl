using Test
using CLIMA.HyperParameters

@testset "HyperParameters" begin

  hyper_params = HyperParameters.HyperParameterSet()
  @test hyper_params.value_typical[:N_subdomains] == 2
  @test hyper_params.value_min[:N_subdomains] == 2
  @test hyper_params.value_max[:N_subdomains] == 10
  @test hyper_params.value_typical[:Cs] == 12/100

end


