using MPI, Test

@testset "Atmos/Dycore" begin
  # The code below was modified from the MPI.jl file runtests.jl
  #
  # Code coverage command line options; must correspond to src/julia.h
  # and src/ui/repl.c
  JL_LOG_NONE = 0
  JL_LOG_USER = 1
  JL_LOG_ALL = 2
  coverage_opts = Dict{Int, String}(JL_LOG_NONE => "none",
                                    JL_LOG_USER => "user",
                                    JL_LOG_ALL => "all")
  coverage_opt = coverage_opts[Base.JLOptions().code_coverage]
  testdir = dirname(@__FILE__)

  for (n, f) in [(1, "tracers_test.jl")
                 (3, "tracers_test.jl")
                 (1, "rising_thermal_bubble.jl")
                 (3, "rising_thermal_bubble.jl")
                ]
    cmd =  `mpiexec -n $n $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --code-coverage=$coverage_opt $(joinpath(testdir, f))`
    @info "Running MPI test..." n f cmd
    # Running this way prevents:
    #   Atmos/Dycore | No tests
    # since external tests are not returned as passed/fail
    @test (run(cmd); true)
  end
end
