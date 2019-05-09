using MPI, Test

@testset "Balance Law Solver" begin
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

  for (n, f) in [
                 (1, "Euler/isentropic_vortex_standalone.jl")
                 (1, "Euler/isentropic_vortex_standalone_aux.jl")
                 (1, "util/grad_test.jl")
                 (1, "Euler/isentropic_vortex_standalone_source.jl")
                 (1, "Euler/isentropic_vortex_standalone_bc.jl")
                 (1, "conservation/sphere.jl")
                 (1, "compressible_Navier_Stokes/mms_bc.jl")
                 (3, "compressible_Navier_Stokes/mms_bc.jl")
                 (1, "Sphere/advection_sphere.jl")
                ]
    cmd =  `mpiexec -n $n $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --code-coverage=$coverage_opt $(joinpath(testdir, f))`
    @info "Running MPI test..." n f cmd
    # Running this way prevents:
    #   Balance Law Solver | No tests
    # since external tests are not returned as passed/fail
    @test (run(cmd); true)
  end

  if !parse(Bool, lowercase(get(ENV,"TRAVIS","false")))
    for (n, f) in [
                 (3, "Euler/isentropic_vortex_standalone.jl")
                 (3, "Euler/isentropic_vortex_standalone_aux.jl")
                 (3, "Euler/isentropic_vortex_standalone_source.jl")
                 (3, "Euler/isentropic_vortex_standalone_bc.jl")
                 (3, "conservation/sphere.jl")
                 (3, "compressible_Navier_Stokes/mms_bc.jl")
                 (2, "Sphere/advection_sphere.jl")
                  ]
      cmd =  `mpiexec -n $n $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --code-coverage=$coverage_opt $(joinpath(testdir, f))`
      @info "Running MPI test..." n f cmd
      # Running this way prevents:
      #   Balance Law Solver | No tests
      # since external tests are not returned as passed/fail
      @test (run(cmd); true)
    end
  end

  if "linux" != lowercase(get(ENV,"TRAVIS_OS_NAME",""))
    for (n, f) in [
                   (1, "../examples/ex_001_periodic_advection.jl")
                   (3, "../examples/ex_001_periodic_advection.jl")
                   (1, "../examples/ex_002_solid_body_rotation.jl")
                   (3, "../examples/ex_002_solid_body_rotation.jl")
                  ]
      cmd =  `mpiexec -n $n $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --code-coverage=$coverage_opt $(joinpath(testdir, f))`
      @info "Running MPI test..." n f cmd
      # Running this way prevents:
      #   Balance Law Solver | No tests
      # since external tests are not returned as passed/fail
      @test (run(cmd); true)
    end
  end

end
