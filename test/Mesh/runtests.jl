using MPI, Test

MPI.Init()

include("topology.jl")

MPI.Finalize()

@testset "MPI Jobs" begin
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

  for (n, f) in [(5, "mpi_connect_1d.jl")
                 (2, "mpi_connect_ell.jl")
                 (3, "mpi_connect.jl")
                 (3, "mpi_connect_stacked.jl")
                 (2, "mpi_connect_stacked_3d.jl")
                 (5, "mpi_connect_sphere.jl")]
    if haskey(ENV, "SLURM_JOB_ID")
      cmd =  `mpiexec --oversubscribe -n $n $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --code-coverage=$coverage_opt $(joinpath(testdir, f))`
    else
      cmd =  `mpiexec -n $n $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --code-coverage=$coverage_opt $(joinpath(testdir, f))`
    end
    @info "Running MPI test..." n f cmd
    @test (run(cmd); true)
  end
end
