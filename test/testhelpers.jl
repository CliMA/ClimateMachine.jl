using MPI

function runmpi(tests, file)
  MPI.Initialized() && !MPI.Finalized() &&
  error("runmpi does not work if MPI has been "*
        "Initialized but not Finalized")

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
  testdir = dirname(file)

  if !Sys.iswindows() && occursin( "OpenRTE", read(`mpiexec --version`, String))
    oversubscribe = `--oversubscribe`
  else
    oversubscribe = ``
  end

  for (n, f) in tests
    cmd = `mpiexec $oversubscribe -n $n $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --code-coverage=$coverage_opt $(joinpath(testdir, f))`

    @info "Running MPI test..." n f cmd
    # Running this way prevents:
    #   Balance Law Solver | No tests
    # since external tests are not returned as passed/fail
    @time @test (run(cmd); true)
  end
end
