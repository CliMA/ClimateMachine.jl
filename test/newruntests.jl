using Test, Pkg.TOML
using MPI

test_dir = joinpath(@__DIR__,"test")
test_suite = TOML.parsefile(joinpath(test_dir,"suite.TOML"))

_active_CI = "azure" # "slurm", "azure", get from ENV

"""
    runmpi(filename, n)

Run a single MPI job. Borrowed from CLIMA/test/testhelpers.jl
"""
function runmpi(filename, n)
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
  testdir = dirname(filename)

  if !Sys.iswindows() && occursin( "OpenRTE", read(`mpiexec --version`, String))
    oversubscribe = `--oversubscribe`
  else
    oversubscribe = ``
  end

  cmd = `mpiexec $oversubscribe -n $n $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --code-coverage=$coverage_opt $(joinpath(testdir, filename))`

  @info "Running MPI test..." n filename cmd
  # Running this way prevents:
  #   Balance Law Solver | No tests
  # since external tests are not returned as passed/fail
  @time @test (run(cmd); true)
end

"""
    runtests(tests)

Run test suite from imported `tests` specified in TOML file.
Assumes TOML structure, for example,

```
[BrickMesh]
filename = "test/Mesh/BrickMesh.jl"

    [BrickMesh.matrix.CPU]
    njobs      = 1
    frequency  = ["continuously"]
    active_CI  = ["azure"]
    time_limit = [""]
    args       = [""]
    parallel   = [true]
    mpiranks   = [3]

    [BrickMesh.matrix.GPU]
    njobs      = 3
    frequency  = ["continuously", "nightly", "weekly"   ]
    active_CI  = ["azure"       , "azure"  , "slurm"    ]
    time_limit = [""            , ""       , "02:00:00" ]
    args       = [""            , ""       , ""         ]
    parallel   = [true          , true     , true       ]
    mpiranks   = [1             , 3        , 10         ]
```

"""
function runtests(tests)
  local t
  for (jobname, config) in tests
    filename = config["filename"]
    _test = joinpath(@__DIR__,"..",filename)
    for (arch,joblist) in config["matrix"]
      for (time_limit,
           parallel,
           active_CI,
           args,
           frequency,
           njobs,
           mpiranks) in
        zip(joblist["time_limit"],
            joblist["parallel"],
            joblist["active_CI"],
            joblist["args"],
            joblist["frequency"],
            joblist["njobs"],
            joblist["mpiranks"])

        t = 0
        # ------ Single job
        if _active_CI == active_CI
          if parallel

            t = runmpi(filename, mpiranks)

          else

              println("Starting tests for $(jobname)")
              t = @elapsed include(_test)
              println("Completed tests for $(jobname), $(round(Int, t)) seconds elapsed")

          end
        end
        # ------
        tests[jobname]["matrix"][arch]["runtime"] = t

      end
    end
  end
  return nothing
end

MPI.Initialized() && MPI.Finalize()

runtests(test_suite)

# Export to see runtime for each job:
# open(joinpath(test_dir,"suite_results.TOML"), "w") do io
#   TOML.print(io, tests)
# end
