using Test, Pkg, Pkg.TOML
using MPI

ENV["DATADEPS_ALWAYS_ACCEPT"] = true
ENV["JULIA_LOG_LEVEL"] = "WARN"

test_suite_file = joinpath(@__DIR__, "..", "test", "suite.toml")
test_suite = TOML.parsefile(test_suite_file)

_active_CI = "azure" # "slurm", "azure", get from ENV

"""
    runmpi(filename, n)

Run a single MPI job. Borrowed from CLIMA/test/testhelpers.jl
"""
function runmpi(filename, n)
    MPI.Initialized() &&
    !MPI.Finalized() &&
    error(
        "runmpi does not work if MPI has been " *
        "Initialized but not Finalized",
    )

    # The code below was modified from the MPI.jl file runtests.jl
    #
    # Code coverage command line options; must correspond to src/julia.h
    # and src/ui/repl.c
    JL_LOG_NONE = 0
    JL_LOG_USER = 1
    JL_LOG_ALL = 2
    coverage_opts = Dict{Int, String}(
        JL_LOG_NONE => "none",
        JL_LOG_USER => "user",
        JL_LOG_ALL => "all",
    )
    coverage_opt = coverage_opts[Base.JLOptions().code_coverage]
    testdir = dirname(filename)

    if !Sys.iswindows() &&
       occursin("OpenRTE", read(`mpiexec --version`, String))
        oversubscribe = `--oversubscribe`
    else
        oversubscribe = ``
    end

    cmd = `mpiexec $oversubscribe -n $n $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --code-coverage=$coverage_opt $(joinpath(testdir, filename))`

    @info "Running MPI test..." n filename cmd
    # Running this way prevents:
    #   Balance Law Solver | No tests
    # since external tests are not returned as passed/fail
    val, t, bytes, gctime, memallocs = @timed @test (run(cmd); true)
    return t
end

function validate_job(jobname, joblist, filename)
    njobs = joblist["njobs"]
    @assert njobs isa Int
    @assert njobs ≥ 0
    isfile(filename) || error("File $(filename) does not exist")

    if !all([length(v) == njobs for (k, v) in joblist if k ≠ "njobs"])
        @show jobname, joblist
        error("njobs ≠ array size")
    end
end

"""
    runtests(tests)

Run test suite from imported `tests` specified in TOML file.
Assumes TOML structure, for example,

```
[BrickMesh]
filename = "test/Mesh/BrickMesh.jl"

    [BrickMesh.matrix.CPU]
    njobs        = 1
    frequency    = ["continuously"]
    active_CI    = ["azure"]
    time_limit   = [""]
    args         = [""]
    parallel     = [true]
    use_mpiexec  = [false]
    mpiranks     = [3]

    [BrickMesh.matrix.GPU]
    njobs        = 3
    frequency    = ["continuously", "nightly", "weekly"   ]
    active_CI    = ["azure"       , "azure"  , "slurm"    ]
    time_limit   = [""            , ""       , "02:00:00" ]
    args         = [""            , ""       , ""         ]
    parallel     = [true          , true     , true       ]
    use_mpiexec  = [true          , true     , true       ]
    mpiranks     = [1             , 3        , 10         ]
```

"""
function runtests(tests; validate_only = false)
    local t
    for (jobname, config) in tests
        filename = joinpath(@__DIR__, "..", config["filename"])
        for (arch, joblist) in config["matrix"]
            validate_job(jobname, joblist, filename)
            joblist["njobs"] == 0 && continue
            validate_only && continue

            for (
                time_limit,
                parallel,
                use_mpiexec,
                active_CI,
                args,
                frequency,
                njobs,
                mpiranks,
            ) in zip(
                joblist["time_limit"],
                joblist["parallel"],
                joblist["use_mpiexec"],
                joblist["active_CI"],
                joblist["args"],
                joblist["frequency"],
                joblist["njobs"],
                joblist["mpiranks"],
            )
                t = 0
                # ------ Single job
                if _active_CI == active_CI
                    if parallel && use_mpiexec
                        println("Starting tests for parallel job $(jobname)")
                        t = runmpi(filename, mpiranks)
                        println("Completed tests for parallel job $(jobname), $(round(Int, t)) seconds elapsed")
                    else
                        println("Starting tests for serial job $(jobname)")
                        t = @elapsed include(filename)
                        println("Completed tests for serial job $(jobname), $(round(Int, t)) seconds elapsed")
                    end
                end
                # ------
                tests[jobname]["matrix"][arch]["runtime"] = t

            end
        end
    end
    return tests
end

MPI.Initialized() && MPI.Finalize()

tests = runtests(test_suite; validate_only = true)
println("Total number of CPU jobs: $(sum([config["matrix"]["CPU"]["njobs"] for config in values(tests)] ))")
println("Total number of GPU jobs: $(sum([config["matrix"]["GPU"]["njobs"] for config in values(tests)] ))")
runtests(test_suite)

# Export to see runtime for each job:
# open(joinpath(test_dir,"suite_results.TOML"), "w") do io
#   TOML.print(io, tests)
# end
