using MPI

function runmpi(file; ntasks = 1)
    MPI.Initialized() &&
    !MPI.Finalized() &&
    error(
        "runmpi does not work if MPI has been " *
        "Initialized but not Finalized",
    )

    if MPI.MPI_LIBRARY == MPI.OpenMPI
        oversubscribe = `--oversubscribe`
    else
        oversubscribe = ``
    end

    @info "Running MPI test..." file ntasks
    # Running this way prevents:
    #   Balance Law Solver | No tests
    # since external tests are not returned as passed/fail
    @time @test mpiexec() do cmd
        run(`$cmd $oversubscribe -n $ntasks $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $file`)
        true
    end
end
