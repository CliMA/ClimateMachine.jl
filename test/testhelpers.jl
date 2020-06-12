using MPI

function runmpi(file; ntasks = 1)
    # by default some mpi runtimes will
    # complain if more resources (processes)
    # are requested than available on the node
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
