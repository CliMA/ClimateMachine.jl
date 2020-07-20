using MPI

function runmpi(file; ntasks = 1, localhost = false)
    localhostenv = try
        parse(Bool, get(ENV, "CLIMATEMACHINE_TEST_RUNMPI_LOCALHOST", "false"))
    catch
        false
    end
    # Force mpiexec to exec on the localhost node
    # TODO: MicrosoftMPI mpiexec has issues if mpiexec.exe is in a different 
    # folder as the MPI script to run so ignore for now.
    if (localhost || localhostenv) && (MPI.MPI_LIBRARY != MPI.MicrosoftMPI)
        localhostonly = `-host localhost`
    else
        localhostonly = ``
    end
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
        run(`$cmd $localhostonly $oversubscribe -np $ntasks $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $file`)
        true
    end
end
