const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

function do_output_vtk(
    mpicomm,
    vtkdir,
    dg,
    rhs_DGsource,
    rhs_analytical,
    model,
)
    mkpath(vtkdir)

    ## name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/compare_mpirank%04d",
        vtkdir,
        MPI.Comm_rank(mpicomm),
    )

    statenames = flattenednames(vars_state(model, Prognostic(), eltype(rhs_DGsource)))
    analytical_names = statenames .* "_analytical"

    writevtk(filename, rhs_DGsource, dg, statenames, rhs_analytical, analytical_names)

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## name of the pvtu file
        pvtuprefix = @sprintf("%s/compare", vtkdir)

        ## name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("compare_mpirank%04d", i - 1)
        end

        writepvtu(
            pvtuprefix,
            prefixes,
            (statenames..., analytical_names...),
            eltype(rhs_DGsource),
        )

        @info "Done writing VTK: $pvtuprefix"
    end
end

function do_output_jld2() end

