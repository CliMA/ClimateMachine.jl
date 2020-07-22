function dump_init(dgngrp, currtime, st::AbstractStateType)
    FT = eltype(Settings.Q)
    bl = Settings.dg.balance_law
    mpicomm = Settings.mpicomm
    mpirank = MPI.Comm_rank(mpicomm)

    if mpirank == 0
        # get dimensions for the interpolated grid
        dims = dimensions(dgngrp.interpol)

        # set up the variables we're going to be writing
        vars = OrderedDict()
        statenames = flattenednames(vars_state(bl, st, FT))
        for varname in statenames
            vars[varname] = (tuple(collect(keys(dims))...), FT, Dict())
        end

        dprefix = @sprintf(
            "%s_%s-%s",
            dgngrp.out_prefix,
            dgngrp.name,
            Settings.starttime,
        )
        dfilename = joinpath(Settings.output_dir, dprefix)
        init_data(dgngrp.writer, dfilename, dims, vars)
    end

    return nothing
end
