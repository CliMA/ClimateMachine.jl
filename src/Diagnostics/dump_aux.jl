function dump_aux_init(dgngrp, currtime)
    FT = eltype(Settings.Q)
    bl = Settings.dg.balance_law
    mpicomm = Settings.mpicomm
    mpirank = MPI.Comm_rank(mpicomm)

    if mpirank == 0
        # get dimensions for the interpolated grid
        dims = dimensions(dgngrp.interpol)

        # set up the variables we're going to be writing
        vars = OrderedDict()
        auxnames = flattenednames(vars_state_auxiliary(bl, FT))
        for varname in auxnames
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

function dump_aux_collect(dgngrp, currtime)
    interpol = dgngrp.interpol
    mpicomm = Settings.mpicomm
    dg = Settings.dg
    Q = Settings.Q
    FT = eltype(Q.data)
    bl = dg.balance_law
    mpirank = MPI.Comm_rank(mpicomm)

    iaux = similar(
        dg.state_auxiliary.data,
        interpol.Npl,
        number_state_auxiliary(bl, FT),
    )
    interpolate_local!(interpol, dg.state_auxiliary.data, iaux)

    all_aux_data = accumulate_interpolated_data(mpicomm, interpol, iaux)

    if mpirank == 0
        auxnames = flattenednames(vars_state_auxiliary(bl, FT))
        varvals = OrderedDict()
        for (vari, varname) in enumerate(auxnames)
            varvals[varname] = all_aux_data[:, :, :, vari]
        end
        append_data(dgngrp.writer, varvals, currtime)
    end

    MPI.Barrier(mpicomm)
    return nothing
end

function dump_aux_fini(dgngrp, currtime) end
