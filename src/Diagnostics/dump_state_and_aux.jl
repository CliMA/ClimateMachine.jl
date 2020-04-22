function dump_state_and_aux_init(dgngrp, currtime)
    if isnothing(dgngrp.interpol)
        @warn """
            Diagnostics ($dgngrp.name): requires interpolation!
            """
    end
end

function dump_state_and_aux_collect(dgngrp, currtime)
    interpol = dgngrp.interpol
    if isnothing(interpol)
        @warn """
            Diagnostics ($dgngrp.name): requires interpolation!
            """
        return nothing
    end

    mpicomm = Settings.mpicomm
    dg = Settings.dg
    Q = Settings.Q
    FT = eltype(Q.data)
    bl = dg.balance_law
    mpirank = MPI.Comm_rank(mpicomm)

    # filename (may also want to take out)
    dprefix = @sprintf(
        "%s_%s-%s-num%04d",
        dgngrp.out_prefix,
        dgngrp.name,
        Settings.starttime,
        dgngrp.num
    )
    statefilename = joinpath(Settings.output_dir, dprefix)
    auxfilename = joinpath(Settings.output_dir, "$(dprefix)_aux")

    statenames = flattenednames(vars_state_conservative(bl, FT))
    auxnames = flattenednames(vars_state_auxiliary(bl, FT))

    istate = similar(Q.data, interpol.Npl, number_state_conservative(bl, FT))
    interpolate_local!(interpol, Q.data, istate)

    if interpol isa InterpolationCubedSphere
        # TODO: get indices here without hard-coding them
        _ρu, _ρv, _ρw = 2, 3, 4
        project_cubed_sphere!(interpol, istate, (_ρu, _ρv, _ρw))
    end

    all_state_data = accumulate_interpolated_data(mpicomm, interpol, istate)

    iaux = similar(
        dg.state_auxiliary.data,
        interpol.Npl,
        number_state_auxiliary(bl, FT),
    )
    interpolate_local!(interpol, dg.state_auxiliary.data, iaux)

    all_aux_data = accumulate_interpolated_data(mpicomm, interpol, iaux)

    if mpirank == 0
        dims = dimensions(interpol)
        dim_names = tuple(collect(keys(dims))...)

        statevarvals = OrderedDict()
        for i in 1:number_state_conservative(bl, FT)
            statevarvals[statenames[i]] =
                (dim_names, all_state_data[:, :, :, i], Dict())
        end
        write_data(dgngrp.writer, statefilename, dims, statevarvals, currtime)

        auxvarvals = OrderedDict()
        for i in 1:number_state_auxiliary(bl, FT)
            auxvarvals[auxnames[i]] =
                (dim_names, all_aux_data[:, :, :, i], Dict())
        end
        write_data(dgngrp.writer, auxfilename, dims, auxvarvals, currtime)
    end

    MPI.Barrier(mpicomm)
    return nothing
end

function dump_state_and_aux_fini(dgngrp, currtime) end
