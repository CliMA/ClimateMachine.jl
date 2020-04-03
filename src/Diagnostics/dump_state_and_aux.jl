function dump_state_and_aux_init(dgngrp, currtime)
    @assert dgngrp.interpol !== nothing
end

function get_dims(dgngrp)
    if dgngrp.interpol !== nothing
        if dgngrp.interpol isa InterpolationBrick
            dims = OrderedDict(
                "x" => dgngrp.interpol.x1g,
                "y" => dgngrp.interpol.x2g,
                "z" => dgngrp.interpol.x3g,
            )
        elseif dgngrp.interpol isa InterpolationCubedSphere
            dims = OrderedDict(
                "rad" => dgngrp.interpol.rad_grd,
                "lat" => dgngrp.interpol.lat_grd,
                "long" => dgngrp.interpol.long_grd,
            )
        else
            error("Unsupported interpolation topology $(dgngrp.interpol)")
        end
    else
        error("Dump of non-interpolated data currently unsupported")
    end

    return dims
end

function dump_state_and_aux_collect(dgngrp, currtime)
    DA = CLIMA.array_type()
    mpicomm = Settings.mpicomm
    dg = Settings.dg
    Q = Settings.Q
    FT = eltype(Q.data)
    bl = dg.balancelaw
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

    statenames = flattenednames(vars_state(bl, FT))
    #statenames = ("rho", "rhou", "rhov", "rhow", "rhoe")
    auxnames = flattenednames(vars_aux(bl, FT))

    all_state_data = nothing
    all_aux_data = nothing
    if dgngrp.interpol !== nothing
        istate = DA(Array{FT}(undef, dgngrp.interpol.Npl, num_state(bl, FT)))
        interpolate_local!(dgngrp.interpol, Q.data, istate)

        if dgngrp.project
            if dgngrp.interpol isa InterpolationCubedSphere
                # TODO: get indices here without hard-coding them
                _ρu, _ρv, _ρw = 2, 3, 4
                project_cubed_sphere!(dgngrp.interpol, istate, (_ρu, _ρv, _ρw))
            else
                error("Can only project for InterpolationCubedSphere")
            end
        end

        all_state_data =
            accumulate_interpolated_data(mpicomm, dgngrp.interpol, istate)

        iaux = DA(Array{FT}(undef, dgngrp.interpol.Npl, num_aux(bl, FT)))
        interpolate_local!(dgngrp.interpol, dg.auxstate.data, iaux)

        all_aux_data =
            accumulate_interpolated_data(mpicomm, dgngrp.interpol, iaux)
    else
        error("Dump of non-interpolated data currently unsupported")
    end

    dims = get_dims(dgngrp)

    statevarvals = OrderedDict()
    for i in 1:num_state(bl, FT)
        statevarvals[statenames[i]] = all_state_data[:, :, :, i]
    end
    write_data(dgngrp.writer, statefilename, dims, statevarvals)

    auxvarvals = OrderedDict()
    for i in 1:num_aux(bl, FT)
        auxvarvals[auxnames[i]] = all_aux_data[:, :, :, i]
    end
    write_data(dgngrp.writer, auxfilename, dims, auxvarvals)

    return nothing
end

function dump_state_and_aux_fini(dgngrp, currtime) end
