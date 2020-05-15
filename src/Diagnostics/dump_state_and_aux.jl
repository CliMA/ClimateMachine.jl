function dump_state_and_aux_init(dgngrp, currtime)
    @assert dgngrp.interpol !== nothing
end

function get_dims(dgngrp)
    if dgngrp.interpol !== nothing
        if dgngrp.interpol isa InterpolationBrick
            if Array ∈ typeof(dgngrp.interpol.x1g).parameters
                h_x1g = dgngrp.interpol.x1g
                h_x2g = dgngrp.interpol.x2g
                h_x3g = dgngrp.interpol.x3g
            else
                h_x1g = Array(dgngrp.interpol.x1g)
                h_x2g = Array(dgngrp.interpol.x2g)
                h_x3g = Array(dgngrp.interpol.x3g)
            end
            dims = OrderedDict("x" => h_x1g, "y" => h_x2g, "z" => h_x3g)
        elseif dgngrp.interpol isa InterpolationCubedSphere
            if Array ∈ typeof(dgngrp.interpol.rad_grd).parameters
                h_rad_grd = dgngrp.interpol.rad_grd
                h_lat_grd = dgngrp.interpol.lat_grd
                h_long_grd = dgngrp.interpol.long_grd
            else
                h_rad_grd = Array(dgngrp.interpol.rad_grd)
                h_lat_grd = Array(dgngrp.interpol.lat_grd)
                h_long_grd = Array(dgngrp.interpol.long_grd)
            end
            dims = OrderedDict(
                "rad" => h_rad_grd,
                "lat" => h_lat_grd,
                "long" => h_long_grd,
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
    DA = ClimateMachine.array_type()
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
    #statenames = ("rho", "rhou", "rhov", "rhow", "rhoe")
    auxnames = flattenednames(vars_state_auxiliary(bl, FT))

    all_state_data = nothing
    all_aux_data = nothing
    if dgngrp.interpol !== nothing
        istate = DA(Array{FT}(
            undef,
            dgngrp.interpol.Npl,
            number_state_conservative(bl, FT),
        ))
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

        iaux = DA(Array{FT}(
            undef,
            dgngrp.interpol.Npl,
            number_state_auxiliary(bl, FT),
        ))
        interpolate_local!(dgngrp.interpol, dg.state_auxiliary.data, iaux)

        all_aux_data =
            accumulate_interpolated_data(mpicomm, dgngrp.interpol, iaux)
    else
        error("Dump of non-interpolated data currently unsupported")
    end

    if mpirank == 0
        dims = get_dims(dgngrp)
        dim_names = tuple(collect(keys(dims))...)

        statevarvals = OrderedDict()
        for i in 1:number_state_conservative(bl, FT)
            statevarvals[statenames[i]] =
                (dim_names, all_state_data[:, :, :, i])
        end
        write_data(dgngrp.writer, statefilename, dims, statevarvals, currtime)

        auxvarvals = OrderedDict()
        for i in 1:number_state_auxiliary(bl, FT)
            auxvarvals[auxnames[i]] = (dim_names, all_aux_data[:, :, :, i])
        end
        write_data(dgngrp.writer, auxfilename, dims, auxvarvals, currtime)
    end

    MPI.Barrier(mpicomm)
    return nothing
end

function dump_state_and_aux_fini(dgngrp, currtime) end
