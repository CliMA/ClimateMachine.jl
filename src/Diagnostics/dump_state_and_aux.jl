function dump_state_and_aux_init(diagrp, currtime)
    @assert diagrp.interpol !== nothing
end

function dump_state_and_aux_collect(diagrp, currtime)
    DA = CLIMA.array_type()
    dg = Settings.dg
    Q = Settings.Q
    FT = eltype(Q.data)
    bl = dg.balancelaw

    istate = DA(Array{FT}(undef, diagrp.interpol.Npl, num_state(bl, FT)))
    iaux = DA(Array{FT}(undef, diagrp.interpol.Npl, num_aux(bl, FT)))

    # interpolate and save
    interpolate_local!(
        diagrp.interpol,
        Q.data,
        istate,
        project = diagrp.project,
    )
    interpolate_local!(
        diagrp.interpol,
        dg.auxstate.data,
        iaux,
        project = diagrp.project,
    )

    # filename (may also want to take out)
    nprefix = @sprintf(
        "%s_%s-%s-step%04d",
        diagrp.out_prefix,
        diagrp.name,
        Settings.starttime,
        diagrp.step
    )
    statefilename = joinpath(Settings.output_dir, "$(nprefix).nc")
    auxfilename = joinpath(Settings.output_dir, "$(nprefix)_aux.nc")

    statenames = flattenednames(vars_state(bl, FT))
    #statenames = ("rho", "rhou", "rhov", "rhow", "rhoe")
    auxnames = flattenednames(vars_aux(bl, FT))

    write_interpolated_data(diagrp.interpol, istate, statenames, statefilename)
    write_interpolated_data(diagrp.interpol, iaux, auxnames, auxfilename)

    return nothing
end

function dump_state_and_aux_fini(diagrp, currtime) end
