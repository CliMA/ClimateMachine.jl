# Spectrum calculator for AtmosGCM

struct AtmosGCMSpectraDiagnosticsParams <: DiagnosticsGroupParams
    nor::Float64
end

"""
    setup_atmos_spectra_diagnostics(
        ::AtmosGCMConfigType,
        interval::String,
        out_prefix::String;
        writer = NetCDFWriter(),
        interpol = nothing,
        nor = 1.0,
    )

Create and return a `DiagnosticsGroup` containing a diagnostic that dumps
the spectrum at the specified `interval` after the velocity fields have
been interpolated, into NetCDF files prefixed by `out_prefix`.
"""
function setup_atmos_spectra_diagnostics(
    ::AtmosGCMConfigType,
    interval::String,
    out_prefix::String;
    writer = NetCDFWriter(),
    interpol = nothing,
    nor = 1.0,
)
    @assert !isnothing(interpol)

    return DiagnosticsGroup(
        "AtmosGCMSpectra",
        Diagnostics.atmos_gcm_spectra_init,
        Diagnostics.atmos_gcm_spectra_fini,
        Diagnostics.atmos_gcm_spectra_collect,
        interval,
        out_prefix,
        writer,
        interpol,
        AtmosGCMSpectraDiagnosticsParams(nor),
    )
end

function get_spectra(mpicomm, mpirank, Q, bl, interpol, nor)
    FT = eltype(Q)
    if array_device(Q) isa CPU
        ArrayType = Array
    else
        ArrayType = CUDA.CuArray
    end

    istate = ArrayType{FT}(undef, interpol.Npl, number_states(bl, Prognostic()))
    interpolate_local!(interpol, Q.realdata, istate)

    # TODO: get indices here without hard-coding them
    _ρu, _ρv, _ρw = 2, 3, 4
    project_cubed_sphere!(interpol, istate, (_ρu, _ρv, _ρw))

    all_state_data = accumulate_interpolated_data(mpicomm, interpol, istate)

    if mpirank == 0
        var_grid = all_state_data[:, :, :, 3] ./ all_state_data[:, :, :, 1]
        dims = dimensions(interpol)
        lat = dims["lat"][1]
        lon = dims["long"][1]
        level = dims["level"][1] .- FT(planet_radius(Settings.param_set))

        mass_weight = ones(FT, length(level)) # TODO: interpolate on pressure levs and do mass weighted calculations
        spectrum_1d, m = power_spectrum_1d(
            AtmosGCMConfigType(),
            var_grid,
            level,
            lat,
            lon,
            mass_weight,
        )
        spectrum_2d, m_and_n, _, __ =
            power_spectrum_2d(AtmosGCMConfigType(), var_grid, mass_weight)
        return spectrum_1d, m, spectrum_2d, m_and_n
    end
    return nothing, nothing, nothing, nothing
end

function atmos_gcm_spectra_init(dgngrp, currtime)
    Q = Settings.Q
    bl = Settings.dg.balance_law
    mpicomm = Settings.mpicomm
    mpirank = MPI.Comm_rank(mpicomm)
    FT = eltype(Q)
    interpol = dgngrp.interpol
    nor = dgngrp.params.nor

    # get the 1d and 2d spectra and their associated wavenumbers
    spectrum_1d, m, spectrum_2d, m_and_n =
        get_spectra(mpicomm, mpirank, Q, bl, interpol, nor)

    if mpirank == 0
        # Setup dimensions
        interpol_dims = dimensions(interpol)
        lat = interpol_dims["lat"]
        level = interpol_dims["level"]
        level = (level[1] .- FT(planet_radius(Settings.param_set)), level[2])
        num_fourier = ((2 * length(lat[1]) - 1) / 3) # number of truncated zonal wavenumbers
        num_spherical = (num_fourier + 1) # number of total wavenumbers (n)
        dims = OrderedDict(
            "m" => (collect(FT, 1:1:length(lat[1])) .- 1, Dict()),
            "m_t" => (collect(FT, 1:1:num_fourier) .- 1, Dict()),
            "n" => (collect(FT, 1:1:num_spherical) .- 1, Dict()),
            "level" => level,
            "lat" => lat,
        )
        # Setup variables (NB: 2d spectrum is on the truncated wavenumber grid)
        vars = OrderedDict(
            "spectrum_1d" => (("m", "lat", "level"), FT, Dict()),
            "spectrum_2d" => (("m_t", "n", "level"), FT, Dict()),
        )

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

function atmos_gcm_spectra_collect(dgngrp, currtime)
    Q = Settings.Q
    bl = Settings.dg.balance_law
    mpicomm = Settings.mpicomm
    mpirank = MPI.Comm_rank(mpicomm)
    FT = eltype(Q)
    interpol = dgngrp.interpol
    nor = dgngrp.params.nor

    spectrum_1d, _, spectrum_2d, _ =
        get_spectra(mpicomm, mpirank, Q, bl, interpol, nor)

    if mpirank == 0
        varvals = OrderedDict(
            "spectrum_1d" => spectrum_1d,
            "spectrum_2d" => spectrum_2d,
        )
        append_data(dgngrp.writer, varvals, currtime)
    end

    MPI.Barrier(mpicomm)
    return nothing
end

function atmos_gcm_spectra_fini(dgngrp, currtime) end
