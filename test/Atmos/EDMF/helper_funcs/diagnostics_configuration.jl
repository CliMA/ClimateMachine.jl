"""
    config_diagnostics(driver_config, timeend; interval=nothing)

Returns the state and tendency diagnostic groups
"""
function config_diagnostics(driver_config, timeend; interval = nothing)
    FT = eltype(driver_config.grid)
    info = driver_config.config_info
    if interval == nothing
        interval = "$(cld(timeend, 2) + 10)ssecs"
        #interval = "10steps"
    end

    boundaries = [
        FT(0) FT(0) FT(0)
        FT(info.hmax) FT(info.hmax) FT(info.zmax)
    ]
    axes = (
        [FT(1)],
        [FT(1)],
        collect(range(boundaries[1, 3], boundaries[2, 3], step = FT(50)),),
    )
    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config,
        boundaries;
        axes = axes,
    )
    ds_dgngrp = setup_dump_state_diagnostics(
        SingleStackConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    )
    dt_dgngrp = setup_dump_tendencies_diagnostics(
        SingleStackConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    )
    return ClimateMachine.DiagnosticsConfiguration([ds_dgngrp, dt_dgngrp])
end
