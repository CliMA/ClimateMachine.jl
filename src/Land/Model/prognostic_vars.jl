#####
##### Prognostic Variables
#####

prognostic_vars(water::PrescribedWaterModel) = ()
prognostic_vars(water::SoilWaterModel) =
    (VolumetricLiquidFraction(), VolumetricIceFraction())

prognostic_vars(heat::PrescribedTemperatureModel) = ()
prognostic_vars(heat::SoilHeatModel) = (VolumetricInternalEnergy(),)

prognostic_vars(surface::NoSurfaceFlowModel) = ()
prognostic_vars(surface::OverlandFlowModel) = (SurfaceWaterHeight(),)

prognostic_vars(surface::NoSnowModel) = ()
prognostic_vars(surface:: SingleLayerSnowModel) = (SnowWaterEquivalent(),
                                                         SnowVolumetricInternalEnergy(),
                                                         )


prognostic_vars(land::LandModel) = (
    prognostic_vars(land.soil.water)...,
    prognostic_vars(land.soil.heat)...,
    prognostic_vars(land.surface)...,
    prognostic_vars(land.snow)...,
)



get_prog_state(state, ::VolumetricLiquidFraction) = (state.soil.water, :ϑ_l)
get_prog_state(state, ::VolumetricIceFraction) = (state.soil.water, :θ_i)
get_prog_state(state, ::VolumetricInternalEnergy) = (state.soil.heat, :ρe_int)
get_prog_state(state, ::SurfaceWaterHeight) = (state.surface, :height)
get_prog_state(state, ::SnowWaterEquivalent) = (state.snow, :swe)
get_prog_state(state, ::SnowVolumetricInternalEnergy) = (state.snow, :ρe_int)
