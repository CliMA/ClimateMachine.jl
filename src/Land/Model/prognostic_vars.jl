#####
##### Prognostic Variables
#####

prognostic_vars(water::PrescribedWaterModel) = ()
prognostic_vars(water::SoilWaterModel) =
    (VolumetricLiquidFraction(), VolumetricIceFraction())
prognostic_vars(heat::PrescribedTemperatureModel) = ()
prognostic_vars(heat::SoilHeatModel) = (VolumetricInternalEnergy(),)

prognostic_vars(land::LandModel) =
    (prognostic_vars(land.soil.water)..., prognostic_vars(land.soil.heat)...)

get_prog_state(state, ::VolumetricLiquidFraction) = (state.soil.water, :ϑ_l)
get_prog_state(state, ::VolumetricIceFraction) = (state.soil.water, :θ_i)
get_prog_state(state, ::VolumetricInternalEnergy) = (state.soil.heat, :ρe_int)
