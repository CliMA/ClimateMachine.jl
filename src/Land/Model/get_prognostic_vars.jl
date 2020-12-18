prognostic_vars(land::LandModel) = (
    prognostic_vars(land.soil.heat)...,
)

