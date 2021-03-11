#####
##### Sources
#####

eq_tends(pv::AbstractPrognosticVariable, m::LandModel, tt::Source) =
    (m.source_dt[pv]...,)

#####
##### First order fluxes
#####

eq_tends(::PV, ::LandModel, ::Flux{FirstOrder}) where {PV} = ()


#####
##### Second order fluxes
#####

# Empty by default
eq_tends(pv::PV, ::AbstractSoilComponentModel, ::Flux{SecondOrder}) where {PV} =
    ()

eq_tends(pv::PV, land::LandModel, tt::Flux{SecondOrder}) where {PV} =
    (eq_tends(pv, land.soil.heat, tt)..., eq_tends(pv, land.soil.water, tt)...)

eq_tends(
    pv::PV,
    land::LandModel,
    tt::Flux{SecondOrder},
) where {PV <: VolumetricInternalEnergy} =
    (eq_tends(pv, land.soil.heat, land.soil.water, tt)...,)

# TODO: move to soi_heat.jl?
eq_tends(
    ::VolumetricInternalEnergy,
    ::SoilHeatModel,
    ::SoilWaterModel,
    ::Flux{SecondOrder},
) = (DiffHeatFlux(), DarcyDrivenHeatFlux())

# TODO: move to soi_water.jl?
eq_tends(
    ::VolumetricInternalEnergy,
    ::SoilHeatModel,
    ::PrescribedWaterModel,
    ::Flux{SecondOrder},
) = (DiffHeatFlux(),)

eq_tends(::VolumetricLiquidFraction, ::SoilWaterModel, ::Flux{SecondOrder}) =
    (DarcyFlux(),)
