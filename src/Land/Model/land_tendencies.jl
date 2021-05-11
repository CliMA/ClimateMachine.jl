#####
##### Sources
#####

eq_tends(pv::AbstractPrognosticVariable, m::LandModel, tt::Source) =
    (m.source_dt[pv]...,)

#####
##### First order fluxes
#####

eq_tends(pv::PV, land::LandModel, tt::Flux{FirstOrder}) where {PV} = (
    eq_tends(pv, land.soil.heat, tt)...,
    eq_tends(pv, land.soil.water, tt)...,
    eq_tends(pv, land.surface, tt)...,
)

eq_tends(::PV, ::AbstractSoilComponentModel, ::Flux{FirstOrder}) where {PV} = ()
eq_tends(::PV, ::NoSurfaceFlowModel, ::Flux{FirstOrder}) where {PV} = ()
# eq_tends gets called for each PV for each subcomponent! so we need this
eq_tends(pv::PV, ::OverlandFlowModel, ::Flux{FirstOrder}) where {PV} = ()
# before refining to this
eq_tends(::SurfaceWaterHeight, ::OverlandFlowModel, ::Flux{FirstOrder}) =
    (VolumeAdvection(),)

#####
##### Second order fluxes
#####

# Empty by default
# maybe replace with PV, BalanceLaw?
eq_tends(pv::PV, ::AbstractSoilComponentModel, ::Flux{SecondOrder}) where {PV} =
    ()
eq_tends(pv::PV, ::OverlandFlowModel, ::Flux{SecondOrder}) where {PV} = ()
eq_tends(pv::PV, ::NoSurfaceFlowModel, ::Flux{SecondOrder}) where {PV} = ()

eq_tends(pv::PV, land::LandModel, tt::Flux{SecondOrder}) where {PV} = (
    eq_tends(pv, land.soil.heat, tt)...,
    eq_tends(pv, land.soil.water, tt)...,
    eq_tends(pv, land.surface, tt)...,
)

eq_tends(
    pv::PV,
    land::LandModel,
    tt::Flux{SecondOrder},
) where {PV <: VolumetricInternalEnergy} =
    (eq_tends(pv, land.soil.heat, land.soil.water, tt)...,)

eq_tends(
    ::VolumetricInternalEnergy,
    ::SoilHeatModel,
    ::SoilWaterModel,
    ::Flux{SecondOrder},
) = (DiffHeatFlux(), DarcyDrivenHeatFlux())

eq_tends(
    ::VolumetricInternalEnergy,
    ::SoilHeatModel,
    ::PrescribedWaterModel,
    ::Flux{SecondOrder},
) = (DiffHeatFlux(),)

eq_tends(::VolumetricLiquidFraction, ::SoilWaterModel, ::Flux{SecondOrder}) =
    (DarcyFlux(),)
