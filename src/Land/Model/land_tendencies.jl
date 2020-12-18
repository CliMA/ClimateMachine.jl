eq_tends(pv::PV, land::LandModel, ::Flux{FirstOrder}) where {PV} = ()
eq_tends(pv::PV, land::LandModel, ::Source) where {PV} = ()
eq_tends(pv::PV, land::LandModel, tt::Flux{SecondOrder}) where {PV} = (
    eq_tends(pv, land.soil.heat, tt)...,
)

struct HeatDiffusionFlux{PV <: VolIntEnergy} <: TendencyDef{Flux{SecondOrder}, PV} end
function flux(
    ::HeatDiffusionFlux{VolIntEnergy},
    land,
    args
)
    @unpack aux, diffusive = args
    ρe_int_l = volumetric_internal_energy_liq(aux.soil.heat.T, land.param_set)
    diffusive_heat_flux = -diffusive.soil.heat.κ∇T
    return diffusive_heat_flux
end


eq_tends(pv::PV, heat::SoilHeatModel, tt::Flux{SecondOrder}) where {PV <: VolIntEnergy} = (
    HeatDiffusionFlux{PV}(),
)
