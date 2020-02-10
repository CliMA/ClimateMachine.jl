export get_unit, unit_alias, space_unit, time_unit, mass_unit, temperature_unit 

const unit_glossary = Dict(
  :accel       => u"m/s^2",
  :force       => u"kg*m/s^2",
  :frequency   => u"s^-1",
  :velocity    => u"m/s",
  :space       => u"m",
  :time        => u"s",
  :temperature => u"K",
  :mass        => u"kg",
  :energy      => u"J",
  :gravpot     => u"J/kg",
  :latent      => u"J/kg",
  :shc         => u"J/kg/K",
  :energypv    => u"J/m^3",
  :energyflux  => u"J/m^2/s",
  :density     => u"kg/m^3",
  :dynvisc     => u"kg/m/s",
  :kinvisc     => u"m^2/s",
  :massflux    => u"kg/m^2/s",
  :pressure    => u"Pa",
  :lincond     => u"K/m"
)

@inline unit_alias(s::Symbol) = _unit_alias(Val(s))
@generated _unit_alias(::Val{sym}) where {sym} = upreferred(unit_glossary[sym])

function unit_annotations end
unit_annotations(m) = false
@inline get_unit(bl, s::Symbol) = unit_annotations(bl) ? unit_alias(s) : NoUnits
@inline get_unit(b::Bool, s::Symbol) = b ? unit_alias(s) : NoUnits

space_unit(bl)       = get_unit(bl, :space)
time_unit(bl)        = get_unit(bl, :time)
mass_unit(bl)        = get_unit(bl, :mass)
temperature_unit(bl) = get_unit(bl, :temperature)
