const unit_glossary = Dict(
  :accel    => u"m/s^2",
  :velocity => u"m/s",
  :space    => u"m",
  :time     => u"s",
  :temp     => u"K",
  :mass     => u"kg",
  :energy   => u"J",
  :gravpot  => u"J/kg",
  :latent   => u"J/kg",
  :shc      => u"J/kg/K",
  :energypv => u"J/m^3",
  :density  => u"kg/m^3",
  :dinvisc  => u"kg/m/s",
  :massflux => u"kg/m^2/s",
  :pressure => u"Pa",
  :lincond  => u"K/m"
)

space_unit(x...)    = NoUnits
time_unit(x...)     = NoUnits
mass_unit(x...)     = NoUnits
temp_unit(x...)     = NoUnits

velocity_unit(x...) = space_unit(x...) / time_unit(x...)
accel_unit(x...)    = velocity_unit(x...) / time_unit(x...)
energy_unit(x...)   = mass_unit(x...) * accel_unit(x...) * space_unit(x...)
gravpot_unit(x...)  = energy_unit(x...) / mass_unit(x...)
