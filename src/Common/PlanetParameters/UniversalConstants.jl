"""
    UniversalConstants

These constants are planet-independent.
"""
module UniversalConstants

export gas_constant, light_speed, h_Planck, k_Boltzmann, Stefan, astro_unit

"""
    gas_constant

Universal gas constant (J/mol/K)
"""
gas_constant() = 8.3144598

"""
    light_speed

Speed of light in vacuum (m/s)
"""
light_speed() = 2.99792458e8

"""
    h_Planck

Planck constant (m^2 kg/s)
"""
h_Planck() = 6.626e-34

"""
    k_Boltzmann

Boltzmann constant (m^2 kg/s^2/K)
"""
k_Boltzmann() = 1.381e-23

"""
    Stefan

Stefan-Boltzmann constant (W/m^2/K^4)
"""
Stefan() = 5.670e-8

"""
    astro_unit

Astronomical unit (m)
"""
astro_unit() = 1.4959787e11

end
