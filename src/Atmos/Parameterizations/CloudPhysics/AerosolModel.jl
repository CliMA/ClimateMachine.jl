"""
AerosolModel is a module that contains the struct basis for defining aerosol
    model inputs into the AerosolActivation parametrization. 

mode{T} struct:
    --contains a mode within an aerosol model
    --can contain multiple components in the form of tuple inputs that are
    n_components long

aerosol_model{T} struct:
    --input a tuple of each of your modes into this struct
    --mostly used to count the number of modes in your model

Aerosol parameters:
    --particle density (particles/m^3)
    --osmotic coefficient 
    --molar mass (kg/m)
    --dissociation: number of particles your aerosol dissolves in water
        (e.g., for NaCl, dissoc = 2)
    --mass_frac
    --dry radius (m): radius of dry aerosol particle
    --radius stdev (m)
    --aerosol density (kg/m^3)
    --n_components: number of aerosol species in this mode
"""

module AerosolModel

export mode
export aerosol_model

# individual aerosol mode struct
struct mode{T}
    particle_density::T
    osmotic_coeff::T
    molar_mass::T
    dissoc::T
    mass_frac::T
    mass_mix_ratio::T
    dry_radius::T
    radius_stdev::T
    aerosol_density::T
    n_components::Int64
end

# complete aerosol model struct
struct aerosol_model{T}
    modes::T
    N::Int
    function aerosol_model(modes::T) where {T}
        return new{T}(modes, length(modes)) #modes new{T}
    end
end

end