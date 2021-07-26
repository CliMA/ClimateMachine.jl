module AerosolModel

export mode
export aerosol_model

# individual aerosol mode struct
# TODO:
# the first 3 lines will probably model variables
# the rest are constants depending on aerosol type and
# could be packed into something like chemical composition struct
# TODO - should we name the elements?

"""
The aerosol mode structure stores information about aerosol particles at a certain mode.
The struct takes in real number and tuples as inputs. The real number inputs are the
values that remain constant in the mode, and the tuples are the inputs that vary within 
the mode depending on the material that the aerosol particle is composed of. 

"""

struct mode{T}
    r_dry::Real
    stdev::Real
    N::Real
    mass_mix_ratio::T
    soluble_mass_frac::T
    osmotic_coeff::T
    molar_mass::T
    dissoc::T
    aerosol_density::T
    n_components::Int64
end

"""
The model stores all the modes of aerosol particles. It stores a tuple of all the
modes, and an integer value of the number of modes in the model. 
"""
# complete aerosol model struct
struct aerosol_model{T}
    modes::T
    N::Int
    function aerosol_model(modes::T) where {T}
        return new{T}(modes, length(modes)) #modes new{T}
    end
end

end
