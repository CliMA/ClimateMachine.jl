"""


"""

module AerosolModel

export mode
export aerosol_model

# individual aerosol mode struct
# TODO:
# the first 3 lines will probably model variables
# the rest are constants depending on aerosol type and
# could be packed into something like chemical composition struct
# TODO - should we name the elements?
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

# complete aerosol model struct
struct aerosol_model{T}
    modes::T
    N::Int
    function aerosol_model(modes::T) where {T}
        return new{T}(modes, length(modes)) #modes new{T}
    end
end

end
