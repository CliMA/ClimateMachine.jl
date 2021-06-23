"""
    Activation
TODO
"""
module Activation

using Thermodynamics

using CLIMAParameters
using CLIMAParameters.Atmos.Microphysics_0M
const APS = AbstractParameterSet

export activation_cloud_droplet_number

"""
    activation_cloud_droplet_number(param_set::APS, q, T, ρ)
TODO
"""
function activation_cloud_droplet_number(
    param_set::APS,
    q::PhasePartition{FT},
    T::FT,
    ρ::FT,
) where {FT <: Real}

    _τ_precip::FT = τ_precip(param_set)
    _qc_0::FT = qc_0(param_set)

    # TODO
    water_density::FT = 42

    return 42
end

function hygroscopicity_test(osmotic_coefficient,
               temperature,
               aerosol_density,
               aerosol_molecular_weight,
               aerosol_particle_density
               water_density,
               water_particle_density,
               water_molecular_weight)

    updraft_velocity = 3
    R = 8.31446261815324
    avogadro =  6.02214076 × 10^23

    h = (updraft_velocity * osmotic_coefficient 
        * (aerosol_particle_density * 1/avogadro 
        * aerosol_molecular_weight)/(1/water_particle_density
        * 1/1000 * water_molecular_weight) * water_molecular_weight
        * aerosol_particle_density) / (aerosol_molecular_weight
        * water_density)
    return h 
end 

function mean_hygroscopicity(aerosol_mode_number, 
                             aerosol_component, 
                             mass_mixing_ratio,
                             disassociation,
                             phi,
                             epsilon,
                             molecular_weight)
    
    add_top = 0
    add_bottom = 0
    water_molecular_weight = 0.01801528 # kg/mol
    water_density = 1000 # kg/m^3
    
    for j in 1:length(aerosol_component)
        add_top = mass_mixing_ratio[aerosol_mode_number][j] 
                  * disassociation[aerosol_mode_number][j] 
                  * phi[aerosol_mode_number][j]
                  * epsilon[aerosol_mode_number][j]
                  * molecular_weight[aerosol_mode_number][j]
    end
    m_h = water_molecular_weight * (add_top) / (add_bottom * water_density)
    return m_h
end

)
end #module Activation.jl