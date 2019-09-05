abstract type RadiationModel
end


vars_state(::RadiationModel, T) = @vars()
vars_gradient(::RadiationModel, T) = @vars()
vars_diffusive(::RadiationModel, T) = @vars()
vars_aux(::RadiationModel, T) = @vars()
vars_integrals(::RadiationModel, T) = @vars()

function flux!(::RadiationModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
end
function preodefun!(::RadiationModel, aux::Vars, state::Vars, t::Real)
end
function integrate_aux!(::RadiationModel, integ::Vars, state::Vars, aux::Vars)
end

struct NoRadiation <: RadiationModel
end

struct StevensRadiation <: RadiationModel
end
vars_aux(m::StevensRadiation) = @vars(z::T, zero_to_z::T, z_to_inf::T)

function flux!(m::StevensRadiation, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
    T = eltype(flux)

    z_i = T(840)  # Start with constant inversion height of 840 meters then build in check based on q_tot
    Δz_i = max(aux.z - z_i, zero(T))

    # Constants
    α_z = T(1)
    ρ_i = T(1.22)
    D_subsidence = T(3.75e-6)
    cloud_top_cooling = T(70) * exp(-aux.z_to_inf)
    cloud_base_warming = T(22) * exp(-aux.zero_to_z)
    free_troposphere_cooling = ρ_i * T(cp_d) * D_subsidence * α_z * ((cbrt(Δz_i))^4 / 4 + z_i * cbrt(Δz_i))
    F_rad = cloud_base_warming + cloud_base_warming + free_troposphere_cooling

    flux.ρe -= SVector(T(0), T(0), state.ρ * F_rad)
end
function preodefun!(m::StevensRadiation, aux::Vars, state::Vars, t::Real)
end
