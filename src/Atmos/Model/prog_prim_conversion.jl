####
#### prognostic_to_primitive! and primitive_to_prognostic!
####

####
#### Wrappers (entry point)
####

"""
    prognostic_to_primitive!(atmos::AtmosModel, prim::Vars, prog::Vars, aux::Vars)

Convert prognostic variables `prog` to primitive
variables `prim` for the atmos model `atmos`.

!!! note
    The only field in `aux` required for this
    method is the geo-potential.
"""
function prognostic_to_primitive!(
    atmos::AtmosModel,
    prim::Vars,
    prog::Vars,
    aux,
)
    atmos.energy isa TotalEnergyModel ||
        error("TotalEnergyModel only supported")
    prognostic_to_primitive!(atmos, atmos.moisture, prim, prog, aux)
    prognostic_to_primitive!(
        turbconv_model(atmos),
        atmos,
        atmos.moisture,
        prim,
        prog,
    )
end

"""
    primitive_to_prognostic!(atmos::AtmosModel, prog::Vars, prim::Vars, aux::Vars)

Convert primitive variables `prim` to prognostic
variables `prog` for the atmos model `atmos`.

!!! note
    The only field in `aux` required for this
    method is the geo-potential.
"""
function primitive_to_prognostic!(
    atmos::AtmosModel,
    prog::Vars,
    prim::Vars,
    aux,
)
    primitive_to_prognostic!(atmos, atmos.moisture, prog, prim, aux)
    primitive_to_prognostic!(
        turbconv_model(atmos),
        atmos,
        atmos.moisture,
        prog,
        prim,
    )
end

####
#### prognostic to primitive
####

function prognostic_to_primitive!(
    atmos,
    moist::DryModel,
    prim::Vars,
    prog::Vars,
    aux::Vars,
)
    ts = new_thermo_state(atmos, prog, aux)
    prim.ρ = air_density(ts) # Needed for recovery of energy, not prog.ρ in anelastic1d
    prim.u = prog.ρu ./ density(atmos, prog, aux)
    prim.p = pressure(atmos, ts, aux)
end

function prognostic_to_primitive!(
    atmos,
    moist::EquilMoist,
    prim::Vars,
    prog::Vars,
    aux::Vars,
)
    ts = new_thermo_state(atmos, prog, aux)
    prim.ρ = air_density(ts) # Needed for recovery of energy, not prog.ρ in anelastic1d
    prim.u = prog.ρu ./ density(atmos, prog, aux)
    prim.p = pressure(atmos, ts, aux)
    prim.moisture.q_tot = PhasePartition(ts).tot
end

function prognostic_to_primitive!(
    atmos,
    moist::NonEquilMoist,
    prim::Vars,
    prog::Vars,
    aux::Vars,
)
    ts = new_thermo_state(atmos, prog, aux)
    prim.ρ = air_density(ts) # Needed for recovery of energy, not prog.ρ in anelastic1d
    prim.u = prog.ρu ./ density(atmos, prog, aux)
    prim.p = pressure(atmos, ts, aux)
    prim.moisture.q_tot = PhasePartition(ts).tot
    prim.moisture.q_liq = PhasePartition(ts).liq
    prim.moisture.q_ice = PhasePartition(ts).ice
end

####
#### primitive to prognostic
####

function primitive_to_prognostic!(
    atmos,
    moist::DryModel,
    prog::Vars,
    prim::Vars,
    aux::Vars,
)
    atmos.energy isa TotalEnergyModel ||
        error("TotalEnergyModel only supported")
    ts = PhaseDry_ρp(parameter_set(atmos), prim.ρ, prim.p)
    e_kin = prim.u' * prim.u / 2
    ρ = density(atmos, prim, aux)
    e_pot = gravitational_potential(atmos.orientation, aux)

    prog.ρ = ρ
    prog.ρu = ρ .* prim.u
    prog.energy.ρe = ρ * total_energy(e_kin, e_pot, ts)
end

function primitive_to_prognostic!(
    atmos,
    moist::EquilMoist,
    prog::Vars,
    prim::Vars,
    aux::Vars,
)
    atmos.energy isa TotalEnergyModel ||
        error("TotalEnergyModel only supported")
    ts = PhaseEquil_ρpq(
        parameter_set(atmos),
        prim.ρ,
        prim.p,
        prim.moisture.q_tot,
        true,
    )
    e_kin = prim.u' * prim.u / 2
    ρ = density(atmos, prim, aux)
    e_pot = gravitational_potential(atmos.orientation, aux)

    prog.ρ = ρ
    prog.ρu = ρ .* prim.u
    prog.energy.ρe = ρ * total_energy(e_kin, e_pot, ts)
    prog.moisture.ρq_tot = ρ * PhasePartition(ts).tot
end

function primitive_to_prognostic!(
    atmos,
    moist::NonEquilMoist,
    prog::Vars,
    prim::Vars,
    aux::Vars,
)
    atmos.energy isa TotalEnergyModel ||
        error("TotalEnergyModel only supported")
    q_pt = PhasePartition(
        prim.moisture.q_tot,
        prim.moisture.q_liq,
        prim.moisture.q_ice,
    )
    ts = PhaseNonEquil_ρpq(parameter_set(atmos), prim.ρ, prim.p, q_pt)
    e_kin = prim.u' * prim.u / 2
    ρ = density(atmos, prim, aux)
    e_pot = gravitational_potential(atmos.orientation, aux)

    prog.ρ = ρ
    prog.ρu = ρ .* prim.u
    prog.energy.ρe = ρ * total_energy(e_kin, e_pot, ts)
    prog.moisture.ρq_tot = ρ * PhasePartition(ts).tot
    prog.moisture.ρq_liq = ρ * PhasePartition(ts).liq
    prog.moisture.ρq_ice = ρ * PhasePartition(ts).ice
end


function construct_face_auxiliary_state!(
    bl::AtmosModel,
    aux_face::AbstractArray,
    aux_cell::AbstractArray,
    Δz::FT,
) where {FT <: Real}
    param_set = parameter_set(bl)
    _grav = FT(grav(param_set))
    var_aux = Vars{vars_state(bl, Auxiliary(), FT)}
    aux_face .= aux_cell

    if !(bl.orientation isa NoOrientation)
        var_aux(aux_face).orientation.Φ =
            var_aux(aux_cell).orientation.Φ + _grav * Δz / 2
    end
end
