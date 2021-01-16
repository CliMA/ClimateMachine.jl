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
    The other states in `aux` have been 
    computed in update_auxiliary_state, 
    which can be used to speed up the compuation
"""
prognostic_to_primitive!(atmos::AtmosModel, prim::Vars, prog::Vars, aux::Vars) =
    prognostic_to_primitive!(atmos, atmos.moisture, prim, prog, aux)

"""
    primitive_to_prognostic!(atmos::AtmosModel, prog::Vars, prim::Vars, aux::Vars)

Convert primitive variables `prim` to prognostic
variables `prog` for the atmos model `atmos`.

!!! note
    The only field in `aux` required for this
    method is the geo-potential, which is in `aux`
    The other states in `aux`, which are required for
    flux computations are filled
"""
primitive_to_prognostic!(atmos::AtmosModel, prog::Vars, prim::Vars, aux::Vars) =
    primitive_to_prognostic!(
        atmos,
        atmos.moisture,
        prog,
        prim,
        aux,
        gravitational_potential(atmos.orientation, aux),
    )

"""
    construct_face_auxiliary_state!(bl::AtmosModel, aux_face::AbstractArray, aux_cell::AbstractArray, Δz::FT)

 - `bl` balance law
 - `aux_face` face auxiliary variables to be constructed 
 - `aux_cell` cell center auxiliary variable
 - `Δz` cell vertical size 
"""
function construct_face_auxiliary_state!(
    bl::AtmosModel,
    aux_face::AbstractArray,
    aux_cell::AbstractArray,
    Δz::FT,
) where {FT}
    _grav = grav(bl.param_set)
    var_aux = Vars{vars_state(bl, Auxiliary(), FT)}
    aux_face .= aux_cell

    if bl.orientation != NoOrientation()
        var_aux(aux_face).orientation.Φ =
            var_aux(aux_cell).orientation.Φ + _grav * Δz / 2
    end
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
    ts = recover_thermo_state(atmos, prog, aux)
    prim.ρ = prog.ρ
    prim.u = prog.ρu ./ prog.ρ
    prim.p = air_pressure(ts)
end

function prognostic_to_primitive!(
    atmos,
    moist::EquilMoist,
    prim::Vars,
    prog::Vars,
    aux::Vars,
)
    FT = eltype(prim)
    ts = recover_thermo_state(atmos, prog, aux)

    prim.ρ = prog.ρ
    prim.u = prog.ρu ./ prog.ρ
    prim.p = air_pressure(ts)
    prim.moisture.q_tot = PhasePartition(ts).tot
end

function prognostic_to_primitive!(
    atmos,
    moist::NonEquilMoist,
    prim::Vars,
    prog::Vars,
    aux::Vars,
)
    ts = recover_thermo_state(atmos, prog, aux)

    prim.ρ = prog.ρ
    prim.u = prog.ρu ./ prog.ρ
    prim.p = air_pressure(ts)
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
    e_pot::AbstractFloat,
)
    ts = PhaseDry_ρp(atmos.param_set, prim.ρ, prim.p)
    e_kin = prim.u' * prim.u / 2

    prog.ρ = prim.ρ
    prog.ρu = prim.ρ .* prim.u
    prog.ρe = prim.ρ * total_energy(e_kin, e_pot, ts)
end

function primitive_to_prognostic!(
    atmos,
    moist::EquilMoist,
    prog::Vars,
    prim::Vars,
    aux::Vars,
    e_pot::AbstractFloat,
)
    ts = PhaseEquil_ρpq(
        atmos.param_set,
        prim.ρ,
        prim.p,
        prim.moisture.q_tot,
        true,
    )
    e_kin = prim.u' * prim.u / 2

    prog.ρ = prim.ρ
    prog.ρu = prim.ρ .* prim.u
    prog.ρe = prim.ρ * total_energy(e_kin, e_pot, ts)
    prog.moisture.ρq_tot = prim.ρ * PhasePartition(ts).tot

    aux.moisture.temperature = air_temperature(ts)
end

function primitive_to_prognostic!(
    atmos,
    moist::NonEquilMoist,
    prog::Vars,
    prim::Vars,
    aux::Vars,
    e_pot::AbstractFloat,
)
    q_pt = PhasePartition(
        prim.moisture.q_tot,
        prim.moisture.q_liq,
        prim.moisture.q_ice,
    )
    ts = PhaseNonEquil_ρpq(atmos.param_set, prim.ρ, prim.p, q_pt)
    e_kin = prim.u' * prim.u / 2

    prog.ρ = prim.ρ
    prog.ρu = prim.ρ .* prim.u
    prog.ρe = prim.ρ * total_energy(e_kin, e_pot, ts)
    prog.moisture.ρq_tot = prim.ρ * PhasePartition(ts).tot
    prog.moisture.ρq_liq = prim.ρ * PhasePartition(ts).liq
    prog.moisture.ρq_ice = prim.ρ * PhasePartition(ts).ice

    aux.moisture.temperature = air_temperature(ts)
end
