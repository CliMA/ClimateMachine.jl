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
    atmos.energy isa EnergyModel || error("EnergyModel only supported")
    prognostic_to_primitive!(
        atmos,
        atmos.moisture,
        prim,
        prog,
        Thermodynamics.internal_energy(
            prog.ρ,
            prog.energy.ρe,
            prog.ρu,
            gravitational_potential(atmos.orientation, aux),
        ),
    )
    prognostic_to_primitive!(atmos.turbconv, atmos, atmos.moisture, prim, prog)
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
    primitive_to_prognostic!(
        atmos,
        atmos.moisture,
        prog,
        prim,
        gravitational_potential(atmos.orientation, aux),
    )
    primitive_to_prognostic!(atmos.turbconv, atmos, atmos.moisture, prog, prim)
end

####
#### prognostic to primitive
####

function prognostic_to_primitive!(
    atmos,
    moist::DryModel,
    prim::Vars,
    prog::Vars,
    e_int::AbstractFloat,
)
    ts = PhaseDry(atmos.param_set, e_int, prog.ρ)
    prim.ρ = prog.ρ
    prim.u = prog.ρu ./ prog.ρ
    prim.p = air_pressure(ts)
end

function prognostic_to_primitive!(
    atmos,
    moist::EquilMoist,
    prim::Vars,
    prog::Vars,
    e_int::AbstractFloat,
)
    FT = eltype(prim)
    ts = PhaseEquil(
        atmos.param_set,
        e_int,
        prog.ρ,
        prog.moisture.ρq_tot / prog.ρ,
        # 20,       # can improve test error with better convergence
        # FT(1e-3), # can improve test error with better convergence
    )
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
    e_int::AbstractFloat,
)
    q_pt = PhasePartition(
        prog.moisture.ρq_tot / prog.ρ,
        prog.moisture.ρq_liq / prog.ρ,
        prog.moisture.ρq_ice / prog.ρ,
    )
    ts = PhaseNonEquil(atmos.param_set, e_int, prog.ρ, q_pt)
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
    e_pot::AbstractFloat,
)
    atmos.energy isa EnergyModel || error("EnergyModel only supported")
    ts = PhaseDry_ρp(atmos.param_set, prim.ρ, prim.p)
    e_kin = prim.u' * prim.u / 2

    prog.ρ = prim.ρ
    prog.ρu = prim.ρ .* prim.u
    prog.energy.ρe = prim.ρ * total_energy(e_kin, e_pot, ts)
end

function primitive_to_prognostic!(
    atmos,
    moist::EquilMoist,
    prog::Vars,
    prim::Vars,
    e_pot::AbstractFloat,
)
    atmos.energy isa EnergyModel || error("EnergyModel only supported")
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
    prog.energy.ρe = prim.ρ * total_energy(e_kin, e_pot, ts)
    prog.moisture.ρq_tot = prim.ρ * PhasePartition(ts).tot
end

function primitive_to_prognostic!(
    atmos,
    moist::NonEquilMoist,
    prog::Vars,
    prim::Vars,
    e_pot::AbstractFloat,
)
    atmos.energy isa EnergyModel || error("EnergyModel only supported")
    q_pt = PhasePartition(
        prim.moisture.q_tot,
        prim.moisture.q_liq,
        prim.moisture.q_ice,
    )
    ts = PhaseNonEquil_ρpq(atmos.param_set, prim.ρ, prim.p, q_pt)
    e_kin = prim.u' * prim.u / 2

    prog.ρ = prim.ρ
    prog.ρu = prim.ρ .* prim.u
    prog.energy.ρe = prim.ρ * total_energy(e_kin, e_pot, ts)
    prog.moisture.ρq_tot = prim.ρ * PhasePartition(ts).tot
    prog.moisture.ρq_liq = prim.ρ * PhasePartition(ts).liq
    prog.moisture.ρq_ice = prim.ρ * PhasePartition(ts).ice
end


function construct_face_auxiliary_state!(
    bl::AtmosModel,
    aux_face::AbstractArray,
    aux_cell::AbstractArray,
    Δz::FT,
) where {FT <: Real}
    _grav = FT(grav(bl.param_set))
    var_aux = Vars{vars_state(bl, Auxiliary(), FT)}
    aux_face .= aux_cell

    if !(bl.orientation isa NoOrientation)
        var_aux(aux_face).orientation.Φ =
            var_aux(aux_cell).orientation.Φ + _grav * Δz / 2
    end
end
