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
prognostic_to_primitive!(atmos::AtmosModel, prim::Vars, prog::Vars, aux) =
    prognostic_to_primitive!(
        atmos,
        atmos.moisture,
        prim,
        prog,
        Thermodynamics.internal_energy(
            prog.ρ,
            prog.ρe,
            prog.ρu,
            gravitational_potential(atmos.orientation, aux),
        ),
    )

"""
    primitive_to_prognostic!(atmos::AtmosModel, prog::Vars, prim::Vars, aux::Vars)

Convert primitive variables `prim` to prognostic
variables `prog` for the atmos model `atmos`.

!!! note
    The only field in `aux` required for this
    method is the geo-potential.
"""
primitive_to_prognostic!(atmos::AtmosModel, prog::Vars, prim::Vars, aux) =
    primitive_to_prognostic!(
        atmos,
        atmos.moisture,
        prog,
        prim,
        gravitational_potential(atmos.orientation, aux),
    )

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
end

function primitive_to_prognostic!(
    atmos,
    moist::NonEquilMoist,
    prog::Vars,
    prim::Vars,
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
end
