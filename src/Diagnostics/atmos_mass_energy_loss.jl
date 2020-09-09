# AtmosMassEnergyLoss
#
# Dump mass and energy loss

using ..Atmos
using ..BalanceLaws
using ..Mesh.Topologies
using ..Mesh.Grids
using ..MPIStateArrays

"""
    setup_atmos_mass_energy_loss(
        ::ClimateMachineConfigType,
        interval::String,
        out_prefix::String,
        writer = NetCDFWriter(),
        interpol = nothing,
    )

Create and return a `DiagnosticsGroup` containing the
"AtmosMassEnergyLoss" diagnostics for Atmos LES and GCM
configurations. All the diagnostics in the group will run at the
specified `interval`, be interpolated to the specified boundaries
and resolution, and written to files prefixed by `out_prefix`
using `writer`.
"""
function setup_atmos_mass_energy_loss(
    ::ClimateMachineConfigType,
    interval::String,
    out_prefix::String,
    writer = NetCDFWriter(),
    interpol = nothing,
)
    @assert isnothing(interpol)

    return DiagnosticsGroup(
        "AtmosMassEnergyLoss",
        Diagnostics.atmos_mass_energy_loss_init,
        Diagnostics.atmos_mass_energy_loss_fini,
        Diagnostics.atmos_mass_energy_loss_collect,
        interval,
        out_prefix,
        writer,
        interpol,
    )
end

# store initial mass and energy
mutable struct AtmosMassEnergyLoss
    Σρ₀::Union{Nothing, AbstractFloat}
    Σρe₀::Union{Nothing, AbstractFloat}

    AtmosMassEnergyLoss() = new(nothing, nothing)
end
const AtmosMassEnergyLoss₀ = AtmosMassEnergyLoss()

function atmos_mass_energy_loss_init(dgngrp, currtime)
    mpicomm = Settings.mpicomm
    mpirank = MPI.Comm_rank(mpicomm)
    dg = Settings.dg
    bl = dg.balance_law
    Q = Settings.Q
    FT = eltype(Q)

    ρ_idx = varsindices(vars_state(bl, Prognostic(), FT), "ρ")
    ρe_idx = varsindices(vars_state(bl, Prognostic(), FT), "ρe")
    Σρ₀ = weightedsum(Q, ρ_idx)
    Σρe₀ = weightedsum(Q, ρe_idx)

    if mpirank == 0
        AtmosMassEnergyLoss₀.Σρ₀ = Σρ₀
        AtmosMassEnergyLoss₀.Σρe₀ = Σρe₀

        # outputs are scalars
        dims = OrderedDict()

        # set up the variables we're going to be writing
        vars = OrderedDict(
            "mass_loss" => ((), FT, Variables["mass_loss"].attrib),
            "energy_loss" => ((), FT, Variables["energy_loss"].attrib),
        )

        # create the output file
        dprefix = @sprintf(
            "%s_%s_%s",
            dgngrp.out_prefix,
            dgngrp.name,
            Settings.starttime,
        )
        dfilename = joinpath(Settings.output_dir, dprefix)
        init_data(dgngrp.writer, dfilename, dims, vars)
    end

    return nothing
end

function atmos_mass_energy_loss_collect(dgngrp, currtime)
    mpicomm = Settings.mpicomm
    mpirank = MPI.Comm_rank(mpicomm)
    dg = Settings.dg
    bl = dg.balance_law
    Q = Settings.Q
    FT = eltype(Q)

    ρ_idx = varsindices(vars_state(bl, Prognostic(), FT), "ρ")
    ρe_idx = varsindices(vars_state(bl, Prognostic(), FT), "ρe")
    Σρ = weightedsum(Q, ρ_idx)
    Σρe = weightedsum(Q, ρe_idx)

    if mpirank == 0
        δρ = (Σρ - AtmosMassEnergyLoss₀.Σρ₀) / AtmosMassEnergyLoss₀.Σρ₀
        δρe = (Σρe - AtmosMassEnergyLoss₀.Σρe₀) / AtmosMassEnergyLoss₀.Σρe₀

        varvals = OrderedDict("mass_loss" => δρ, "energy_loss" => δρe)

        # write output
        append_data(dgngrp.writer, varvals, currtime)
    end

end

function atmos_mass_energy_loss_fini(dgngrp, currtime) end
