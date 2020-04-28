using ..Atmos
using ..Atmos: thermo_state, turbulence_tensors
using ..Mesh.Topologies
using ..Mesh.Grids
using ..MoistThermodynamics
using LinearAlgebra

function atmos_core_init(dgngrp::DiagnosticsGroup, currtime)
    mpicomm = Settings.mpicomm
    dg = Settings.dg
    Q = Settings.Q
    FT = eltype(Q)
    grid = dg.grid
    topology = grid.topology
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dimensionality(grid) == 2 ? 1 : Nq
    nrealelem = length(topology.realelems)
    nvertelem = topology.stacksize
    nhorzelem = div(nrealelem, nvertelem)

    if Array ∈ typeof(Q).parameters
        localvgeo = grid.vgeo
    else
        localvgeo = Array(grid.vgeo)
    end

    @warn """
        Diagnostics ($dgngrp.name): not yet implemented
        """

    return nothing
end

"""
    atmos_core_collect(bl, currtime)

Perform a global grid traversal to compute various diagnostics.
"""
function atmos_core_collect(dgngrp::DiagnosticsGroup, currtime)
    mpicomm = Settings.mpicomm
    dg = Settings.dg
    Q = Settings.Q
    mpirank = MPI.Comm_rank(mpicomm)
    current_time = string(currtime)

    # extract grid information
    bl = dg.balancelaw
    grid = dg.grid
    topology = grid.topology
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dimensionality(grid) == 2 ? 1 : Nq
    npoints = Nq * Nq * Nqk
    nrealelem = length(topology.realelems)
    nvertelem = topology.stacksize
    nhorzelem = div(nrealelem, nvertelem)

    # get the state, auxiliary and geo variables onto the host if needed
    if Array ∈ typeof(Q).parameters
        localQ = Q.realdata
        localaux = dg.auxstate.realdata
        localvgeo = grid.vgeo
        localdiff = dg.diffstate.realdata
    else
        localQ = Array(Q.realdata)
        localaux = Array(dg.auxstate.realdata)
        localvgeo = Array(grid.vgeo)
        localdiff = Array(dg.diffstate.realdata)
    end
    FT = eltype(localQ)

    nstate = num_state(bl, FT)
    nauxstate = num_aux(bl, FT)
    ndiff = num_diffusive(bl, FT)

    # TODO

    if mpirank == 0
        dprefix = @sprintf(
            "%s_%s-%s-num%04d",
            dgngrp.out_prefix,
            dgngrp.name,
            Settings.starttime,
            dgngrp.num
        )
        dfilename = joinpath(Settings.output_dir, dprefix)
    end

    return nothing
end # function collect

function atmos_core_fini(dgngrp::DiagnosticsGroup, currtime) end
