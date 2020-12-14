# Helpers to gather and store some information useful to multiple diagnostics
# groups.
#

Base.@kwdef mutable struct AtmosCollectedDiagnostics
    onetime_done::Bool = false
    zvals::Union{Nothing, Array} = nothing
    MH_z::Union{Nothing, Array} = nothing
end
const AtmosCollected = AtmosCollectedDiagnostics()

function atmos_collect_onetime(mpicomm, dg, Q)
    if !AtmosCollected.onetime_done

        FT = eltype(Q)
        grid = dg.grid
        grid_info = basic_grid_info(dg)
        topl_info = basic_topology_info(grid.topology)
        Nqk = grid_info.Nqk
        nvertelem = topl_info.nvertelem
        localvgeo = array_device(Q) isa CPU ? grid.vgeo : Array(grid.vgeo)
        AtmosCollected.zvals = zeros(FT, Nqk * nvertelem)
        AtmosCollected.MH_z = zeros(FT, Nqk * nvertelem)
        @traverse_dg_grid grid_info topl_info begin
            z = localvgeo[ijk, grid.x3id, e]
            MH = localvgeo[ijk, grid.MHid, e]
            AtmosCollected.zvals[evk] = z
            AtmosCollected.MH_z[evk] += MH
        end

        # compute the full number of points on a slab
        MPI.Allreduce!(AtmosCollected.MH_z, +, mpicomm)

        AtmosCollected.onetime_done = true
    end

    return nothing
end
