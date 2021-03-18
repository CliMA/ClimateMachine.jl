Base.@kwdef mutable struct CollectedDiagnostics
    onetime_done::Bool = false
    ΣMH_z::Union{Nothing, Array} = nothing
end
const Collected = CollectedDiagnostics()

function collect_onetime(mpicomm, dg, Q)
    if !Collected.onetime_done
        FT = eltype(Q)
        grid = dg.grid
        grid_info = basic_grid_info(dg)
        topl_info = basic_topology_info(grid.topology)
        topology = grid.topology
        Nq1 = grid_info.Nq[1]
        Nq2 = grid_info.Nq[2]
        Nq3 = grid_info.Nqk
        npoints = prod(grid_info.Nq)
        nrealelem = topl_info.nrealelem
        nvertelem = topl_info.nvertelem
        nhorzelem = topl_info.nhorzrealelem

        vgeo = array_device(Q) isa CPU ? grid.vgeo : Array(grid.vgeo)

        Collected.ΣMH_z = zeros(FT, Nq3, nvertelem)

        for eh in 1:nhorzelem, ev in 1:nvertelem
            e = ev + (eh - 1) * nvertelem
            for k in 1:Nq3, j in 1:Nq2, i in 1:Nq1
                ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
                MH = vgeo[ijk, grid.MHid, e]
                Collected.ΣMH_z[k, ev] += MH
            end
        end

        # compute the full number of points on a slab on rank 0
        MPI.Reduce!(Collected.ΣMH_z, +, 0, mpicomm)

        Collected.onetime_done = true
    end

    return nothing
end
