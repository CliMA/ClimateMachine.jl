# Helpers to gather and store some information useful to multiple diagnostics
# groups.
#

Base.@kwdef mutable struct AtmosCollectedDiagnostics
    onetime_done::Bool = false
    zvals::Union{Nothing, Array} = nothing
    repdvsr::Union{Nothing, Array} = nothing
end
const AtmosCollected = AtmosCollectedDiagnostics()

function atmos_collect_onetime(mpicomm, dg, Q)
    if !AtmosCollected.onetime_done
        FT = eltype(Q)
        grid = dg.grid
        topology = grid.topology
        N = polynomialorder(grid)
        Nq = N + 1
        Nqk = dimensionality(grid) == 2 ? 1 : Nq
        nrealelem = length(topology.realelems)
        nvertelem = topology.stacksize
        nhorzelem = div(nrealelem, nvertelem)

        localvgeo = array_device(Q) isa CPU ? grid.vgeo : Array(grid.vgeo)

        AtmosCollected.zvals = zeros(FT, Nqk * nvertelem)
        AtmosCollected.repdvsr = zeros(FT, Nqk * nvertelem)

        @visitQ nhorzelem nvertelem Nqk Nq begin
            evk = Nqk * (ev - 1) + k
            z = localvgeo[ijk, grid.x3id, e]
            MH = localvgeo[ijk, grid.MHid, e]
            AtmosCollected.zvals[evk] += MH * z
            AtmosCollected.repdvsr[evk] += MH
        end

        # compute the full number of points on a slab
        MPI.Allreduce!(AtmosCollected.repdvsr, +, mpicomm)

        AtmosCollected.zvals ./= AtmosCollected.repdvsr
        AtmosCollected.onetime_done = true
    end

    return nothing
end
