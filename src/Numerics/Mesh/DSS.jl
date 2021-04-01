module DSS

using ..Grids
using ClimateMachine.MPIStateArrays
using CUDA
using KernelAbstractions
using CUDAKernels
using DocStringExtensions

export dss!

"""
    dss3d(Q::MPIStateArray,
        grid::DiscontinuousSpectralElementGrid)

This function computes the 3D direct stiffness summation for all variables in the MPIStateArray.

# Fields
 - `Q`: MPIStateArray
 - `grid`: Discontinuous Spectral Element Grid
"""
function dss!(
    Q::MPIStateArray,
    grid::DiscontinuousSpectralElementGrid{FT, 3};
    max_threads = 256,
) where {FT}
    DA = arraytype(grid)                                      # device array
    device = arraytype(grid) <: Array ? CPU() : CUDADevice()  # device
    #----communication--------------------------
    event = MPIStateArrays.begin_ghost_exchange!(Q)
    event = MPIStateArrays.end_ghost_exchange!(Q, dependencies = event)
    wait(event)
    #----Direct Stiffness Summation-------------
    vertmap = grid.vertmap
    edgemap = grid.edgemap
    facemap = grid.facemap

    vtconn = grid.topology.vtconn
    fcconn = grid.topology.fcconn
    edgconn = grid.topology.edgconn

    vtconnoff = grid.topology.vtconnoff
    edgconnoff = grid.topology.edgconnoff

    nvt, nfc, nedg =
        length(vtconnoff) - 1, size(fcconn, 1), length(edgconnoff) - 1

    Nq = polynomialorders(grid) .+ 1
    Nqmax = maximum(Nq)
    Nemax = Nqmax - 2
    Nfmax = size(facemap, 1)
    args = (
        Q.data,
        vertmap,
        edgemap,
        facemap,
        vtconn,
        vtconnoff,
        edgconn,
        edgconnoff,
        fcconn,
        nvt,
        nedg,
        nfc,
        Nemax,
        Nfmax,
    )
    if device == CPU()
        dss3d_CPU!(args...)
    else
        n_items = nvt + nfc + nedg
        tx = max(n_items, max_threads)
        bx = cld(n_items, tx)
        @cuda threads = (tx) blocks = (bx) dss3d_CUDA!(args...)
    end
    return nothing
end

function dss3d_CUDA!(
    data,
    vertmap,
    edgemap,
    facemap,
    vtconn,
    vtconnoff,
    edgconn,
    edgconnoff,
    fcconn,
    nvt,
    nedg,
    nfc,
    Nemax,
    Nfmax,
)
    I = eltype(nvt)
    FT = eltype(data)

    tx = threadIdx().x        # threaid id
    bx = blockIdx().x         # block id
    bxdim = blockDim().x      # block dimension
    glx = tx + (bx - 1) * bxdim # global id
    nvars = size(data, 2)

    # A mesh node is either
    # - interior (no dss required)
    # - on an element corner / vertex,
    # - edge (excluding end point element corners / vertices)
    # - face (excluding edges and corners)
    if glx ≤ nvt #vertex DSS
        vx = glx
        for ivar in 1:nvars
            dss_vertex!(vtconn, vtconnoff, vertmap, data, ivar, vx, FT)
        end
    elseif glx > nvt && glx ≤ (nvt + nedg) # edge DSS
        ex = glx - nvt
        for ivar in 1:nvars
            dss_edge!(edgconn, edgconnoff, edgemap, Nemax, data, ivar, ex, FT)
        end
    elseif glx > (nvt + nedg) && glx ≤ (nvt + nedg + nfc) # face DSS
        fx = glx - (nvt + nedg)
        for ivar in 1:nvars
            dss_face!(fcconn, facemap, Nfmax, data, ivar, fx)
        end
    end

    return nothing
end

function dss3d_CPU!(
    data,
    vertmap,
    edgemap,
    facemap,
    vtconn,
    vtconnoff,
    edgconn,
    edgconnoff,
    fcconn,
    nvt,
    nedg,
    nfc,
    Nemax,
    Nfmax,
)
    I = eltype(nvt)
    FT = eltype(data)
    nvars = size(data, 2)

    # A mesh node is either
    # - interior (no dss required)
    # - on an element corner / vertex,
    # - edge (excluding end point element corners / vertices)
    # - face (excluding edges and corners)
    for ivar in 1:nvars
        for vx in 1:nvt # vertex DSS
            dss_vertex!(vtconn, vtconnoff, vertmap, data, ivar, vx, FT)
        end
        for ex in 1:nedg # edge DSS
            dss_edge!(edgconn, edgconnoff, edgemap, Nemax, data, ivar, ex, FT)
        end
        for fx in 1:nfc # face DSS
            dss_face!(fcconn, facemap, Nfmax, data, ivar, fx)
        end
    end
    return nothing
end

"""
    dss_vertex!(
    vtconn,
    vtconnoff,
    vertmap,
    data,
    ivar,
    vx,
    ::Type{FT},
) where {FT}

This function computes the direct stiffness summation for the vertex `vx`.

# Fields
 - `vtconn`: vertex connectivity array
 - `vtconnoff`: offsets for vertex connectivity array
 - `vertmap`: map to vertex degrees of freedom: `vertmap[vx]` contains the 
    degree of freedom located at vertex `vx`.
 - `data`: data field of MPIStateArray
 - `ivar`: variable # in the MPIStateArray
 - `vx`: unique edge number
 - `::Type{FT}`: Floating point type
"""
function dss_vertex!(
    vtconn,
    vtconnoff,
    vertmap,
    data,
    ivar,
    vx,
    ::Type{FT},
) where {FT}
    @inbounds st = vtconnoff[vx]
    @inbounds nlvt = Int((vtconnoff[vx + 1] - vtconnoff[vx]) / 2)
    sumv = -FT(0)
    @inbounds for j in 1:nlvt
        lvt = vtconn[st + (j - 1) * 2]
        lelem = vtconn[st + (j - 1) * 2 + 1]
        loc = vertmap[lvt]
        sumv += data[loc, ivar, lelem]
    end
    @inbounds for j in 1:nlvt
        lvt = vtconn[st + (j - 1) * 2]
        lelem = vtconn[st + (j - 1) * 2 + 1]
        loc = vertmap[lvt]
        data[loc, ivar, lelem] = sumv
    end
end

"""
    dss_edge!(
    edgconn,
    edgconnoff,
    edgemap,
    Nemax,
    data,
    ivar,
    ex,
    ::Type{FT},
) where {FT}

This function computes the direct stiffness summation for 
all degrees of freedom corresponding to edge `ex`. dss_edge!
applies only to interior (non-vertex) edge nodes.

# Fields
 - `edgconn`: edge connectivity array
 - `edgconnoff`: offsets for edge connectivity array
 - `edgemap`: map to edge degrees of freedom: 
    `edgemap[i, edgno, orient]` contains the element node index of 
    the `i`th interior node on edge `edgno`, under orientation `orient`. 
 - `Nemax`: # of relevant degrees of freedom per edge (other dof are marked as -1)
 - `data`: data field of MPIStateArray
 - `ivar`: variable # in the MPIStateArray
 - `ex`: unique edge number
 - `::Type{FT}`: Floating point type
"""
function dss_edge!(
    edgconn,
    edgconnoff,
    edgemap,
    Nemax,
    data,
    ivar,
    ex,
    ::Type{FT},
) where {FT}
    @inbounds st = edgconnoff[ex]
    @inbounds nledg = Int((edgconnoff[ex + 1] - edgconnoff[ex]) / 3)

    @inbounds for k in 1:Nemax
        sume = -FT(0)
        @inbounds for j in 1:nledg
            ledg = edgconn[st + (j - 1) * 3]
            lor = edgconn[st + (j - 1) * 3 + 1]
            lelem = edgconn[st + (j - 1) * 3 + 2]
            loc = edgemap[k, ledg, lor]
            if loc ≠ -1
                sume += data[loc, ivar, lelem]
            end
        end
        @inbounds for j in 1:nledg
            ledg = edgconn[st + (j - 1) * 3]
            lor = edgconn[st + (j - 1) * 3 + 1]
            lelem = edgconn[st + (j - 1) * 3 + 2]
            loc = edgemap[k, ledg, lor]
            if loc ≠ -1
                data[loc, ivar, lelem] = sume
            end
        end
    end
end
"""
    dss_face!(fcconn, facemap, Nfmax, data, ivar, fx)

This function computes the direct stiffness summation for 
all degrees of freedom corresponding to face `fx`. dss_face!
applies only to interior (non-vertex and non-edge) face nodes.

# Fields
 - `fcconn`: face connectivity array
 - `facemap`: map to face degrees of freedom: `facemap[ij, fcno, orient]` 
    contains the element node index of the `ij`th 
    interior node on face `fcno` under orientation `orient`
 - `Nfmax`: # of relevant degrees of freedom per face (other dof are marked as -1)
 - `data`: data field of MPIStateArray
 - `ivar`: variable # in the MPIStateArray
 - `fx`: unique face number
"""
function dss_face!(fcconn, facemap, Nfmax, data, ivar, fx)
    @inbounds lfc = fcconn[fx, 1]
    @inbounds lel = fcconn[fx, 2]
    @inbounds nabrlfc = fcconn[fx, 3]
    @inbounds nabrlel = fcconn[fx, 4]
    @inbounds ordr = fcconn[fx, 5]
    ordr = ordr == 3 ? 2 : 1
    # mesh orientation 3 is a flip along the horizontal edges.
    # This is the only orientation currently support by the mesh generator
    # see vertsortandorder in
    # src/Numerics/Mesh/BrickMesh.jl                
    @inbounds for j in 1:Nfmax
        loc1 = facemap[j, lfc, ordr]
        loc2 = facemap[j, nabrlfc, 1]
        if loc1 ≠ -1 && loc2 ≠ -1
            sumf = data[loc1, ivar, lel] + data[loc2, ivar, nabrlel]
            data[loc1, ivar, lel] = sumf
            data[loc2, ivar, nabrlel] = sumf
        end
    end
end

end
