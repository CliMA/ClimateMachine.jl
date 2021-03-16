using CUDA

"""
    effect_dss3d!(Q::MPIStateArray,
        grid::DiscontinuousSpectralElementGrid)

This function computes the 3D direct stiffness summation for all variables in the MPIStateArray.

# Fields
 - `Q`: MPIStateArray
 - `grid`: Discontinuous Spectral Element Grid
"""
function effect_dss3d!(Q::MPIStateArray, grid::DiscontinuousSpectralElementGrid)
    DA = arraytype(grid)                                      # device array
    device = arraytype(grid) <: Array ? CPU() : CUDADevice()  # device
    FT = eltype(Q.data)
    #----communication--------------------------
    event = MPIStateArrays.begin_ghost_exchange!(Q)
    event = MPIStateArrays.end_ghost_exchange!(Q, dependencies = event)
    wait(event)
    #----Direct Stiffness Summation-------------
    nvars = size(Q.data, 2)
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

    if device == CPU()
        effect_dss3d_CPU!(
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
    else
        max_threads = 256
        n_items = nvt + nfc + nedg
        tx = max(n_items, max_threads)
        bx = cld(n_items, tx)
        @cuda threads = (tx) blocks = (bx) effect_dss3d_CUDA!(
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
    end
    return nothing
end

function effect_dss3d_CUDA!(
    data::AbstractArray{FT, 3},
    vertmap::AbstractArray{I, 1},
    edgemap::AbstractArray{I, 3},
    facemap::AbstractArray{I, 3},
    vtconn::AbstractArray{I, 1},
    vtconnoff::AbstractArray{I, 1},
    edgconn::AbstractArray{I, 1},
    edgconnoff::AbstractArray{I, 1},
    fcconn::AbstractArray{I, 2},
    nvt::I,
    nedg::I,
    nfc::I,
    Nemax::I,
    Nfmax::I,
) where {I <: Int, FT <: AbstractFloat}

    tx = threadIdx().x        # threaid id
    bx = blockIdx().x         # block id
    bxdim = blockDim().x      # block dimension
    glx = tx + (bx - 1) * bxdim # global id

    vx = (glx ≥ 1 && glx ≤ nvt) ? glx : -1
    ex = (glx > nvt && glx ≤ (nvt + nedg)) ? (glx - nvt) : -1
    fx =
        (glx > (nvt + nedg) && glx ≤ (nvt + nedg + nfc)) ? (glx - nvt - nedg) :
        -1

    nvars = size(data, 2)

    #vertex DSS
    if vx ≠ -1
        sumv = FT(0)
        for ivar in 1:nvars
            st = vtconnoff[vx]
            nlvt = Int((vtconnoff[vx + 1] - vtconnoff[vx]) / 2)
            sumv = -FT(0)
            for j in 1:nlvt
                lvt = vtconn[st + (j - 1) * 2]
                lelem = vtconn[st + (j - 1) * 2 + 1]
                loc = vertmap[lvt]
                sumv += data[loc, ivar, lelem]
            end
            for j in 1:nlvt
                lvt = vtconn[st + (j - 1) * 2]
                lelem = vtconn[st + (j - 1) * 2 + 1]
                loc = vertmap[lvt]
                data[loc, ivar, lelem] = sumv
            end
        end
    end
    # edge DSS
    if ex ≠ -1
        sume = FT(0)
        for ivar in 1:nvars
            st = edgconnoff[ex]
            nledg = Int((edgconnoff[ex + 1] - edgconnoff[ex]) / 3)

            for k in 1:Nemax
                sume = -FT(0)
                for j in 1:nledg
                    ledg = edgconn[st + (j - 1) * 3]
                    lor = edgconn[st + (j - 1) * 3 + 1]
                    lelem = edgconn[st + (j - 1) * 3 + 2]
                    loc = edgemap[k, ledg, lor]
                    if loc ≠ -1
                        sume += data[loc, ivar, lelem]
                    end
                end
                for j in 1:nledg
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
    end
    # face DSS
    if fx ≠ -1
        sumf = FT(0)
        for ivar in 1:nvars
            if fcconn[fx, 1] ≠ -1
                lfc = fcconn[fx, 1]
                lel = fcconn[fx, 2]
                nabrlfc = fcconn[fx, 3]
                nabrlel = fcconn[fx, 4]
                ordr = fcconn[fx, 5]
                ordr = ordr == 3 ? 2 : 1
                for j in 1:Nfmax
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
    end

    return nothing
end

function effect_dss3d_CPU!(
    data::AbstractArray{FT, 3},
    vertmap::AbstractArray{I, 1},
    edgemap::AbstractArray{I, 3},
    facemap::AbstractArray{I, 3},
    vtconn::AbstractArray{I, 1},
    vtconnoff::AbstractArray{I, 1},
    edgconn::AbstractArray{I, 1},
    edgconnoff::AbstractArray{I, 1},
    fcconn::AbstractArray{I, 2},
    nvt::I,
    nedg::I,
    nfc::I,
    Nemax::I,
    Nfmax::I,
) where {I <: Int, FT <: AbstractFloat}

    nvars = size(data, 2)

    # vertex DSS
    sumv = FT(0)
    for ivar in 1:nvars
        for i in 1:nvt
            st = vtconnoff[i]
            nlvt = Int((vtconnoff[i + 1] - vtconnoff[i]) / 2)
            sumv = -FT(0)
            for j in 1:nlvt
                lvt = vtconn[st + (j - 1) * 2]
                lelem = vtconn[st + (j - 1) * 2 + 1]
                loc = vertmap[lvt]
                sumv += data[loc, ivar, lelem]
            end
            for j in 1:nlvt
                lvt = vtconn[st + (j - 1) * 2]
                lelem = vtconn[st + (j - 1) * 2 + 1]
                loc = vertmap[lvt]
                data[loc, ivar, lelem] = sumv
            end
        end
    end
    # edge DSS
    sume = FT(0)
    for ivar in 1:nvars
        for i in 1:nedg
            st = edgconnoff[i]
            nledg = Int((edgconnoff[i + 1] - edgconnoff[i]) / 3)

            for k in 1:Nemax
                sume = -FT(0)
                for j in 1:nledg
                    ledg = edgconn[st + (j - 1) * 3]
                    lor = edgconn[st + (j - 1) * 3 + 1]
                    lelem = edgconn[st + (j - 1) * 3 + 2]
                    loc = edgemap[k, ledg, lor]
                    if loc ≠ -1
                        sume += data[loc, ivar, lelem]
                    end
                end
                for j in 1:nledg
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
    end
    # face DSS
    sumf = FT(0)
    for ivar in 1:nvars
        for i in 1:nfc
            if fcconn[i, 1] ≠ -1
                lfc = fcconn[i, 1]
                lel = fcconn[i, 2]
                nabrlfc = fcconn[i, 3]
                nabrlel = fcconn[i, 4]
                ordr = fcconn[i, 5]
                ordr = ordr == 3 ? 2 : 1
                for j in 1:Nfmax
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
    end
    return nothing
end
