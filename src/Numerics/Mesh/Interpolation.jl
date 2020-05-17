module Interpolation
using DocStringExtensions
using ClimateMachine
using MPI
import GaussQuadrature
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Geometry
using ClimateMachine.Mesh.Elements
using LinearAlgebra
using StaticArrays
import KernelAbstractions: CPU, CUDA

using CUDAnative
using CuArrays

export InterpolationBrick,
    accumulate_interpolated_data,
    accumulate_interpolated_data!,
    InterpolationCubedSphere,
    interpolate_local!,
    project_cubed_sphere!,
    InterpolationTopology

abstract type InterpolationTopology end

"""
    InterpolationBrick{
    FT <: AbstractFloat,
    T <: Int,
    FTV <: AbstractVector{FT},
    FTVD <: AbstractVector{FT},
    TVD <: AbstractVector{T},
    FTA2 <: Array{FT, 2},
    UI8AD <: AbstractArray{UInt8, 2},
    UI16VD <: AbstractVector{UInt16},
    I32V <: AbstractVector{Int32},
    } <: InterpolationTopology

This interpolation data structure and the corresponding functions works for a brick, where stretching/compression happens only along the x1, x2 & x3 axis.
Here x1 = X1(ξ1), x2 = X2(ξ2) and x3 = X3(ξ3).

# Fields

$(DocStringExtensions.FIELDS)

# Usage

    InterpolationBrick(grid::DiscontinuousSpectralElementGrid{FT}, xbnd::Array{FT,2}, xres) where FT <: AbstractFloat

This interpolation structure and the corresponding functions works for a brick, where stretching/compression happens only along the x1, x2 & x3 axis.
Here x1 = X1(ξ1), x2 = X2(ξ2) and x3 = X3(ξ3).

# Arguments for the inner constructor
 - `grid`: DiscontinousSpectralElementGrid
 - `xbnd`: Domain boundaries in x1, x2 and x3 directions
 - `xres`: Resolution of the interpolation grid in x1, x2 and x3 directions
"""
struct InterpolationBrick{
    FT <: AbstractFloat,
    T <: Int,
    FTV <: AbstractVector{FT},
    FTVD <: AbstractVector{FT},
    TVD <: AbstractVector{T},
    FTA2 <: Array{FT, 2},
    UI8AD <: AbstractArray{UInt8, 2},
    UI16VD <: AbstractVector{UInt16},
    I32V <: AbstractVector{Int32},
} <: InterpolationTopology
    "Number of elements"
    Nel::T
    "Total number of interpolation points"
    Np::T
    "Total number of interpolation points on local process"
    Npl::T
    "Polynomial order of spectral element approximation"
    poly_order::T
    "Domain bounds in x1, x2 and x3 directions"
    xbnd::FTA2
    "Interpolation grid in x1 direction"
    x1g::FTV
    "Interpolation grid in x2 direction"
    x2g::FTV
    "Interpolation grid in x3 direction"
    x3g::FTV
    "Unique ξ1 coordinates of interpolation points within each spectral element"
    ξ1::FTVD
    "Unique ξ2 coordinates of interpolation points within each spectral element"
    ξ2::FTVD
    "Unique ξ3 coordinates of interpolation points within each spectral element"
    ξ3::FTVD
    "Flags when ξ1/ξ2/ξ3 interpolation point matches with a GLL point"
    flg::UI8AD
    "Normalization factor"
    fac::FTVD
    "x1 interpolation grid index of interpolation points within each element on the local process"
    x1i::UI16VD
    "x2 interpolation grid index of interpolation points within each element on the local process"
    x2i::UI16VD
    "x3 interpolation grid index of interpolation points within each element on the local process"
    x3i::UI16VD
    "Offsets for each element"
    offset::TVD    # offsets for each element for v
    "GLL points"
    m1_r::FTVD
    "GLL weights"
    m1_w::FTVD
    "Barycentric weights"
    wb::FTVD
    # MPI setup for gathering interpolated variable on proc # 0
    "Number of interpolation points on each of the processes"
    Np_all::I32V

    "x1 interpolation grid index of interpolation points within each element on all processes stored only on proc 0"
    x1i_all::UI16VD
    "x2 interpolation grid index of interpolation points within each element on all processes stored only on proc 0"
    x2i_all::UI16VD
    "x3 interpolation grid index of interpolation points within each element on all processes stored only on proc 0"
    x3i_all::UI16VD

    function InterpolationBrick(
        grid::DiscontinuousSpectralElementGrid{FT},
        xbnd::Array{FT, 2},
        x1g::AbstractArray{FT, 1},
        x2g::AbstractArray{FT, 1},
        x3g::AbstractArray{FT, 1},
    ) where {FT <: AbstractFloat}
        mpicomm = grid.topology.mpicomm
        pid = MPI.Comm_rank(mpicomm)
        npr = MPI.Comm_size(mpicomm)

        DA = arraytype(grid)                    # device array
        device = arraytype(grid) <: Array ? CPU() : CUDA()

        poly_order = polynomialorder(grid)
        qm1 = poly_order + 1
        ndim = 3
        toler = 4 * eps(FT) # tolerance

        n1g = length(x1g)
        n2g = length(x2g)
        n3g = length(x3g)

        Np = n1g * n2g * n3g
        marker = BitArray{3}(undef, n1g, n2g, n3g)
        fill!(marker, true)

        Nel = length(grid.topology.realelems) # # of elements on local process
        offset = Vector{Int}(undef, Nel + 1) # offsets for the interpolated variable
        n123 = zeros(Int, ndim)        # # of unique ξ1, ξ2, ξ3 points in each cell
        xsten = zeros(Int, 2, ndim)        # x1, x2, x3 start and end for each brick element
        xbndl = zeros(FT, 2, ndim)        # x1,x2,x3 limits (min,max) for each brick element

        ξ1 = map(i -> zeros(FT, i), zeros(Int, Nel))
        ξ2 = map(i -> zeros(FT, i), zeros(Int, Nel))
        ξ3 = map(i -> zeros(FT, i), zeros(Int, Nel))

        x1i = map(i -> zeros(UInt16, i), zeros(UInt16, Nel))
        x2i = map(i -> zeros(UInt16, i), zeros(UInt16, Nel))
        x3i = map(i -> zeros(UInt16, i), zeros(UInt16, Nel))

        x = map(i -> zeros(FT, ndim, i), zeros(Int, Nel)) # interpolation grid points embedded in each cell

        offset[1] = 0

        for el in 1:Nel
            for (xg, dim) in zip((x1g, x2g, x3g), 1:ndim)
                xbndl[1, dim], xbndl[2, dim] =
                    extrema(grid.topology.elemtocoord[dim, :, el])

                st = findfirst(xg .≥ xbndl[1, dim] .- toler)
                if st ≠ nothing
                    if xg[st] > (xbndl[2, dim] + toler)
                        st = nothing
                    end
                end

                if st ≠ nothing
                    xsten[1, dim] = st
                    xsten[2, dim] =
                        findlast(temp -> temp .≤ xbndl[2, dim] .+ toler, xg)
                    n123[dim] = xsten[2, dim] - xsten[1, dim] + 1
                else
                    n123[dim] = 0
                end
            end

            if prod(n123) > 0
                for k in xsten[1, 3]:xsten[2, 3],
                    j in xsten[1, 2]:xsten[2, 2],
                    i in xsten[1, 1]:xsten[2, 1]

                    if marker[i, j, k]
                        push!(
                            ξ1[el],
                            2 * (x1g[i] - xbndl[1, 1]) /
                            (xbndl[2, 1] - xbndl[1, 1]) - 1,
                        )
                        push!(
                            ξ2[el],
                            2 * (x2g[j] - xbndl[1, 2]) /
                            (xbndl[2, 2] - xbndl[1, 2]) - 1,
                        )
                        push!(
                            ξ3[el],
                            2 * (x3g[k] - xbndl[1, 3]) /
                            (xbndl[2, 3] - xbndl[1, 3]) - 1,
                        )

                        push!(x1i[el], UInt16(i))
                        push!(x2i[el], UInt16(j))
                        push!(x3i[el], UInt16(k))
                        marker[i, j, k] = false
                    end
                end
                offset[el + 1] = offset[el] + length(ξ1[el])
            end

        end # el loop

        m1_r, m1_w = GaussQuadrature.legendre(FT, qm1, GaussQuadrature.both)
        wb = Elements.baryweights(m1_r)

        Npl = offset[end]

        ξ1_d = Array{FT}(undef, Npl)
        ξ2_d = Array{FT}(undef, Npl)
        ξ3_d = Array{FT}(undef, Npl)
        x1i_d = Array{UInt16}(undef, Npl)
        x2i_d = Array{UInt16}(undef, Npl)
        x3i_d = Array{UInt16}(undef, Npl)
        fac_d = zeros(FT, Npl)
        flg_d = zeros(UInt8, 3, Npl)

        for i in 1:Nel
            ctr = 1
            for j in (offset[i] + 1):offset[i + 1]
                ξ1_d[j] = ξ1[i][ctr]
                ξ2_d[j] = ξ2[i][ctr]
                ξ3_d[j] = ξ3[i][ctr]
                x1i_d[j] = x1i[i][ctr]
                x2i_d[j] = x2i[i][ctr]
                x3i_d[j] = x3i[i][ctr]
                # set up interpolation
                fac1 = FT(0)
                fac2 = FT(0)
                fac3 = FT(0)
                for ib in 1:qm1
                    if abs(m1_r[ib] - ξ1_d[j]) < toler
                        @inbounds flg_d[1, j] = UInt8(ib)
                    else
                        @inbounds fac1 += wb[ib] / (ξ1_d[j] - m1_r[ib])
                    end

                    if abs(m1_r[ib] - ξ2_d[j]) < toler
                        @inbounds flg_d[2, j] = UInt8(ib)
                    else
                        @inbounds fac2 += wb[ib] / (ξ2_d[j] - m1_r[ib])
                    end

                    if abs(m1_r[ib] - ξ3_d[j]) < toler
                        @inbounds flg_d[3, j] = UInt8(ib)
                    else
                        @inbounds fac3 += wb[ib] / (ξ3_d[j] - m1_r[ib])
                    end
                end

                flg_d[1, j] ≠ UInt8(0) && (fac1 = FT(1))
                flg_d[2, j] ≠ UInt8(0) && (fac2 = FT(1))
                flg_d[3, j] ≠ UInt8(0) && (fac3 = FT(1))

                fac_d[j] = FT(1) / (fac1 * fac2 * fac3)

                ctr += 1
            end
        end
        # MPI setup for gathering data on proc 0
        root = 0
        Np_all = zeros(Int32, npr)
        Np_all[pid + 1] = Npl

        MPI.Allreduce!(Np_all, +, mpicomm)

        if pid ≠ root
            x1i_all = zeros(UInt16, 0)
            x2i_all = zeros(UInt16, 0)
            x3i_all = zeros(UInt16, 0)
        else
            x1i_all = Array{UInt16}(undef, sum(Np_all))
            x2i_all = Array{UInt16}(undef, sum(Np_all))
            x3i_all = Array{UInt16}(undef, sum(Np_all))
        end

        MPI.Gatherv!(x1i_d, x1i_all, Np_all, root, mpicomm)
        MPI.Gatherv!(x2i_d, x2i_all, Np_all, root, mpicomm)
        MPI.Gatherv!(x3i_d, x3i_all, Np_all, root, mpicomm)

        if device == CUDA()
            ξ1_d = DA(ξ1_d)
            ξ2_d = DA(ξ2_d)
            ξ3_d = DA(ξ3_d)
            x1i_d = DA(x1i_d)
            x2i_d = DA(x2i_d)
            x3i_d = DA(x3i_d)
            flg_d = DA(flg_d)
            fac_d = DA(fac_d)
            offset = DA(offset)
            m1_r = DA(m1_r)
            m1_w = DA(m1_w)
            wb = DA(wb)
            x1i_all = DA(x1i_all)
            x2i_all = DA(x2i_all)
            x3i_all = DA(x3i_all)
        end
        return new{
            FT,
            Int,
            typeof(x1g),
            typeof(ξ1_d),
            typeof(offset),
            typeof(xbnd),
            typeof(flg_d),
            typeof(x1i_d),
            typeof(Np_all),
        }(
            Nel,
            Np,
            Npl,
            poly_order,
            xbnd,
            x1g,
            x2g,
            x3g,
            ξ1_d,
            ξ2_d,
            ξ3_d,
            flg_d,
            fac_d,
            x1i_d,
            x2i_d,
            x3i_d,
            offset,
            m1_r,
            m1_w,
            wb,
            Np_all,
            x1i_all,
            x2i_all,
            x3i_all,
        )

    end

end # struct InterpolationBrick

"""
    interpolate_local!(intrp_brck::InterpolationBrick{FT},
                               sv::AbstractArray{FT},
                                v::AbstractArray{FT}) where {FT <: AbstractFloat}

This interpolation function works for a brick, where stretching/compression happens only along the x1, x2 & x3 axis.
Here x1 = X1(ξ1), x2 = X2(ξ2) and x3 = X3(ξ3)

# Arguments
 - `intrp_brck`: Initialized InterpolationBrick structure
 - `sv`: State Array consisting of various variables on the discontinuous Galerkin grid
 - `v`:  Interpolated variables
"""
function interpolate_local!(
    intrp_brck::InterpolationBrick{FT},
    sv::AbstractArray{FT},
    v::AbstractArray{FT},
) where {FT <: AbstractFloat}

    offset = intrp_brck.offset
    m1_r = intrp_brck.m1_r
    wb = intrp_brck.wb
    ξ1 = intrp_brck.ξ1
    ξ2 = intrp_brck.ξ2
    ξ3 = intrp_brck.ξ3
    flg = intrp_brck.flg
    fac = intrp_brck.fac

    qm1 = length(m1_r)
    Nel = length(offset) - 1
    nvars = size(sv, 2)

    device = typeof(sv) <: Array ? CPU() : CUDA()

    if device == CPU()

        Nel = length(offset) - 1

        vout = zeros(FT, nvars)
        vout_ii = zeros(FT, nvars)
        vout_ij = zeros(FT, nvars)

        for el in 1:Nel # for each element elno
            np = offset[el + 1] - offset[el]
            off = offset[el]

            for i in 1:np # interpolate point-by-point
                ξ1l = ξ1[off + i]
                ξ2l = ξ2[off + i]
                ξ3l = ξ3[off + i]

                f1 = flg[1, off + i]
                f2 = flg[2, off + i]
                f3 = flg[3, off + i]
                fc = fac[off + i]

                vout .= 0.0
                f3 == 0 ? (ikloop = 1:qm1) : (ikloop = f3:f3)

                for ik in ikloop
                    vout_ij .= 0.0
                    f2 == 0 ? (ijloop = 1:qm1) : (ijloop = f2:f2)
                    for ij in ijloop #1:qm1

                        vout_ii .= 0.0

                        if f1 == 0
                            for ii in 1:qm1
                                for vari in 1:nvars
                                    @inbounds vout_ii[vari] +=
                                        sv[
                                            ii + (ij - 1) * qm1 + (ik - 1) *
                                                                  qm1 *
                                                                  qm1,
                                            vari,
                                            el,
                                        ] * wb[ii] / (ξ1l - m1_r[ii])#phir[ii]
                                end
                            end
                        else
                            for vari in 1:nvars
                                @inbounds vout_ii[vari] = sv[
                                    f1 + (ij - 1) * qm1 + (ik - 1) * qm1 * qm1,
                                    vari,
                                    el,
                                ]
                            end
                        end
                        if f2 == 0
                            for vari in 1:nvars
                                @inbounds vout_ij[vari] +=
                                    vout_ii[vari] * wb[ij] / (ξ2l - m1_r[ij])#phis[ij]
                            end
                        else
                            for vari in 1:nvars
                                @inbounds vout_ij[vari] = vout_ii[vari]
                            end
                        end

                    end
                    if f3 == 0
                        for vari in 1:nvars
                            @inbounds vout[vari] +=
                                vout_ij[vari] * wb[ik] / (ξ3l - m1_r[ik])#phit[ik]
                        end
                    else
                        for vari in 1:nvars
                            @inbounds vout[vari] = vout_ij[vari]
                        end
                    end

                end
                for vari in 1:nvars
                    @inbounds v[off + i, vari] = vout[vari] * fc
                end
            end
        end

    else
        @cuda threads = (qm1, qm1) blocks = (Nel, nvars) shmem =
            qm1 * (qm1 + 2) * sizeof(FT) interpolate_brick_CUDA!(
            offset,
            m1_r,
            wb,
            ξ1,
            ξ2,
            ξ3,
            flg,
            fac,
            sv,
            v,
        )
    end
end

function interpolate_brick_CUDA!(
    offset::AbstractArray{T, 1},
    m1_r::AbstractArray{FT, 1},
    wb::AbstractArray{FT, 1},
    ξ1::AbstractArray{FT, 1},
    ξ2::AbstractArray{FT, 1},
    ξ3::AbstractArray{FT, 1},
    flg::AbstractArray{UInt8, 2},
    fac::AbstractArray{FT, 1},
    sv::AbstractArray{FT},
    v::AbstractArray{FT},
) where {T <: Int, FT <: AbstractFloat}

    tj = threadIdx().x
    tk = threadIdx().y # thread ids
    el = blockIdx().x                       # assigning one element per block
    st_idx = blockIdx().y
    qm1 = length(m1_r)
    # create views for shared memory
    shm_FT = @cuDynamicSharedMem(FT, (qm1, qm1 + 2))

    vout_jk = view(shm_FT, :, 1:qm1)
    wb_sh = view(shm_FT, :, qm1 + 1)
    m1_r_sh = view(shm_FT, :, qm1 + 2)
    # load shared memory
    if tk == 1
        wb_sh[tj] = wb[tj]
        m1_r_sh[tj] = m1_r[tj]
    end
    sync_threads()

    np = offset[el + 1] - offset[el]
    off = offset[el]

    for i in 1:np # interpolate point-by-point

        ξ1l = ξ1[off + i]
        ξ2l = ξ2[off + i]
        ξ3l = ξ3[off + i]

        f1 = flg[1, off + i]
        f2 = flg[2, off + i]
        f3 = flg[3, off + i]
        fc = fac[off + i]


        if f1 == 0 # apply phir
            @inbounds vout_jk[tj, tk] =
                sv[1 + (tj - 1) * qm1 + (tk - 1) * qm1 * qm1, st_idx, el] *
                wb_sh[1] / (ξ1l - m1_r_sh[1])
            for ii in 2:qm1
                @inbounds vout_jk[tj, tk] +=
                    sv[ii + (tj - 1) * qm1 + (tk - 1) * qm1 * qm1, st_idx, el] *
                    wb_sh[ii] / (ξ1l - m1_r_sh[ii])
            end
        else
            @inbounds vout_jk[tj, tk] =
                sv[f1 + (tj - 1) * qm1 + (tk - 1) * qm1 * qm1, st_idx, el]
        end

        if f2 == 0 # apply phis
            @inbounds vout_jk[tj, tk] *= (wb_sh[tj] / (ξ2l - m1_r_sh[tj]))
        end
        sync_threads()

        if tj == 1 # reduction
            if f2 == 0
                for ij in 2:qm1
                    @inbounds vout_jk[1, tk] += vout_jk[ij, tk]
                end
            else
                if f2 ≠ 1
                    @inbounds vout_jk[1, tk] = vout_jk[f2, tk]
                end
            end

            if f3 == 0 # apply phit
                @inbounds vout_jk[1, tk] *= (wb_sh[tk] / (ξ3l - m1_r_sh[tk]))
            end
        end
        sync_threads()

        if tj == 1 && tk == 1 # reduction
            if f3 == 0
                for ik in 2:qm1
                    @inbounds vout_jk[1, 1] += vout_jk[1, ik]
                end
            else
                if f3 ≠ 1
                    @inbounds vout_jk[1, 1] = vout_jk[1, f3]
                end
            end
            @inbounds v[off + i, st_idx] = vout_jk[1, 1] * fc
        end

    end

    return nothing
end

"""
    InterpolationCubedSphere{
    FT <: AbstractFloat,
    T <: Int,
    FTV <: AbstractVector{FT},
    FTVD <: AbstractVector{FT},
    TVD <: AbstractVector{T},
    UI8AD <: AbstractArray{UInt8, 2},
    UI16VD <: AbstractVector{UInt16},
    I32V <: AbstractVector{Int32},
    } <: InterpolationTopology

This interpolation structure and the corresponding functions works for a cubed sphere topology. The data is interpolated along a lat/long/rad grid.

-90⁰  ≤ lat  ≤ 90⁰

-180⁰ ≤ long ≤ 180⁰

Rᵢ ≤ r ≤ Rₒ

# Fields

$(DocStringExtensions.FIELDS)

# Usage

    InterpolationCubedSphere(grid::DiscontinuousSpectralElementGrid, vert_range::AbstractArray{FT}, nhor::Int, lat_res::FT, long_res::FT, rad_res::FT) where {FT <: AbstractFloat}

This interpolation structure and the corresponding functions works for a cubed sphere topology. The data is interpolated along a lat/long/rad grid.

-90⁰  ≤ lat  ≤ 90⁰

-180⁰ ≤ long ≤ 180⁰

Rᵢ ≤ r ≤ Rₒ

# Arguments for the inner constructor
 - `grid`: DiscontinousSpectralElementGrid
 - `vert_range`: Vertex range along the radial coordinate
 - `lat_res`: Resolution of the interpolation grid along the latitude coordinate in radians
 - `long_res`: Resolution of the interpolation grid along the longitude coordinate in radians
 - `rad_res`: Resolution of the interpolation grid along the radial coordinate
"""
struct InterpolationCubedSphere{
    FT <: AbstractFloat,
    T <: Int,
    FTV <: AbstractVector{FT},
    FTVD <: AbstractVector{FT},
    TVD <: AbstractVector{T},
    UI8AD <: AbstractArray{UInt8, 2},
    UI16VD <: AbstractVector{UInt16},
    I32V <: AbstractVector{Int32},
} <: InterpolationTopology
    "Number of elements"
    Nel::T
    "Number of interpolation points"
    Np::T
    "Number of interpolation points on local process"
    Npl::T            # # of interpolation points on the local process
    "Polynomial order of spectral element approximation"
    poly_order::T
    "Number of interpolation points in radial direction"
    n_rad::T
    "Number of interpolation points in lat direction"
    n_lat::T
    "Number of interpolation points in long direction"
    n_long::T
    "Interpolation grid in radial direction"
    rad_grd::FTV
    "Interpolation grid in lat direction"
    lat_grd::FTV
    "Interpolation grid in long direction"
    long_grd::FTV # rad, lat & long locations of interpolation grid
    "Device array containing ξ1 coordinates of interpolation points within each element"
    ξ1::FTVD
    "Device array containing ξ2 coordinates of interpolation points within each element"
    ξ2::FTVD
    "Device array containing ξ3 coordinates of interpolation points within each element"
    ξ3::FTVD
    "flags when ξ1/ξ2/ξ3 interpolation point matches with a GLL point"
    flg::UI8AD
    "Normalization factor"
    fac::FTVD
    "Radial coordinates of interpolation points withing each element"
    radi::UI16VD
    "Latitude coordinates of interpolation points withing each element"
    lati::UI16VD
    "Longitude coordinates of interpolation points withing each element"
    longi::UI16VD
    "Offsets for each element"
    offset::TVD
    "GLL points"
    m1_r::FTVD
    "GLL weights"
    m1_w::FTVD
    "Barycentric weights"
    wb::FTVD
    # MPI setup for gathering interpolated variable on proc 0
    "Number of interpolation points on each of the processes"
    Np_all::I32V
    "Radial interpolation grid index of interpolation points within each element on all processes stored only on proc 0"
    radi_all::UI16VD
    "Latitude interpolation grid index of interpolation points within each element on all processes stored only on proc 0"
    lati_all::UI16VD
    "Longitude interpolation grid index of interpolation points within each element on all processes stored only on proc 0"
    longi_all::UI16VD

    function InterpolationCubedSphere(
        grid::DiscontinuousSpectralElementGrid,
        vert_range::AbstractArray{FT},
        nhor::Int,
        lat_grd::AbstractArray{FT, 1},
        long_grd::AbstractArray{FT, 1},
        rad_grd::AbstractArray{FT},
    ) where {FT <: AbstractFloat}
        mpicomm = MPI.COMM_WORLD
        pid = MPI.Comm_rank(mpicomm)
        npr = MPI.Comm_size(mpicomm)

        DA = arraytype(grid)                    # device array
        device = arraytype(grid) <: Array ? CPU() : CUDA()
        poly_order = polynomialorder(grid)
        qm1 = poly_order + 1
        toler1 = FT(eps(FT) * vert_range[1] * 2.0) # tolerance for unwarp function
        toler2 = FT(eps(FT) * 4.0)                 # tolerance
        toler3 = FT(eps(FT) * vert_range[1] * 10.0) # tolerance for Newton-Raphson

        Nel = length(grid.topology.realelems) # # of local elements on the local process

        nvert_range = length(vert_range)
        nvert = nvert_range - 1              # # of elements in vertical direction
        Nel_glob = nvert * nhor * nhor * 6

        nblck = nhor * nhor * nvert
        Δh = 2 / nhor                               # horizontal grid spacing in unwarped grid

        n_lat, n_long, n_rad =
            Int(length(lat_grd)), Int(length(long_grd)), Int(length(rad_grd))

        Np = n_lat * n_long * n_rad

        uw_grd = zeros(FT, 3)
        diffv = zeros(FT, 3)
        ξ = zeros(FT, 3)

        glob_ord = grid.topology.origsendorder # to account for reordering of elements after the partitioning process

        glob_elem_no = zeros(Int, nvert * length(glob_ord))

        for i in 1:length(glob_ord), j in 1:nvert
            glob_elem_no[j + (i - 1) * nvert] = (glob_ord[i] - 1) * nvert + j
        end
        glob_to_loc = Dict(glob_elem_no[i] => Int(i) for i in 1:Nel) # using dictionary for speedup

        ξ1, ξ2, ξ3 = map(i -> zeros(FT, i), zeros(Int, Nel)),
        map(i -> zeros(FT, i), zeros(Int, Nel)),
        map(i -> zeros(FT, i), zeros(Int, Nel))

        radi, lati, longi = map(i -> zeros(UInt16, i), zeros(UInt16, Nel)),
        map(i -> zeros(UInt16, i), zeros(UInt16, Nel)),
        map(i -> zeros(UInt16, i), zeros(UInt16, Nel))


        offset_d = zeros(Int, Nel + 1)

        for i in 1:n_rad
            rad = rad_grd[i]
            if rad ≤ vert_range[1]       # accounting for minor rounding errors from unwarp function at boundaries
                vert_range[1] - rad < toler1 ? l_nrm = 1 :
                error(
                    "fatal error, rad lower than inner radius: ",
                    vert_range[1] - rad,
                    " $x1_grd /// $x2_grd //// $x3_grd",
                )
            elseif rad ≥ vert_range[end] # accounting for minor rounding errors from unwarp function at boundaries
                rad - vert_range[end] < toler1 ? l_nrm = nvert :
                error("fatal error, rad greater than outer radius")
            else                         # normal scenario
                for l in 2:nvert_range
                    if vert_range[l] - rad > FT(0)
                        l_nrm = l - 1
                        break
                    end
                end
            end

            for j in 1:n_lat
                @inbounds x3_grd = rad * sind(lat_grd[j])
                for k in 1:n_long
                    @inbounds x1_grd =
                        rad * cosd(lat_grd[j]) * cosd(long_grd[k]) # inclination -> latitude; azimuthal -> longitude.
                    @inbounds x2_grd =
                        rad * cosd(lat_grd[j]) * sind(long_grd[k]) # inclination -> latitude; azimuthal -> longitude.

                    uw_grd[1], uw_grd[2], uw_grd[3] =
                        Topologies.cubedshellunwarp(x1_grd, x2_grd, x3_grd) # unwarping from sphere to cubed shell

                    x1_uw2_grd = uw_grd[1] / rad # unwrapping cubed shell on to a 2D grid (in 3D space, -1 to 1 cube)
                    x2_uw2_grd = uw_grd[2] / rad
                    x3_uw2_grd = uw_grd[3] / rad

                    if abs(x1_uw2_grd + 1) < toler2 # face 1 (x1 == -1 plane)
                        l2 = min(div(x2_uw2_grd + 1, Δh) + 1, nhor)
                        l3 = min(div(x3_uw2_grd + 1, Δh) + 1, nhor)
                        el_glob = Int(
                            l_nrm +
                            (nhor - l2) * nvert +
                            (l3 - 1) * nvert * nhor,
                        )
                    elseif abs(x2_uw2_grd + 1) < toler2 # face 2 (x2 == -1 plane)
                        l1 = min(div(x1_uw2_grd + 1, Δh) + 1, nhor)
                        l3 = min(div(x3_uw2_grd + 1, Δh) + 1, nhor)
                        el_glob = Int(
                            l_nrm +
                            (l1 - 1) * nvert +
                            (l3 - 1) * nvert * nhor +
                            nblck * 1,
                        )
                    elseif abs(x1_uw2_grd - 1) < toler2 # face 3 (x1 == +1 plane)
                        l2 = min(div(x2_uw2_grd + 1, Δh) + 1, nhor)
                        l3 = min(div(x3_uw2_grd + 1, Δh) + 1, nhor)
                        el_glob = Int(
                            l_nrm +
                            (l2 - 1) * nvert +
                            (l3 - 1) * nvert * nhor +
                            nblck * 2,
                        )
                    elseif abs(x3_uw2_grd - 1) < toler2 # face 4 (x3 == +1 plane)
                        l1 = min(div(x1_uw2_grd + 1, Δh) + 1, nhor)
                        l2 = min(div(x2_uw2_grd + 1, Δh) + 1, nhor)
                        el_glob = Int(
                            l_nrm +
                            (l1 - 1) * nvert +
                            (l2 - 1) * nvert * nhor +
                            nblck * 3,
                        )
                    elseif abs(x2_uw2_grd - 1) < toler2 # face 5 (x2 == +1 plane)
                        l1 = min(div(x1_uw2_grd + 1, Δh) + 1, nhor)
                        l3 = min(div(x3_uw2_grd + 1, Δh) + 1, nhor)
                        el_glob = Int(
                            l_nrm +
                            (l1 - 1) * nvert +
                            (nhor - l3) * nvert * nhor +
                            nblck * 4,
                        )
                    elseif abs(x3_uw2_grd + 1) < toler2 # face 6 (x3 == -1 plane)
                        l1 = min(div(x1_uw2_grd + 1, Δh) + 1, nhor)
                        l2 = min(div(x2_uw2_grd + 1, Δh) + 1, nhor)
                        el_glob = Int(
                            l_nrm +
                            (l1 - 1) * nvert +
                            (nhor - l2) * nvert * nhor +
                            nblck * 5,
                        )
                    else
                        error("error: unwrapped grid does not lie on any of the 6 faces")
                    end

                    el_loc = get(glob_to_loc, el_glob, nothing)
                    if el_loc ≠ nothing # computing inner coordinates for local elements
                        invert_trilear_mapping_hex!(
                            view(grid.topology.elemtocoord, 1, :, el_loc),
                            view(grid.topology.elemtocoord, 2, :, el_loc),
                            view(grid.topology.elemtocoord, 3, :, el_loc),
                            uw_grd,
                            diffv,
                            toler3,
                            ξ,
                        )
                        push!(ξ1[el_loc], ξ[1])
                        push!(ξ2[el_loc], ξ[2])
                        push!(ξ3[el_loc], ξ[3])
                        push!(radi[el_loc], UInt16(i))
                        push!(lati[el_loc], UInt16(j))
                        push!(longi[el_loc], UInt16(k))
                        offset_d[el_loc + 1] += 1
                    end

                end
            end
        end

        for i in 2:(Nel + 1)
            @inbounds offset_d[i] += offset_d[i - 1]
        end

        Npl = offset_d[Nel + 1]

        v = Vector{FT}(undef, offset_d[Nel + 1]) # Allocating storage for interpolation variable

        ξ1_d = Vector{FT}(undef, Npl)
        ξ2_d = Vector{FT}(undef, Npl)
        ξ3_d = Vector{FT}(undef, Npl)

        flg_d = zeros(UInt8, 3, Npl)
        fac_d = ones(FT, Npl)

        rad_d = Vector{UInt16}(undef, Npl)
        lat_d = Vector{UInt16}(undef, Npl)
        long_d = Vector{UInt16}(undef, Npl)

        m1_r, m1_w = GaussQuadrature.legendre(FT, qm1, GaussQuadrature.both)
        wb = Elements.baryweights(m1_r)
        for i in 1:Nel
            ctr = 1
            for j in (offset_d[i] + 1):offset_d[i + 1]
                @inbounds ξ1_d[j] = ξ1[i][ctr]
                @inbounds ξ2_d[j] = ξ2[i][ctr]
                @inbounds ξ3_d[j] = ξ3[i][ctr]
                @inbounds rad_d[j] = radi[i][ctr]
                @inbounds lat_d[j] = lati[i][ctr]
                @inbounds long_d[j] = longi[i][ctr]
                # set up interpolation
                fac1 = FT(0)
                fac2 = FT(0)
                fac3 = FT(0)
                for ib in 1:qm1
                    if abs(m1_r[ib] - ξ1_d[j]) < toler2
                        @inbounds flg_d[1, j] = UInt8(ib)
                    else
                        @inbounds fac1 += wb[ib] / (ξ1_d[j] - m1_r[ib])
                    end

                    if abs(m1_r[ib] - ξ2_d[j]) < toler2
                        @inbounds flg_d[2, j] = UInt8(ib)
                    else
                        @inbounds fac2 += wb[ib] / (ξ2_d[j] - m1_r[ib])
                    end

                    if abs(m1_r[ib] - ξ3_d[j]) < toler2
                        @inbounds flg_d[3, j] = UInt8(ib)
                    else
                        @inbounds fac3 += wb[ib] / (ξ3_d[j] - m1_r[ib])
                    end
                end

                flg_d[1, j] ≠ 0 && (fac1 = FT(1))
                flg_d[2, j] ≠ 0 && (fac2 = FT(1))
                flg_d[3, j] ≠ 0 && (fac3 = FT(1))

                fac_d[j] = FT(1) / (fac1 * fac2 * fac3)

                ctr += 1
            end
        end
        # MPI setup for gathering data on proc 0
        root = 0
        Np_all = zeros(Int32, npr)
        Np_all[pid + 1] = Int32(Npl)

        MPI.Allreduce!(Np_all, +, mpicomm)

        if pid ≠ root
            radi_all = zeros(UInt16, 0)
            lati_all = zeros(UInt16, 0)
            longi_all = zeros(UInt16, 0)
        else
            radi_all = Array{UInt16}(undef, sum(Np_all))
            lati_all = Array{UInt16}(undef, sum(Np_all))
            longi_all = Array{UInt16}(undef, sum(Np_all))
        end

        MPI.Gatherv!(rad_d, radi_all, Np_all, root, mpicomm)
        MPI.Gatherv!(lat_d, lati_all, Np_all, root, mpicomm)
        MPI.Gatherv!(long_d, longi_all, Np_all, root, mpicomm)

        if device == CUDA()
            ξ1_d = DA(ξ1_d)
            ξ2_d = DA(ξ2_d)
            ξ3_d = DA(ξ3_d)

            flg_d = DA(flg_d)
            fac_d = DA(fac_d)

            rad_d = DA(rad_d)
            lat_d = DA(lat_d)
            long_d = DA(long_d)

            m1_r = DA(m1_r)
            m1_w = DA(m1_w)
            wb = DA(wb)

            offset_d = DA(offset_d)

            rad_grd = DA(rad_grd)
            lat_grd = DA(lat_grd)
            long_grd = DA(long_grd)

            radi_all = DA(radi_all)
            lati_all = DA(lati_all)
            longi_all = DA(longi_all)
        end

        return new{
            FT,
            Int,
            typeof(rad_grd),
            typeof(ξ1_d),
            typeof(offset_d),
            typeof(flg_d),
            typeof(rad_d),
            typeof(Np_all),
        }(
            Nel,
            Np,
            Npl,
            poly_order,
            n_rad,
            n_lat,
            n_long,
            rad_grd,
            lat_grd,
            long_grd,
            ξ1_d,
            ξ2_d,
            ξ3_d,
            flg_d,
            fac_d,
            rad_d,
            lat_d,
            long_d,
            offset_d,
            m1_r,
            m1_w,
            wb,
            Np_all,
            radi_all,
            lati_all,
            longi_all,
        )

    end # Inner constructor InterpolationCubedSphere

end # struct InterpolationCubedSphere

"""
    invert_trilear_mapping_hex!(X1::AbstractArray{FT,1},
                                X2::AbstractArray{FT,1},
                                X3::AbstractArray{FT,1},
                                 x::AbstractArray{FT,1},
                                 d::AbstractArray{FT,1},
                               tol::FT,
                                 ξ::AbstractArray{FT,1}) where FT <: AbstractFloat

This function computes ξ = (ξ1,ξ2,ξ3) given x = (x1,x2,x3) and the (8) vertex coordinates of a Hexahedron. Newton-Raphson method is used.

# Arguments
 - `X1`: X1 coordinates of the (8) vertices of the hexahedron
 - `X2`: X2 coordinates of the (8) vertices of the hexahedron
 - `X3`: X3 coordinates of the (8) vertices of the hexahedron
 - `x`: (x1,x2,x3) coordinates of the point
 - `d`: (x1,x2,x3) coordinates, temporary storage
 - `tol`: Tolerance for convergence
 - `ξ`: (ξ1,ξ2,ξ3) coordinates of the point
"""
function invert_trilear_mapping_hex!(
    X1::AbstractArray{FT, 1},
    X2::AbstractArray{FT, 1},
    X3::AbstractArray{FT, 1},
    x::AbstractArray{FT, 1},
    d::AbstractArray{FT, 1},
    tol::FT,
    ξ::AbstractArray{FT, 1},
) where {FT <: AbstractFloat}
    max_it = 10     # maximum # of iterations
    ξ .= FT(0) #zeros(FT,3,1) # initial guess => cell centroid
    trilinear_map_minus_x!(ξ, X1, X2, X3, x, d)
    err = sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3])
    ctr = 0
    # Newton-Raphson iterations
    while err > tol
        trilinear_map_IJac_x_vec!(ξ, X1, X2, X3, d)
        ξ .-= d
        trilinear_map_minus_x!(ξ, X1, X2, X3, x, d)
        err = sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3]) #norm(d)
        ctr += 1
        if ctr > max_it
            error(
                "invert_trilinear_mapping_hex: Newton-Raphson not converging to desired tolerance after max_it = ",
                max_it,
                " iterations; err = ",
                err,
                "; toler = ",
                tol,
            )
        end
    end

    clamp!(ξ, FT(-1), FT(1))
    return nothing
end

function trilinear_map_minus_x!(
    ξ::AbstractArray{FT, 1},
    x1v::AbstractArray{FT, 1},
    x2v::AbstractArray{FT, 1},
    x3v::AbstractArray{FT, 1},
    x::AbstractArray{FT, 1},
    d::AbstractArray{FT, 1},
) where {FT <: AbstractFloat}
    p1 = 1 + ξ[1]
    p2 = 1 + ξ[2]
    p3 = 1 + ξ[3]
    m1 = 1 - ξ[1]
    m2 = 1 - ξ[2]
    m3 = 1 - ξ[3]


    d[1] =
        (
            m1 * (
                m2 * (m3 * x1v[1] + p3 * x1v[5]) +
                p2 * (m3 * x1v[3] + p3 * x1v[7])
            ) +
            p1 * (
                m2 * (m3 * x1v[2] + p3 * x1v[6]) +
                p2 * (m3 * x1v[4] + p3 * x1v[8])
            )
        ) / 8.0 - x[1]

    d[2] =
        (
            m1 * (
                m2 * (m3 * x2v[1] + p3 * x2v[5]) +
                p2 * (m3 * x2v[3] + p3 * x2v[7])
            ) +
            p1 * (
                m2 * (m3 * x2v[2] + p3 * x2v[6]) +
                p2 * (m3 * x2v[4] + p3 * x2v[8])
            )
        ) / 8.0 - x[2]

    d[3] =
        (
            m1 * (
                m2 * (m3 * x3v[1] + p3 * x3v[5]) +
                p2 * (m3 * x3v[3] + p3 * x3v[7])
            ) +
            p1 * (
                m2 * (m3 * x3v[2] + p3 * x3v[6]) +
                p2 * (m3 * x3v[4] + p3 * x3v[8])
            )
        ) / 8.0 - x[3]

    return nothing
end

function trilinear_map_IJac_x_vec!(
    ξ::AbstractArray{FT, 1},
    x1v::AbstractArray{FT, 1},
    x2v::AbstractArray{FT, 1},
    x3v::AbstractArray{FT, 1},
    v::AbstractArray{FT, 1},
) where {FT <: AbstractFloat}
    p1 = 1 + ξ[1]
    p2 = 1 + ξ[2]
    p3 = 1 + ξ[3]
    m1 = 1 - ξ[1]
    m2 = 1 - ξ[2]
    m3 = 1 - ξ[3]

    Jac11 =
        (
            m2 * (m3 * (x1v[2] - x1v[1]) + p3 * (x1v[6] - x1v[5])) +
            p2 * (m3 * (x1v[4] - x1v[3]) + p3 * (x1v[8] - x1v[7]))
        ) / 8.0

    Jac12 =
        (
            m1 * (m3 * (x1v[3] - x1v[1]) + p3 * (x1v[7] - x1v[5])) +
            p1 * (m3 * (x1v[4] - x1v[2]) + p3 * (x1v[8] - x1v[6]))
        ) / 8.0

    Jac13 =
        (
            m1 * (m2 * (x1v[5] - x1v[1]) + p2 * (x1v[7] - x1v[3])) +
            p1 * (m2 * (x1v[6] - x1v[2]) + p2 * (x1v[8] - x1v[4]))
        ) / 8.0

    Jac21 =
        (
            m2 * (m3 * (x2v[2] - x2v[1]) + p3 * (x2v[6] - x2v[5])) +
            p2 * (m3 * (x2v[4] - x2v[3]) + p3 * (x2v[8] - x2v[7]))
        ) / 8.0

    Jac22 =
        (
            m1 * (m3 * (x2v[3] - x2v[1]) + p3 * (x2v[7] - x2v[5])) +
            p1 * (m3 * (x2v[4] - x2v[2]) + p3 * (x2v[8] - x2v[6]))
        ) / 8.0

    Jac23 =
        (
            m1 * (m2 * (x2v[5] - x2v[1]) + p2 * (x2v[7] - x2v[3])) +
            p1 * (m2 * (x2v[6] - x2v[2]) + p2 * (x2v[8] - x2v[4]))
        ) / 8.0

    Jac31 =
        (
            m2 * (m3 * (x3v[2] - x3v[1]) + p3 * (x3v[6] - x3v[5])) +
            p2 * (m3 * (x3v[4] - x3v[3]) + p3 * (x3v[8] - x3v[7]))
        ) / 8.0

    Jac32 =
        (
            m1 * (m3 * (x3v[3] - x3v[1]) + p3 * (x3v[7] - x3v[5])) +
            p1 * (m3 * (x3v[4] - x3v[2]) + p3 * (x3v[8] - x3v[6]))
        ) / 8.0

    Jac33 =
        (
            m1 * (m2 * (x3v[5] - x3v[1]) + p2 * (x3v[7] - x3v[3])) +
            p1 * (m2 * (x3v[6] - x3v[2]) + p2 * (x3v[8] - x3v[4]))
        ) / 8.0

    # computing cofactor matrix
    C11 = Jac22 * Jac33 - Jac23 * Jac32
    C12 = -Jac21 * Jac33 + Jac23 * Jac31
    C13 = Jac21 * Jac32 - Jac22 * Jac31
    C21 = -Jac12 * Jac33 + Jac13 * Jac32
    C22 = Jac11 * Jac33 - Jac13 * Jac31
    C23 = -Jac11 * Jac32 + Jac12 * Jac31
    C31 = Jac12 * Jac23 - Jac13 * Jac22
    C32 = -Jac11 * Jac23 + Jac13 * Jac21
    C33 = Jac11 * Jac22 - Jac12 * Jac21

    # computing determinant
    det = Jac11 * C11 + Jac12 * C12 + Jac13 * C13

    Jac11 = (C11 * v[1] + C21 * v[2] + C31 * v[3]) / det
    Jac21 = (C12 * v[1] + C22 * v[2] + C32 * v[3]) / det
    Jac31 = (C13 * v[1] + C23 * v[2] + C33 * v[3]) / det

    v[1] = Jac11
    v[2] = Jac21
    v[3] = Jac31

    return nothing
end

"""
    interpolate_local!(intrp_cs::InterpolationCubedSphere{FT},
                             sv::AbstractArray{FT},
                              v::AbstractArray{FT}) where {FT <: AbstractFloat}

This interpolation function works for cubed spherical shell geometry.

# Arguments
 - `intrp_cs`: Initialized cubed sphere structure
 - `sv`: Array consisting of various variables on the discontinuous Galerkin grid
 - `v`:  Array consisting of variables on the interpolated grid
"""
function interpolate_local!(
    intrp_cs::InterpolationCubedSphere{FT},
    sv::AbstractArray{FT},
    v::AbstractArray{FT},
) where {FT <: AbstractFloat}

    offset = intrp_cs.offset
    m1_r = intrp_cs.m1_r
    wb = intrp_cs.wb
    ξ1 = intrp_cs.ξ1
    ξ2 = intrp_cs.ξ2
    ξ3 = intrp_cs.ξ3
    flg = intrp_cs.flg
    fac = intrp_cs.fac
    lati = intrp_cs.lati
    longi = intrp_cs.longi
    lat_grd = intrp_cs.lat_grd
    long_grd = intrp_cs.long_grd

    qm1 = length(m1_r)
    nvars = size(sv, 2)
    Nel = length(offset) - 1
    np_tot = size(v, 1)
    _ρu, _ρv, _ρw = 2, 3, 4

    device = typeof(sv) <: Array ? CPU() : CUDA()

    if device == CPU()

        Nel = length(offset) - 1

        vout = zeros(FT, nvars) #FT(0)
        vout_ii = zeros(FT, nvars) #FT(0)
        vout_ij = zeros(FT, nvars) #FT(0)


        for el in 1:Nel # for each element elno
            np = offset[el + 1] - offset[el]
            off = offset[el]

            for i in 1:np # interpolating point-by-point
                ξ1l = ξ1[off + i]
                ξ2l = ξ2[off + i]
                ξ3l = ξ3[off + i]

                f1 = flg[1, off + i]
                f2 = flg[2, off + i]
                f3 = flg[3, off + i]
                fc = fac[off + i]

                vout .= 0.0
                f3 == 0 ? (ikloop = 1:qm1) : (ikloop = f3:f3)

                for ik in ikloop
                    vout_ij .= 0.0
                    f2 == 0 ? (ijloop = 1:qm1) : (ijloop = f2:f2)
                    for ij in ijloop #1:qm1

                        vout_ii .= 0.0

                        if f1 == 0
                            for ii in 1:qm1
                                for vari in 1:nvars
                                    @inbounds vout_ii[vari] +=
                                        sv[
                                            ii + (ij - 1) * qm1 + (ik - 1) *
                                                                  qm1 *
                                                                  qm1,
                                            vari,
                                            el,
                                        ] * wb[ii] / (ξ1l - m1_r[ii])#phir[ii]
                                end
                            end
                        else
                            for vari in 1:nvars
                                @inbounds vout_ii[vari] = sv[
                                    f1 + (ij - 1) * qm1 + (ik - 1) * qm1 * qm1,
                                    vari,
                                    el,
                                ]
                            end
                        end
                        if f2 == 0
                            for vari in 1:nvars
                                @inbounds vout_ij[vari] +=
                                    vout_ii[vari] * wb[ij] / (ξ2l - m1_r[ij])#phis[ij]
                            end
                        else
                            for vari in 1:nvars
                                @inbounds vout_ij[vari] = vout_ii[vari]
                            end
                        end

                    end
                    if f3 == 0
                        for vari in 1:nvars
                            @inbounds vout[vari] +=
                                vout_ij[vari] * wb[ik] / (ξ3l - m1_r[ik])#phit[ik]
                        end
                    else
                        for vari in 1:nvars
                            @inbounds vout[vari] = vout_ij[vari]
                        end
                    end

                end
                for vari in 1:nvars
                    @inbounds v[off + i, vari] = vout[vari] * fc
                end
            end
        end

    else

        @cuda threads = (qm1, qm1) blocks = (Nel, nvars) shmem =
            qm1 * (qm1 + 2) * sizeof(FT) interpolate_cubed_sphere_CUDA!(
            offset,
            m1_r,
            wb,
            ξ1,
            ξ2,
            ξ3,
            flg,
            fac,
            sv,
            v,
        )

    end
    return nothing
end

function interpolate_cubed_sphere_CUDA!(
    offset::AbstractArray{T, 1},
    m1_r::AbstractArray{FT, 1},
    wb::AbstractArray{FT, 1},
    ξ1::AbstractArray{FT, 1},
    ξ2::AbstractArray{FT, 1},
    ξ3::AbstractArray{FT, 1},
    flg::AbstractArray{UInt8, 2},
    fac::AbstractArray{FT, 1},
    sv::AbstractArray{FT},
    v::AbstractArray{FT},
) where {T <: Integer, FT <: AbstractFloat}

    tj = threadIdx().x
    tk = threadIdx().y # thread ids
    el = blockIdx().x                       # assigning one element per block
    st_no = blockIdx().y

    qm1 = length(m1_r)
    nvars = size(sv, 2)
    _ρu, _ρv, _ρw = 2, 3, 4
    # creating views for shared memory
    shm_FT = @cuDynamicSharedMem(FT, (qm1, qm1 + 2))

    vout_jk = view(shm_FT, :, 1:qm1)
    wb_sh = view(shm_FT, :, qm1 + 1)
    m1_r_sh = view(shm_FT, :, qm1 + 2)
    # load shared memory
    if tk == 1
        wb_sh[tj] = wb[tj]
        m1_r_sh[tj] = m1_r[tj]
    end
    sync_threads()

    np = offset[el + 1] - offset[el]
    off = offset[el]

    for i in 1:np # interpolating point-by-point

        ξ1l = ξ1[off + i]
        ξ2l = ξ2[off + i]
        ξ3l = ξ3[off + i]

        f1 = flg[1, off + i]
        f2 = flg[2, off + i]
        f3 = flg[3, off + i]
        fc = fac[off + i]


        if f1 == 0 # applying phir
            @inbounds vout_jk[tj, tk] =
                sv[1 + (tj - 1) * qm1 + (tk - 1) * qm1 * qm1, st_no, el] *
                wb_sh[1] / (ξ1l - m1_r_sh[1])
            for ii in 2:qm1
                @inbounds vout_jk[tj, tk] +=
                    sv[ii + (tj - 1) * qm1 + (tk - 1) * qm1 * qm1, st_no, el] *
                    wb_sh[ii] / (ξ1l - m1_r_sh[ii])
            end
        else
            @inbounds vout_jk[tj, tk] =
                sv[f1 + (tj - 1) * qm1 + (tk - 1) * qm1 * qm1, st_no, el]
        end

        if f2 == 0 # applying phis
            @inbounds vout_jk[tj, tk] *= (wb_sh[tj] / (ξ2l - m1_r_sh[tj]))
        end
        sync_threads()

        if tj == 1 # reduction
            if f2 == 0
                for ij in 2:qm1
                    @inbounds vout_jk[1, tk] += vout_jk[ij, tk]
                end
            else
                if f2 ≠ 1
                    @inbounds vout_jk[1, tk] = vout_jk[f2, tk]
                end
            end

            if f3 == 0 # applying phit
                @inbounds vout_jk[1, tk] *= (wb_sh[tk] / (ξ3l - m1_r_sh[tk]))
            end
        end
        sync_threads()

        if tj == 1 && tk == 1 # reduction
            if f3 == 0
                for ik in 2:qm1
                    @inbounds vout_jk[1, 1] += vout_jk[1, ik]
                end
            else
                if f3 ≠ 1
                    @inbounds vout_jk[1, 1] = vout_jk[1, f3]
                end
            end
            @inbounds v[off + i, st_no] = vout_jk[1, 1] * fc


        end

    end

    return nothing
end

"""
    project_cubed_sphere!(intrp_cs::InterpolationCubedSphere{FT},
                                 v::AbstractArray{FT},
                              uvwi::Tuple{Int,Int,Int}) where {FT <: AbstractFloat}

This function projects the velocity field along unit vectors in radial, lat and long directions for cubed spherical shell geometry.

# Fields
 - `intrp_cs`: Initialized cubed sphere structure
 - `v`: Array consisting of x1, x2 and x3 components of the vector field
 - `uvwi`:  Tuple providing the column numbers for x1, x2 and x3 components of vector field in the array. 
            These columns will be replaced with projected vector fields along unit vectors in rad, lat and long directions.
"""
function project_cubed_sphere!(
    intrp_cs::InterpolationCubedSphere{FT},
    v::AbstractArray{FT},
    uvwi::Tuple{Int, Int, Int},
) where {FT <: AbstractFloat}
    # projecting velocity onto unit vectors in rad, lat and long directions
    # assumes u, v and w are located in columns specified in vector uvwi
    @assert length(uvwi) == 3 "length(uvwi) is not 3"
    lati = intrp_cs.lati
    longi = intrp_cs.longi
    lat_grd = intrp_cs.lat_grd
    long_grd = intrp_cs.long_grd
    _ρu = uvwi[1]
    _ρv = uvwi[2]
    _ρw = uvwi[3]
    np_tot = size(v, 1)

    device = typeof(v) <: Array ? CPU() : CUDA()
    if device == CPU()
        for i in 1:np_tot
            @inbounds vrad =
                v[i, _ρu] * cosd(lat_grd[lati[i]]) * cosd(long_grd[longi[i]]) +
                v[i, _ρv] * cosd(lat_grd[lati[i]]) * sind(long_grd[longi[i]]) +
                v[i, _ρw] * sind(lat_grd[lati[i]])

            @inbounds vlat =
                -v[i, _ρu] * sind(lat_grd[lati[i]]) * cosd(long_grd[longi[i]])
            -v[i, _ρv] * sind(lat_grd[lati[i]]) * sind(long_grd[longi[i]]) +
            v[i, _ρw] * cosd(lat_grd[lati[i]])

            @inbounds vlon =
                -v[i, _ρu] * sind(long_grd[longi[i]]) +
                v[i, _ρv] * cosd(long_grd[longi[i]])

            @inbounds v[i, _ρu] = vrad
            @inbounds v[i, _ρv] = vlat
            @inbounds v[i, _ρw] = vlon
        end
    elseif device == CUDA()
        n_threads = 256
        n_blocks = (
            np_tot % n_threads > 0 ? div(np_tot, n_threads) + 1 :
                div(np_tot, n_threads)
        )
        @cuda threads = (n_threads,) blocks = (n_blocks,) project_cubed_sphere_CUDA!(
            lat_grd,
            long_grd,
            lati,
            longi,
            v,
            _ρu,
            _ρv,
            _ρw,
        )
    else
        error("project_cubed_sphere!: unsupported device, only CPU() and CUDA() supported")
    end
end

function project_cubed_sphere_CUDA!(
    lat_grd::AbstractArray{FT, 1},
    long_grd::AbstractArray{FT, 1},
    lati::AbstractVector{UInt16},
    longi::AbstractVector{UInt16},
    v::AbstractArray{FT},
    _ρu::Int,
    _ρv::Int,
    _ρw::Int,
) where {FT <: AbstractFloat}

    ti = threadIdx().x # thread ids
    bi = blockIdx().x  # block ids
    bs = blockDim().x  # block dim
    idx = ti + (bi - 1) * bs
    np_tot = size(v, 1)
    # projecting velocity onto unit vectors in rad, lat and long directions
    # assumed u, v and w are located in columns 2, 3 and 4
    if idx ≤ np_tot
        vrad =
            v[idx, _ρu] *
            CUDAnative.cos(lat_grd[lati[idx]] * pi / 180.0) *
            CUDAnative.cos(long_grd[longi[idx]] * pi / 180.0) +
            v[idx, _ρv] *
            CUDAnative.cos(lat_grd[lati[idx]] * pi / 180.0) *
            CUDAnative.sin(long_grd[longi[idx]] * pi / 180.0) +
            v[idx, _ρw] * CUDAnative.sin(lat_grd[lati[idx]] * pi / 180.0)

        vlat =
            -v[idx, _ρu] *
            CUDAnative.sin(lat_grd[lati[idx]] * pi / 180.0) *
            CUDAnative.cos(long_grd[longi[idx]] * pi / 180.0)
        -v[idx, _ρv] *
        CUDAnative.sin(lat_grd[lati[idx]] * pi / 180.0) *
        CUDAnative.sin(long_grd[longi[idx]] * pi / 180.0) +
        v[idx, _ρw] * CUDAnative.cos(lat_grd[lati[idx]] * pi / 180.0)

        vlon =
            -v[idx, _ρu] * CUDAnative.sin(long_grd[longi[idx]] * pi / 180.0) +
            v[idx, _ρv] * CUDAnative.cos(long_grd[longi[idx]] * pi / 180.0)

        v[idx, _ρu] = vrad
        v[idx, _ρv] = vlat
        v[idx, _ρw] = vlon
    end # TODO: cosd / sind having issues on GPU. Unable to isolate the issue at this point. Needs to be revisited.
    return nothing
end

"""
    accumulate_interpolated_data!(intrp::InterpolationTopology,
                                     iv::AbstractArray{FT,2},
                                    fiv::AbstractArray{FT,4}) where {FT <: AbstractFloat}

This interpolation function gathers interpolated data onto process # 0.

# Fields
 - `intrp`: Initialized interpolation topology structure
 - `iv`: Interpolated variables on local process
 - `fiv`: Full interpolated variables accumulated on process # 0
"""
function accumulate_interpolated_data!(
    intrp::InterpolationTopology,
    iv::AbstractArray{FT, 2},
    fiv::AbstractArray{FT, 4},
) where {FT <: AbstractFloat}

    DA = ClimateMachine.array_type()           # device array
    device = DA <: Array ? CPU() : CUDA()
    mpicomm = MPI.COMM_WORLD
    pid = MPI.Comm_rank(mpicomm)
    npr = MPI.Comm_size(mpicomm)
    root = 0
    nvars = size(iv, 2)

    if intrp isa InterpolationCubedSphere
        nx1 = length(intrp.rad_grd)
        nx2 = length(intrp.lat_grd)
        nx3 = length(intrp.long_grd)
        np_tot = length(intrp.radi_all)
        i1 = intrp.radi_all
        i2 = intrp.lati_all
        i3 = intrp.longi_all
    elseif intrp isa InterpolationBrick
        nx1 = length(intrp.x1g)
        nx2 = length(intrp.x2g)
        nx3 = length(intrp.x3g)
        np_tot = length(intrp.x1i_all)
        i1 = intrp.x1i_all
        i2 = intrp.x2i_all
        i3 = intrp.x3i_all
    else
        error("Unsupported topology; only InterpolationCubedSphere and InterpolationBrick supported")
    end

    if pid == 0 && size(fiv) ≠ (nx1, nx2, nx3, nvars)
        error("size of fiv = $(size(fiv)); which does not match with ($nx1,$nx2,$nx3,$nvars) ")
    end

    if npr > 1
        Np_all = intrp.Np_all
        pid == 0 ? v_all = Array{FT}(undef, np_tot, nvars) :
        v_all = Array{FT}(undef, 0, nvars)
        if device == CPU()

            for vari in 1:nvars
                MPI.Gatherv!(
                    view(iv, :, vari),
                    view(v_all, :, vari),
                    Np_all,
                    root,
                    mpicomm,
                )
            end

        elseif device == CUDA()

            v = Array(iv)
            for vari in 1:nvars
                MPI.Gatherv!(
                    view(v, :, vari),
                    view(v_all, :, vari),
                    Np_all,
                    root,
                    mpicomm,
                )
            end
            v_all = DA(v_all)

        else
            error("accumulate_interpolate_data: unsupported device, only CPU() and CUDA() supported")
        end
    else
        v_all = iv
    end

    if pid == 0
        if device == CPU()
            for i in 1:np_tot
                for vari in 1:nvars
                    @inbounds fiv[i1[i], i2[i], i3[i], vari] = v_all[i, vari]
                end
            end
        elseif device == CUDA()
            n_threads = 256
            n_blocks = (
                np_tot % n_threads > 0 ? div(np_tot, n_threads) + 1 :
                    div(np_tot, n_threads)
            )
            @cuda threads = (n_threads,) blocks = (n_blocks,) accumulate_helper_CUDA!(
                i1,
                i2,
                i3,
                v_all,
                fiv,
            )
        else
            error("Unsupported device $device; only CUDA() and CPU() supported")
        end

    end
    MPI.Barrier(mpicomm)
    return nothing
end

function accumulate_helper_CUDA!(
    i1::AbstractArray{UInt16, 1},
    i2::AbstractArray{UInt16, 1},
    i3::AbstractArray{UInt16, 1},
    v_all::AbstractArray{FT, 2},
    fiv::AbstractArray{FT, 4},
) where {FT <: AbstractFloat}
    ti = threadIdx().x # thread ids
    bi = blockIdx().x  # block ids
    bs = blockDim().x  # block dim
    idx = ti + (bi - 1) * bs
    np_tot = size(v_all, 1)
    nvars = size(v_all, 2)

    if idx ≤ np_tot
        for vari in 1:nvars
            @inbounds fiv[i1[idx], i2[idx], i3[idx], vari] = v_all[idx, vari]
        end
    end
end


function accumulate_interpolated_data(
    mpicomm::MPI.Comm,
    intrp::InterpolationTopology,
    iv::AbstractArray{FT, 2},
) where {FT <: AbstractFloat}

    mpirank = MPI.Comm_rank(mpicomm)
    numranks = MPI.Comm_size(mpicomm)
    nvars = size(iv, 2)

    if intrp isa InterpolationCubedSphere
        nx1 = length(intrp.rad_grd)
        nx2 = length(intrp.lat_grd)
        nx3 = length(intrp.long_grd)
        np_tot = length(intrp.radi_all)
        i1 = intrp.radi_all
        i2 = intrp.lati_all
        i3 = intrp.longi_all
    elseif intrp isa InterpolationBrick
        nx1 = length(intrp.x1g)
        nx2 = length(intrp.x2g)
        nx3 = length(intrp.x3g)
        np_tot = length(intrp.x1i_all)
        i1 = intrp.x1i_all
        i2 = intrp.x2i_all
        i3 = intrp.x3i_all
    else
        error("Unsupported topology; only InterpolationCubedSphere and InterpolationBrick supported")
    end

    if Array ∈ typeof(iv).parameters
        h_iv = iv
        h_i1 = i1
        h_i2 = i2
        h_i3 = i3
    else
        h_iv = Array(iv)
        h_i1 = Array(i1)
        h_i2 = Array(i2)
        h_i3 = Array(i3)
    end

    if numranks == 1
        v_all = h_iv
    else
        v_all = Array{FT}(undef, mpirank == 0 ? np_tot : 0, nvars)
        for vari in 1:nvars
            MPI.Gatherv!(
                view(h_iv, :, vari),
                view(v_all, :, vari),
                intrp.Np_all,
                0,
                mpicomm,
            )
        end
    end

    if mpirank == 0
        fiv = Array{FT}(undef, nx1, nx2, nx3, nvars)
        for i in 1:np_tot
            for vari in 1:nvars
                @inbounds fiv[h_i1[i], h_i2[i], h_i3[i], vari] = v_all[i, vari]
            end
        end
    else
        fiv = nothing
    end

    return fiv
end

end # module Interpolation
