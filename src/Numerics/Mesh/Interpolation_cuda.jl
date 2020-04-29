using CuArrays, CUDAnative

function interpolate_local!(
    intrp_brck::InterpolationBrick{FT},
    sv::CuArray{FT},
    v::CuArray{FT},
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

function interpolate_local!(
    intrp_cs::InterpolationCubedSphere{FT},
    sv::CuArray{FT},
    v::CuArray{FT},
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

function project_cubed_sphere!(
    intrp_cs::InterpolationCubedSphere{FT},
    v::CuArray{FT},
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
    fiv::CuArray{FT, 4},
) where {FT <: AbstractFloat}

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
        v_all = CuArray(v_all)
    else
        v_all = iv
    end

    if pid == 0
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
