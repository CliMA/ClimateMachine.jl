# This kernel computes the fast tensor-product x vector
# vout = (phit ⊗ phis ⊗ phir) * vin
# using Julia native multi-threading

function ftpxv_hex_CPU!(
    vin::AbstractArray{FT, 2},
    vin_den::Union{AbstractArray{FT, 2}, Nothing},
    vout::AbstractArray{FT, 2},
    phir::Union{AbstractArray{FT, 2}, Nothing},
    phis::Union{AbstractArray{FT, 2}, Nothing},
    phit::Union{AbstractArray{FT, 2}, Nothing},
    si::Int,
    sj::Int,
    sk::Int,
    sr::Int,
    ss::Int,
    st::Int,
    temp::AbstractArray{FT, 3},
) where {FT <: AbstractFloat}
    Nel = size(vin, 2)
    Threads.@threads for e in 1:Nel
        #--------------------------------------------------------------------------------
        if phir isa AbstractArray # apply phir
            for k in 1:sk, j in 1:sj, r in 1:sr
                tot = FT(0)
                if vin_den === nothing
                    for i in 1:si
                        tot +=
                            phir[r, i] *
                            vin[i + ((j - 1) + (k - 1) * sj) * si, e]
                    end
                    temp[r + ((j - 1) + (k - 1) * sj) * sr, 1, e] = tot
                else
                    for i in 1:si
                        tot +=
                            phir[r, i] *
                            vin[i + ((j - 1) + (k - 1) * sj) * si, e] /
                            vin_den[i + ((j - 1) + (k - 1) * sj) * si, e]
                    end
                    temp[r + ((j - 1) + (k - 1) * sj) * sr, 1, e] = tot
                end
            end
        else # in this case, phir is assumed to be an identity matrix
            if vin_den === nothing
                for k in 1:sk, j in 1:sj, r in 1:sr
                    temp[r + ((j - 1) + (k - 1) * sj) * sr, 1, e] =
                        vin[r + ((j - 1) + (k - 1) * sj) * sr, e]
                end
            else
                for k in 1:sk, j in 1:sj, r in 1:sr
                    temp[r + ((j - 1) + (k - 1) * sj) * sr, 1, e] =
                        vin[r + ((j - 1) + (k - 1) * sj) * sr, e] /
                        vin_den[r + ((j - 1) + (k - 1) * sj) * sr, e]
                end
            end
        end
        #--------------------------------------------------------------------------------
        if phis isa AbstractArray # apply phis
            for k in 1:sk, s in 1:ss, r in 1:sr
                tot = FT(0)
                for j in 1:sj
                    tot +=
                        phis[s, j] *
                        temp[r + ((j - 1) + (k - 1) * sj) * sr, 1, e]
                end
                temp[r + ((s - 1) + (k - 1) * ss) * sr, 2, e] = tot
            end
        else # in this case, phis is assumed to be an identity matrix
            for k in 1:sk, s in 1:ss, r in 1:sr
                temp[r + ((s - 1) + (k - 1) * ss) * sr, 2, e] =
                    temp[r + ((s - 1) + (k - 1) * ss) * sr, 1, e]
            end
        end
        #--------------------------------------------------------------------------------
        if phit isa AbstractArray # apply phit
            for t in 1:st, s in 1:ss, r in 1:sr
                tot = FT(0)
                for k in 1:sk
                    tot +=
                        phit[t, k] *
                        temp[r + ((s - 1) + (k - 1) * ss) * sr, 2, e]
                end
                vout[r + ((s - 1) + (t - 1) * ss) * sr, e] = tot
            end
        else # in this case, phit is assumed to be an identity matrix
            for t in 1:st, s in 1:ss, r in 1:sr
                vout[r + ((s - 1) + (t - 1) * ss) * sr, e] =
                    temp[r + ((s - 1) + (t - 1) * ss) * sr, 2, e]
            end
        end
        #--------------------------------------------------------------------------------
    end
    return nothing
end

# This kernel computes the fast tensor-product x vector
# vout = (phit ⊗ phis ⊗ phir) * vin
# using CUDA kernel
# CUDA kernels are faster for light-weight GPU kernels

function ftpxv_hex_CUDA!(
    vin::AbstractArray{FT, 2},
    vin_den::Union{AbstractArray{FT, 2}, Nothing},
    vout::AbstractArray{FT, 2},
    phir::Union{AbstractArray{FT, 2}, Nothing},
    phis::Union{AbstractArray{FT, 2}, Nothing},
    phit::Union{AbstractArray{FT, 2}, Nothing},
    si::Int,
    sj::Int,
    sk::Int,
    sr::Int,
    ss::Int,
    st::Int,
    temp::AbstractArray{FT, 3},
    ::Val{d1m},
    ::Val{d2m},
) where {d1m, d2m, FT <: AbstractFloat}

    i1 = threadIdx().x
    i2 = threadIdx().y # thread ids
    e = blockIdx().x

    s_1 = @cuDynamicSharedMem(FT, (d1m, d2m))

    if (phir isa AbstractArray) && i1 ≤ sr && i2 ≤ si
        l_phir = phir[i1, i2]
    end

    if (phis isa AbstractArray) && i1 ≤ ss && i2 ≤ sj
        l_phis = phis[i1, i2]
    end

    if (phit isa AbstractArray) && i1 ≤ st && i2 ≤ sk
        l_phit = phit[i1, i2]
    end

    # Apply phir -----------------------------------------------------------
    if phir isa AbstractArray
        @inbounds for k in 1:sk, j in 1:sj
            if i1 ≤ sr && i2 ≤ si
                ijk = i2 + ((j - 1) + (k - 1) * sj) * si
                if vin_den === nothing
                    @inbounds s_1[i1, i2] = l_phir * vin[ijk, e]
                else
                    @inbounds s_1[i1, i2] =
                        l_phir * vin[ijk, e] / vin_den[ijk, e]
                end
            end
            sync_threads()
            if i1 ≤ sr && i2 == 1
                @inbounds for i in 2:si
                    s_1[i1, 1] += s_1[i1, i]
                end
            end
            sync_threads()
            if i1 ≤ sr && i2 == 1
                ijk = i1 + ((j - 1) + (k - 1) * sj) * sr
                @inbounds temp[ijk, 1, e] = s_1[i1, 1]
            end
            sync_threads()
        end
    else # in this case, phir is assumed to be an identity matrix
        @inbounds for k in 1:sk
            if i1 ≤ sr && i2 ≤ sj
                ijk = i1 + ((i2 - 1) + (k - 1) * sj) * sr
                if vin_den === nothing
                    @inbounds temp[ijk, 1, e] = vin[ijk, e]
                else
                    @inbounds temp[ijk, 1, e] = vin[ijk, e] / vin_den[ijk, e]
                end
            end
        end
        sync_threads()
    end
    #-----------------------------------------------------------------------

    # Apply phis -----------------------------------------------------------
    if phis isa AbstractArray
        @inbounds for k in 1:sk, r in 1:sr
            if i1 ≤ ss && i2 ≤ sj
                ijk = r + ((i2 - 1) + (k - 1) * sj) * sr
                @inbounds s_1[i1, i2] = l_phis * temp[ijk, 1, e]
            end
            sync_threads()
            if i1 ≤ ss && i2 == 1
                @inbounds for j in 2:sj
                    s_1[i1, 1] += s_1[i1, j]
                end
            end
            sync_threads()
            if i1 ≤ ss && i2 == 1
                ijk = r + ((i1 - 1) + (k - 1) * ss) * sr
                @inbounds temp[ijk, 2, e] = s_1[i1, 1]
            end
            sync_threads()
        end
    else # in this case, phis is assumed to be an identity matrix
        @inbounds for k in 1:sk
            if i1 ≤ sr && i2 ≤ ss
                ijk = i1 + ((i2 - 1) + (k - 1) * ss) * sr
                @inbounds temp[ijk, 2, e] = temp[ijk, 1, e]
            end
        end
        sync_threads()
    end
    #-----------------------------------------------------------------------

    # Apply phit -----------------------------------------------------------
    if phit isa AbstractArray
        @inbounds for s in 1:ss, r in 1:sr
            if i1 ≤ st && i2 ≤ sk
                ijk = r + ((s - 1) + (i2 - 1) * ss) * sr
                @inbounds s_1[i1, i2] = l_phit * temp[ijk, 1, e]
            end
            sync_threads()
            if i1 ≤ st && i2 == 1
                @inbounds for k in 2:sk
                    s_1[i1, 1] += s_1[i1, k]
                end
            end
            sync_threads()
            if i1 ≤ st && i2 == 1
                ijk = r + ((s - 1) + (i1 - 1) * ss) * sr
                @inbounds vout[ijk, e] = s_1[i1, 1]
            end
            sync_threads()
        end
    else # in this case, phit is assumed to be an identity matrix
        @inbounds for t in 1:st
            if i1 ≤ sr && i2 ≤ ss
                ijk = i1 + ((i2 - 1) + (t - 1) * ss) * sr
                @inbounds vout[ijk, e] = temp[ijk, 2, e]
            end
        end
        sync_threads()
    end


    return nothing
end

@kernel function ftpxv_hex_kernel!(
    vin::AbstractArray{FT, 2},
    vin_den::Union{AbstractArray{FT, 2}, Nothing},
    vout::AbstractArray{FT, 2},
    phir::Union{AbstractArray{FT, 2}, Nothing},
    phis::Union{AbstractArray{FT, 2}, Nothing},
    phit::Union{AbstractArray{FT, 2}, Nothing},
    si::Int,
    sj::Int,
    sk::Int,
    sr::Int,
    ss::Int,
    st::Int,
    temp::AbstractArray{FT, 3},
    ::Val{d1m},
    ::Val{d2m},
) where {d1m, d2m, FT <: AbstractFloat}

    e = @index(Group, Linear)
    i1, i2 = @index(Local, NTuple)
    s_1 = @localmem FT (d1m, d2m)
    l_phir = @private FT (1)
    l_phis = @private FT (1)
    l_phit = @private FT (1)

    if (phir isa AbstractArray) && i1 ≤ sr && i2 ≤ si
        l_phir[1] = phir[i1, i2]
    end

    if (phis isa AbstractArray) && i1 ≤ ss && i2 ≤ sj
        l_phis[1] = phis[i1, i2]
    end

    if (phit isa AbstractArray) && i1 ≤ st && i2 ≤ sk
        l_phit[1] = phit[i1, i2]
    end
    # Apply phir -----------------------------------------------------------
    if phir isa AbstractArray
        @inbounds for k in 1:sk, j in 1:sj
            if i1 ≤ sr && i2 ≤ si
                ijk = i2 + ((j - 1) + (k - 1) * sj) * si
                if vin_den === nothing
                    @inbounds s_1[i1, i2] = l_phir[1] * vin[ijk, e]
                else
                    @inbounds s_1[i1, i2] =
                        l_phir[1] * vin[ijk, e] / vin_den[ijk, e]
                end
            end
            @synchronize
            if i1 ≤ sr && i2 == 1
                @inbounds for i in 2:si
                    s_1[i1, 1] += s_1[i1, i]
                end
            end
            @synchronize
            if i1 ≤ sr && i2 == 1
                ijk = i1 + ((j - 1) + (k - 1) * sj) * sr
                @inbounds temp[ijk, 1, e] = s_1[i1, 1]
            end
            @synchronize
        end
    else # in this case, phir is assumed to be an identity matrix
        @inbounds for k in 1:sk
            if i1 ≤ sr && i2 ≤ sj
                ijk = i1 + ((i2 - 1) + (k - 1) * sj) * sr
                if vin_den === nothing
                    @inbounds temp[ijk, 1, e] = vin[ijk, e]
                else
                    @inbounds temp[ijk, 1, e] = vin[ijk, e] / vin_den[ijk, e]
                end
            end
        end
        @synchronize
    end
    #-----------------------------------------------------------------------

    # Apply phis -----------------------------------------------------------
    if phis isa AbstractArray
        @inbounds for k in 1:sk, r in 1:sr
            if i1 ≤ ss && i2 ≤ sj
                ijk = r + ((i2 - 1) + (k - 1) * sj) * sr
                @inbounds s_1[i1, i2] = l_phis[1] * temp[ijk, 1, e]
            end
            @synchronize
            if i1 ≤ ss && i2 == 1
                @inbounds for j in 2:sj
                    s_1[i1, 1] += s_1[i1, j]
                end
            end
            @synchronize
            if i1 ≤ ss && i2 == 1
                ijk = r + ((i1 - 1) + (k - 1) * ss) * sr
                @inbounds temp[ijk, 2, e] = s_1[i1, 1]
            end
            @synchronize
        end
    else # in this case, phis is assumed to be an identity matrix
        @inbounds for k in 1:sk
            if i1 ≤ sr && i2 ≤ ss
                ijk = i1 + ((i2 - 1) + (k - 1) * ss) * sr
                @inbounds temp[ijk, 2, e] = temp[ijk, 1, e]
            end
        end
        @synchronize
    end
    #-----------------------------------------------------------------------

    # Apply phit -----------------------------------------------------------
    if phit isa AbstractArray
        @inbounds for s in 1:ss, r in 1:sr
            if i1 ≤ st && i2 ≤ sk
                ijk = r + ((s - 1) + (i2 - 1) * ss) * sr
                @inbounds s_1[i1, i2] = l_phit[1] * temp[ijk, 1, e]
            end
            @synchronize
            if i1 ≤ st && i2 == 1
                @inbounds for k in 2:sk
                    s_1[i1, 1] += s_1[i1, k]
                end
            end
            @synchronize
            if i1 ≤ st && i2 == 1
                ijk = r + ((s - 1) + (i1 - 1) * ss) * sr
                @inbounds vout[ijk, e] = s_1[i1, 1]
            end
            @synchronize
        end
    else # in this case, phit is assumed to be an identity matrix
        @inbounds for t in 1:st
            if i1 ≤ sr && i2 ≤ ss
                ijk = i1 + ((i2 - 1) + (t - 1) * ss) * sr
                @inbounds vout[ijk, e] = temp[ijk, 2, e]
            end
        end
        @synchronize
    end
    #-----------------------------------------------------------------------
end
