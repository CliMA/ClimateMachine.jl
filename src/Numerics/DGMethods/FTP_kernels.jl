# This kernel computes the fast tensor-product x vector
# vout = (phit ⊗ phis ⊗ phir) * vin
#
@kernel function ftpxv_hex_kernel!(
    vin::AbstractArray{FT, 2},
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

    s_phi = @localmem FT (d1m, d2m) # allocate fast shared memory
    s_1 = @localmem FT (d1m, d2m)

    # Apply phir -----------------------------------------------------------
    if !(phir === nothing)
        if i1 ≤ sr && i2 ≤ si
            s_phi[i1, i2] = phir[i1, i2] # load phir into fast shared memory
        end
        @synchronize

        for k in 1:sk, j in 1:sj
            if i1 ≤ sr && i2 ≤ si
                ijk = i2 + ((j - 1) + (k - 1) * sj) * si
                s_1[i1, i2] = s_phi[i1, i2] * vin[ijk, e]
            end
            @synchronize
            if i1 ≤ sr && i2 == 1
                for i in 2:si
                    s_1[i1, 1] += s_1[i1, i]
                end
            end
            @synchronize
            if i1 ≤ sr && i2 == 1
                ijk = i1 + ((j - 1) + (k - 1) * sj) * sr
                temp[ijk, 1, e] = s_1[i1, 1]
            end
            @synchronize
        end
    else # in this case, phir is assumed to be an identity matrix
        for k in 1:sk
            if i1 ≤ sr && i2 ≤ sj
                ijk = i1 + ((i2 - 1) + (k - 1) * sj) * sr
                temp[ijk, 1, e] = vin[ijk, e]
            end
        end
        @synchronize
    end
    #-----------------------------------------------------------------------

    # Apply phis -----------------------------------------------------------
    if !(phis === nothing)
        if i1 ≤ ss && i2 ≤ sj
            s_phi[i1, i2] = phis[i1, i2] # load phis into fast shared memory
        end
        @synchronize

        for k in 1:sk, r in 1:sr
            if i1 ≤ ss && i2 ≤ sj
                ijk = r + ((i2 - 1) + (k - 1) * sj) * sr
                s_1[i1, i2] = s_phi[i1, i2] * temp[ijk, 1, e]
            end
            @synchronize
            if i1 ≤ ss && i2 == 1
                for j in 2:sj
                    s_1[i1, 1] += s_1[i1, j]
                end
            end
            @synchronize
            if i1 ≤ ss && i2 == 1
                ijk = r + ((i1 - 1) + (k - 1) * ss) * sr
                temp[ijk, 2, e] = s_1[i1, 1]
            end
            @synchronize
        end
    else # in this case, phis is assumed to be an identity matrix
        for k in 1:sk
            if i1 ≤ sr && i2 ≤ ss
                ijk = i1 + ((i2 - 1) + (k - 1) * ss) * sr
                temp[ijk, 2, e] = temp[ijk, 1, e]
            end
        end
        @synchronize
    end
    #-----------------------------------------------------------------------

    # Apply phit -----------------------------------------------------------
    if !(phit === nothing)
        if i1 ≤ st && i2 ≤ sk
            s_phi[i1, i2] = phit[i1, i2] # load phit into fast shared memory
        end
        @synchronize

        for s in 1:ss, r in 1:sr
            if i1 ≤ st && i2 ≤ sk
                ijk = r + ((s - 1) + (i2 - 1) * ss) * sr
                s_1[i1, i2] = s_phi[i1, i2] * temp[ijk, 1, e]
            end
            @synchronize
            if i1 ≤ st && i2 == 1
                for k in 2:sk
                    s_1[i1, 1] += s_1[i1, k]
                end
            end
            @synchronize
            if i1 ≤ st && i2 == 1
                ijk = r + ((s - 1) + (i1 - 1) * ss) * sr
                vout[ijk, e] = s_1[i1, 1]
            end
            @synchronize
        end
    else # in this case, phit is assumed to be an identity matrix
        for t in 1:st
            if i1 ≤ sr && i2 ≤ ss
                ijk = i1 + ((i2 - 1) + (t - 1) * ss) * sr
                vout[ijk, e] = temp[ijk, 2, e]
            end
        end
        @synchronize
    end
    #-----------------------------------------------------------------------

end
