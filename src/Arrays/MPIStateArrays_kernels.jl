using Requires
@init @require CUDAnative = "be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
    using .CUDAnative
end
using KernelAbstractions.Extras: @unroll

@kernel function knl_fillsendbuf!(
    ::Val{Np},
    ::Val{nvar},
    sendbuf,
    buf,
    vmapsend,
    nvmapsend,
) where {Np, nvar}

    i = @index(Global, Linear)
    @inbounds begin
        e, n = fldmod1(vmapsend[i], Np)
        @unroll for s in 1:nvar
            sendbuf[s, i] = buf[n, s, e]
        end
    end
end

@kernel function knl_transferrecvbuf!(
    ::Val{Np},
    ::Val{nvar},
    buf,
    recvbuf,
    vmaprecv,
    nvmaprecv,
) where {Np, nvar}

    i = @index(Global, Linear)
    @inbounds begin
        e, n = fldmod1(vmaprecv[i], Np)
        @unroll for s in 1:nvar
            buf[n, s, e] = recvbuf[s, i]
        end
    end
end
