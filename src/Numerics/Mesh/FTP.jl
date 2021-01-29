using DocStringExtensions
using KernelAbstractions
using CUDA

include("FTP_kernels.jl")

"""
    ftpxv!(grd::DiscontinuousSpectralElementGrid,
        mesh::Symbol,
        opts::Symbol,
        tropts::Bool,
        vin::AbstractArray{FT},
        vout::AbstractArray{FT},
        vin_den::Union{AbstractArray{FT}, Nothing} = nothing,
    ) where {FT<:AbstractFloat}

This function computes the inner derivatives of a field using the fast
tensor-product algorithm on a given mesh. Currently, DG mesh (`:m1`) and
over-integration mesh (`:m2`) are supported. The function provides the 
building blocks for constructing various terms in the DG formulation and 
for constructing some diagnostic fields.

# Arguments
 - `grd`: DiscontinuousSpectralElementGrid
 - `mesh`: quadrature mesh (currently, `:m1` and `:m2` supported)
 - `opts`: inner derivative to be computed (currently, `:ξ1`, `:ξ2`, `:ξ3` supported)
 - `tropts`: transpose (true/false)
 - `vin`: input
 - `vout`: output
 - `vin_den`: Scale vin with vin_den
"""
function ftpxv!(
    grd::DiscontinuousSpectralElementGrid,
    mesh::Symbol,
    opts::Symbol,
    tropts::Bool,
    vin::AbstractArray{FT},
    vout::AbstractArray{FT};
    vin_den::Union{AbstractArray{FT}, Nothing} = nothing,
) where {FT <: AbstractFloat}
    DA = (typeof(parent(vin)) <: Array) ? Array : CuArray
    if mesh == :m1 # DG Spectral element mesh
        #---------------------------------------------------
        si, sj, sk = polynomialorders(grd) .+ 1
        sr, ss, st = si, sj, sk
        if opts == :∂ξ₁
            phir = (tropts ? grd.Dᵀ[1] : grd.D[1])
            phis, phit = nothing, nothing
        elseif opts == :∂ξ₂
            phis = (tropts ? grd.Dᵀ[2] : grd.D[2])
            phir, phit = nothing, nothing
        elseif opts == :∂ξ₃
            phit = (tropts ? grd.Dᵀ[3] : grd.D[3])
            phir, phis = nothing, nothing
        elseif opts == :lag2leg # Lagrangian to modal basis for filtering
            phir = grd.lag2leg[1]
            phis = grd.lag2leg[2]
            phit = grd.lag2leg[3]
        elseif opts == :leg2lag # modal (Legendre) to Lagrangian
            phir = grd.leg2lag[1]
            phis = grd.leg2lag[2]
            phit = grd.leg2lag[3]
        else
            error("ftpxv!, mesh 1: Unsupported opts")
        end
        #---------------------------------------------------
    elseif mesh == :m2 # finer quadrature for over-integration
        #---------------------------------------------------
        if opts == :∂ξ₁
            phir = (tropts ? grd.D_m2ᵀ[1] : grd.D_m2[1])
            phis = (tropts ? grd.B_m2ᵀ[2] : grd.B_m2[2])
            phit = (tropts ? grd.B_m2ᵀ[3] : grd.B_m2[3])
        elseif opts == :∂ξ₂
            phir = (tropts ? grd.B_m2ᵀ[1] : grd.B_m2[1])
            phis = (tropts ? grd.D_m2ᵀ[2] : grd.D_m2[2])
            phit = (tropts ? grd.B_m2ᵀ[3] : grd.B_m2[3])
        elseif opts == :∂ξ₃
            phir = (tropts ? grd.B_m2ᵀ[1] : grd.B_m2[1])
            phis = (tropts ? grd.B_m2ᵀ[2] : grd.B_m2[2])
            phit = (tropts ? grd.D_m2ᵀ[3] : grd.D_m2[3])
        elseif opts == :basis
            phir = (tropts ? grd.B_m2ᵀ[1] : grd.B_m2[1])
            phis = (tropts ? grd.B_m2ᵀ[2] : grd.B_m2[2])
            phit = (tropts ? grd.B_m2ᵀ[3] : grd.B_m2[3])
        else
            error("ftpxv!, mesh 2: Unsupported opts")
        end
        sr, si = size(phir)
        ss, sj = size(phis)
        st, sk = size(phit)
        #---------------------------------------------------
    else
        error("unsupported mesh/quadrature")
    end

    # Launching computational kernel
    Nel = size(vin, 2)
    device = DA <: Array ? CPU() : CUDADevice() # device
    comp_stream = Event(device)

    d1m = max(sr, ss, st)
    d2m = max(si, sj, sk)
    if device === CPU()
        ftpxv_hex_CPU!(
            vin,
            vin_den,
            vout,
            phir,
            phis,
            phit,
            si,
            sj,
            sk,
            sr,
            ss,
            st,
            grd.ftp_storage,
        )
    else
        # CUDA version
        @cuda threads = (d1m, d2m) blocks = (Nel) shmem = d1m * d2m * sizeof(FT) ftpxv_hex_CUDA!(
            vin,
            vin_den,
            vout,
            phir,
            phis,
            phit,
            si,
            sj,
            sk,
            sr,
            ss,
            st,
            grd.ftp_storage,
            Val(d1m),
            Val(d2m),
        )
    end
    return nothing
end

function ftpxv!(
    vin::AbstractArray{FT},
    vout::AbstractArray{FT},
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
    # Launching computational kernel
    Nel = size(vin, 2)
    DA = (typeof(parent(vin)) <: Array) ? Array : CuArray
    device = DA <: Array ? CPU() : CUDADevice() # device
    comp_stream = Event(device)

    d1m = max(sr, ss, st)
    d2m = max(si, sj, sk)
    if device === CPU()
        ftpxv_hex_CPU!(
            vin,
            nothing,
            vout,
            phir,
            phis,
            phit,
            si,
            sj,
            sk,
            sr,
            ss,
            st,
            temp,
        )
    else
        # CUDA version
        @cuda threads = (d1m, d2m) blocks = (Nel) shmem = d1m * d2m * sizeof(FT) ftpxv_hex_CUDA!(
            vin,
            nothing,
            vout,
            phir,
            phis,
            phit,
            si,
            sj,
            sk,
            sr,
            ss,
            st,
            temp,
            Val(d1m),
            Val(d2m),
        )
    end
    return nothing
end
