module FTP
using DocStringExtensions
using KernelAbstractions
using CUDA
using ClimateMachine.Mesh.Grids

export ftpxv!, ftpxv_hex!

include("FTP_kernels.jl")

"""
    ftpxv!(grd::DiscontinuousSpectralElementGrid,
        mesh::Symbol,
        opts::Symbol,
        tropts::Bool,
        vin::AbstractArray{FT},
        vout::AbstractArray{FT},
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
"""
function ftpxv!(
    grd::DiscontinuousSpectralElementGrid,
    mesh::Symbol,
    opts::Symbol,
    tropts::Bool,
    vin::AbstractArray{FT},
    vout::AbstractArray{FT},
) where {FT <: AbstractFloat}
    DA = (typeof(parent(vin)) <: Array) ? Array : CuArray
    println("in ftpxv ****************")
    println("typeof(vout) = $(typeof(vout))")
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
    println("si = $si; sj = $sj; sk = $sk; sr = $sr; ss = $ss; st = $st")
    println("size(vin) = $(size(vin))")
    println("size(vout) = $(size(vout))")
    ftpxv_hex!(
        vin,
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
        DA,
    )

    println("in ftpxv ****************; after ftpxv_hex!")
    return nothing
end


# This function computes the fast tensor-product x vector
# vout = (phit ⊗ phis ⊗ phir) * vin
function ftpxv_hex!(
    vin::AbstractArray{FT},
    vout::AbstractArray{FT},
    phir::Union{FTA2D, Nothing},
    phis::Union{FTA2D, Nothing},
    phit::Union{FTA2D, Nothing},
    si::Int,
    sj::Int,
    sk::Int,
    sr::Int,
    ss::Int,
    st::Int,
    temp::FTA3D,
    ::Type{DA},
) where {
    FT <: AbstractFloat,
    DA,
    FTA2D <: AbstractArray{FT, 2},
    FTA3D <: AbstractArray{FT, 3},
}
    Nel = size(vin, 2)
    @assert size(vin, 1) == si * sj * sk
    @assert size(vout, 1) == sr * ss * st


    device = DA <: Array ? CPU() : CUDADevice() # device
    comp_stream = Event(device)

    d1m = max(sr, ss, st)
    d2m = max(si, sj, sk)

    workgroup = (d1m, d2m)
    ndrange = (Nel * d1m, d2m)
    comp_stream = ftpxv_hex_kernel!(device, workgroup)(
        vin,
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
        ndrange = ndrange,
        dependencies = (comp_stream,),
    )
    wait(comp_stream)
    return nothing
end


end
