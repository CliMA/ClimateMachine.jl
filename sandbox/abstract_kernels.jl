
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using StaticArrays

#=
For the divergence kernel.
input = flux, grid
output = flux_divergence
size(flux) = (gauss-lobatto points, dims, state, # number of elements)
size(flux_divergence) = (gauss-lobatto points, state, # number of elements)
\partial_t \vec{u} = \nabla \cdot F
=#

# Split off into separate parameter struct
const _ξ1x1, _ξ2x1, _ξ3x1 = Grids._ξ1x1, Grids._ξ2x1, Grids._ξ3x1
const _ξ1x2, _ξ2x2, _ξ3x2 = Grids._ξ1x2, Grids._ξ2x2, Grids._ξ3x2
const _ξ1x3, _ξ2x3, _ξ3x3 = Grids._ξ1x3, Grids._ξ2x3, Grids._ξ3x3
const _M, _MI = Grids._M, Grids._MI
const _x1, _x2, _x3 = Grids._x1, Grids._x2, Grids._x3
const _JcV = Grids._JcV

const _n1, _n2, _n3 = Grids._n1, Grids._n2, Grids._n3
const _sM, _vMI = Grids._sM, Grids._vMI

function launch_volume_divergence!(grid::G, flux_divergence, flux; dependencies = nothing) where {G <: AbstractGrid}

    nrealelem = length(grid.interiorelems)
    device = array_device(flux)
    N = polynomialorders(grid)
    dim = length(grid.D)

    FT = eltype(flux)
    workgroup = (N[1]+1, N[2]+1) # same size as shared memory tiles
    ndrange = ( (N[1]+1)*nrealelem, (N[2]+1))
    comp_stream = dependencies
    flux_divergence.realdata[:] .= 0 # necesary for now
    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction `dg.diffusion_direction`
    comp_stream = volume_divergence_kernel!(device, workgroup)(
        flux_divergence.data,
        flux.data,
        Val(dim),
        Val(N[1]),
        grid.vgeo,
        grid.D[1],
        ndrange = ndrange,
        dependencies = comp_stream,
    )

    return comp_stream
end

@kernel function volume_divergence_kernel!(
    flux_divergence,
    flux,
    ::Val{dim},
    ::Val{polyorder},
    vgeo,
    D,
) where {dim, polyorder}
    # This allows for the variables to remain after @synchronize
    # for CPU computation
    @uniform begin
        N = polyorder
        FT = eltype(flux)
        Nq = N + 1
        Nqk = (dim == 2) ? 1 : Nq
        local_flux = MArray{Tuple{3, 1}, FT}(undef)
        local_flux_3 = MArray{Tuple{1, 1}, FT}(undef)
    end

    # Arrays for F, and the differentiation matrix D
    shared_flux = @localmem FT (2, Nq, Nq) # shared memory on the gpu
    s_D = @localmem FT (Nq, Nq)            # shared memory on the gpu (can remove)

    # Storage for tendency and mass inverse M⁻¹, **perhaps in @uniform block**
    local_tendency = @private FT (Nqk,)   # thread private memory
    local_MI = @private FT (Nqk,)         # thread private memory

    # Grab the index associated with the current element `e` and the
    # horizontal quadrature indices `i` (in the ξ1-direction),
    # `j` (in the ξ2-direction) [directions on the reference element].
    # Parallelize over elements, then over columns
    e = @index(Group, Linear)    # Group is the index of the work group
    i, j = @index(Local, NTuple) # i, j is the index within a work group
    
    @inbounds begin
        # load differentiation matrix into local memory
        s_D[i, j] = D[i, j]
        @unroll for k in 1:Nqk # if 2D Nqk = 1, if 3D Nq
            ijk = i + Nq * ((j - 1) + Nq * (k - 1)) # convert to linear indices
            # initialize local tendency
            local_tendency[k] = zero(FT)
            # read in mass matrix inverse for element `e` in each plane k
            local_MI[k] = vgeo[ijk, _MI, e]
        end
        # end of boiler plate loading

        @unroll for k in 1:Nqk # search
            @synchronize # perhaps moved up outside the loop
            ijk = i + Nq * ((j - 1) + Nq * (k - 1)) 
            M = vgeo[ijk, _M, e]
            # Extract Jacobian terms
            # ∂/∂x¹ = ∂ξ¹/ ∂x¹ ∂/∂ξ¹ + ∂ξ²/ ∂x¹ ∂/∂ξ² + ∂ξ³/ ∂x¹ ∂/∂ξ³
            # ∂/∂x² = ∂ξ¹/ ∂x² ∂/∂ξ¹ + ∂ξ²/ ∂x² ∂/∂ξ² + ∂ξ³/ ∂x² ∂/∂ξ³
            # ∂/∂x³ = ∂ξ¹/ ∂x³ ∂/∂ξ¹ + ∂ξ²/ ∂x³ ∂/∂ξ² + ∂ξ³/ ∂x³ ∂/∂ξ³
            # The integration by parts is then done with respect to the
            # (ξ¹, ξ², ξ³) coordinate system 
            # (wich is why terms associated with the columns appear)
            ξ1x1 = vgeo[ijk, _ξ1x1, e]
            ξ1x2 = vgeo[ijk, _ξ1x2, e]
            ξ1x3 = vgeo[ijk, _ξ1x3, e]
            
            ξ2x1 = vgeo[ijk, _ξ2x1, e]
            ξ2x2 = vgeo[ijk, _ξ2x2, e]
            ξ2x3 = vgeo[ijk, _ξ2x3, e]

            ξ3x1 = vgeo[ijk, _ξ3x1, e]
            ξ3x2 = vgeo[ijk, _ξ3x2, e]
            ξ3x3 = vgeo[ijk, _ξ3x3, e]
            # ∂ᵗθ = ∇⋅(u⃗θ -κ∇θ), F1 = uθ , F2 = vθ, F3 = wθ, where u⃗ = (u,v,w)
            # Computes the local inviscid fluxes Fⁱⁿᵛ
            # Need to load in user passed in flux aka make flux into local flux
            fill!(local_flux, -zero(eltype(local_flux))) # can remove this

            # probably becomes flux[ijk, e, 1]
            shared_flux[1, i, j] = flux[ijk, e, 1] # local_flux[1]
            
            # probably becomes flux[ijk, e, 2] 
            shared_flux[2, i, j] = flux[ijk, e, 2] # local_flux[2] 

            # not taking derivative in third direction
            local_flux_3         = flux[ijk, e, 3] # local_flux[3] 

            # Build "inside metrics" flux
            F1 = shared_flux[1, i, j]
            F2 = shared_flux[2, i, j]
            F3 = local_flux_3

            shared_flux[1, i, j] = M * (ξ1x1 * F1 + ξ1x2 * F2 + ξ1x3 * F3)          
            shared_flux[2, i, j] = M * (ξ2x1 * F1 + ξ2x2 * F2 + ξ2x3 * F3)
            local_flux_3         = M * (ξ3x1 * F1 + ξ3x2 * F2 + ξ3x3 * F3)
            ## Everything above is point-wise multiplication
            # Fix Me  
            
            @unroll for n in 1:Nqk
                MI = local_MI[n]
                local_tendency[n] += MI * s_D[k, n] * local_flux_3
            end

            @synchronize
            
            # Weak "inside metrics" derivative.
            # Computes the rest of the volume term: M⁻¹SᵀF (note the transpose)
            # note: (1) The indices are reversed from the usual matrix mult
            #       (2) The stiffness matrix comes in since shared flux has
            #           the mass matrix already multiplied through
            MI = local_MI[k]
            @unroll for n in 1:Nq
                # ξ1-grid lines
                local_tendency[k] += MI * s_D[n, i] * shared_flux[1, n, j]                                                                                
                # ξ2-grid lines
                local_tendency[k] += MI * s_D[n, j] * shared_flux[2, i, n]
            end

            #FIXME: ijk does not persist after @synchronize and must be redefined here
            #       Maybe try @private and change the variable to a tuple
            #ijk = i + Nq * ((j - 1) + Nq * (k - 1))
            #flux_divergence[ijk, e, 1] += local_tendency[k]        
        end

        @unroll for k in 1:Nqk # search
            ijk = i + Nq * ((j - 1) + Nq * (k - 1))
            flux_divergence[ijk, e, 1] += local_tendency[k] 
        end
    end
end

"""
    launch_interface_gradients!(dg, state_prognostic, t; surface::Symbol, dependencies)

Launches horizontal and vertical kernels for computing the interface gradients.
The argument `surface` is either `:interior` or `:exterior`, which denotes whether
we are computing interface gradients on boundaries which are interior (exterior resp.)
to the _parallel_ boundary.
"""
function launch_interface_divergence!(
    grid::G,
    flux_divergence,
    flux;
    dependencies = nothing,
) where {G <: AbstractGrid}
    # MPI
    # @assert surface === :interior || surface === :exterior

    FT = eltype(flux)

    device = array_device(flux)
    dim = length(grid.D)
    N = polynomialorders(grid)
    
    # Assumes poly order is the same in every direction
    # TODO: copy from new kernel with variable polynomial order
    if dim == 1
        Np = (N[1] + 1)
        Nfp = 1
        nface = 2
    elseif dim == 2
        Np = (N[1] + 1) * (N[2] + 1)
        Nfp = (N[1] + 1)
        nface = 4
    elseif dim == 3
        Np = (N[1] + 1) * (N[2] + 1) * (N[3] + 1)
        Nfp = (N[1] + 1) * (N[2] + 1)
        nface = 6
    end
    flux_divergence.realdata[:] .= 0   
    workgroup = Nfp
    elems = grid.interiorelems
    ndrange = Nfp * length(grid.interiorelems)

    comp_stream = dependencies

    #TODO: fix with appropriate function calls
    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction `dg.diffusion_direction`
    comp_stream = interface_divergence_kernel!(device, workgroup)(
        flux_divergence.data,
        flux.data,
        Val(dim),
        Val(N[1]),
        grid.vgeo,
        grid.sgeo,
        grid.vmap⁻,
        grid.vmap⁺,
        grid.elemtobndy,
        elems,
        nface,
        Np,
        ndrange = ndrange,
        dependencies = comp_stream,
    )
    return comp_stream
end

@doc """
    function interface_divergence_kernel!(
        flux_divergence,
        flux,
        ::Val{dim},
        ::Val{polyorder},
        direction, # we will probably want to get rid of this
        numerical_flux_first_order,  # this is metadata, we need to specify what numerical flux we are using
        tendency,
        vgeo,
        sgeo,
        vmap⁻,
        vmap⁺,
        elemtobndy,
        elems,
    )

Compute kernel for evaluating the interface tendencies for the
DG form:

∫ₑ ψ⋅ ∂q/∂t dV - ∫ₑ ∇ψ⋅F⃗ dV + ∮ₑ ψF⃗⋅n̂ dS,

or equivalently in matrix form:

dQ/dt = M⁻¹(MS + DᵀM F⃗ + ∑ᶠ LᵀMᶠF⃗ ).

This kernel computes the surface terms: M⁻¹ ∑ᶠ LᵀMᶠ(F)),
where M is the mass matrix, Mf is the face mass matrix, L is an interpolator
from volume to face, and Fⁱⁿᵛ⋆, Fᵛⁱˢᶜ⋆
are the numerical fluxes for the inviscid and viscous
fluxes, respectively.  interface_tendency!
"""
@kernel function interface_divergence_kernel!(
    flux_divergence,
    flux,
    ::Val{dim},
    ::Val{polyorder},
    vgeo,
    sgeo,
    vmap⁻,
    vmap⁺,
    elemtobndy,
    elems,
    nface,
    Np,
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(flux)

        # faces: (1, 2, 3, 4, 5, 6)  = (West, East, South, North, Bottom, Top)
        # We think...
        faces = 1:nface

        Nq = N + 1
        Nqk = dim == 2 ? 1 : Nq
    end
    eI = @index(Group, Linear)
    n = @index(Local, Linear)

    e = @private Int (1,)
    
    e[1] = elems[eI]
    # goes over too many points right now
    for f in faces
        # Inter-element index
        e⁻ = e[1]
        normal_vector = SVector(
            sgeo[_n1, n, f, e⁻],
            sgeo[_n2, n, f, e⁻],
            sgeo[_n3, n, f, e⁻],
        )
        # Get surface mass, volume mass inverse

        # sM = surface mass matrix
        # vMI = volume mass inverse matrix
        sM, vMI = sgeo[_sM, n, f, e⁻], sgeo[_vMI, n, f, e⁻]

        # Ids corresponding to the workitem on the face
        # vmap⁻ and vmap⁺ are the linear indices
        id⁻, id⁺ = vmap⁻[n, f, e⁻], vmap⁺[n, f, e⁻]

        # neighbor element index
        e⁺ = ((id⁺ - 1) ÷ Np) + 1 # this is why 1 indexing is failure

        # vid's are cartesian indices of the GL points
        # needs to be modified by the faces?
        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1, ((id⁺ - 1) % Np) + 1

        # No Flux BC and Central Fluxes for the interfaces allows the kernel
        # to be extensible
        # TODO: make it no flux on the boundary for now and add the flux later
        # bctype = elemtobndy[f, e⁻]

        # flux_divergence[vmap⁻] = (flux[vmap⁻] + flux[vmap⁺])/2
        # ∫dA F⃗⋅n̂ 
        # normal_vector[1] * (flux[vmap⁻,1] + flux[vmap⁺,1])/2
        # normal_vector[2] * (flux[vmap⁻,2] + flux[vmap⁺,2])/2
        # normal_vector[3] * (flux[vmap⁻,3] + flux[vmap⁺,3])/2
        # # ∂ᵗ∫dVρ = - ∫dS Φ⃗⋅n̂ = - flux on boundary
        
        local_flux  = normal_vector[1] * (flux[vid⁻, e⁻, 1] + flux[vid⁺, e⁺, 1])/2
        local_flux += normal_vector[2] * (flux[vid⁻, e⁻, 2] + flux[vid⁺, e⁺, 2])/2
        local_flux += normal_vector[3] * (flux[vid⁻, e⁻, 3] + flux[vid⁺, e⁺, 3])/2
        
        # Update RHS (in outer face loop): M⁻¹ Mfᵀ(Fⁱⁿᵛ⋆ ))
        flux_divergence[vid⁻, e⁻, 1] += vMI * sM * local_flux # Warning, need to think about this
        # @show vid⁻ e⁻ id⁻ local_flux flux_divergence[vid⁻, e⁻, 1] vMI sM f
        # Need to wait after even faces to avoid race conditions
        # Note: only need synchronization for F%2 == 0, but this crashed...
        @synchronize
    end
end

function launch_volume_gradient!(grid::G, gradient, state; dependencies = nothing) where {G <: AbstractGrid}

    nrealelem = length(grid.interiorelems)
    device = array_device(gradient)
    N = polynomialorders(grid)
    dim = length(grid.D)

    FT = eltype(state)
    workgroup = (N[1]+1, N[2]+1) # same size as shared memory tiles
    ndrange = ( (N[1]+1)*nrealelem, (N[2]+1))
    comp_stream = dependencies
    gradient.realdata[:] .= 0 # necessary for now
    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction `dg.diffusion_direction`
    comp_stream = volume_gradient_kernel!(device, workgroup)(
        gradient.data,
        state.data,
        Val(dim),
        Val(N),
        grid.vgeo,
        grid.D[1],
        ndrange = ndrange,
        dependencies = comp_stream,
    )

    return comp_stream
end

# TODO: should be similar to vectorgradients kernel
@kernel function volume_gradient_kernel!(
    gradient,
    state,
    ::Val{dim},
    ::Val{polyorder},
    vgeo,
    D,
) where {dim, polyorder}
    # This allows for the variables to remain after @synchronize
    # for CPU computation
    @uniform begin
        N = polyorder
        FT = eltype(state)
        @inbounds Nq1 = N[1] + 1
        @inbounds Nq2 = N[2] + 1
        Nq3 = (dim == 2) ? 1 : N[3] + 1
        local_flux = MArray{Tuple{3, 1}, FT}(undef)
        local_flux_3 = MArray{Tuple{1, 1}, FT}(undef)
    end

    # Arrays for F, and the differentiation matrix D
    shared_flux = @localmem FT (2, Nq1, Nq2) # shared memory on the gpu

    # Storage for tendency and mass inverse M⁻¹, **perhaps in @uniform block**
    local_tendency = @private FT (Nq3, dim)   # thread private memory
    local_MI = @private FT (Nq3,)         # thread private memory

    # Grab the index associated with the current element `e` and the
    # horizontal quadrature indices `i` (in the ξ1-direction),
    # `j` (in the ξ2-direction) [directions on the reference element].
    # Parallelize over elements, then over columns
    e = @index(Group, Linear)    # Group is the index of the work group
    i, j = @index(Local, NTuple) # i, j is the index within a work group
    
    @inbounds begin
        # load differentiation matrix into local memory
        @unroll for k in 1:Nq3 # if 2D Nq3 = 1, if 3D Nq
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1)) # convert to linear indices
            # initialize local tendency
            local_tendency[k, 1] = zero(FT)
            local_tendency[k, 2] = zero(FT)
            local_tendency[k, 3] = zero(FT)
            # read in mass matrix inverse for element `e` in each plane k
            local_MI[k] = vgeo[ijk, _MI, e]
        end
        # end of boiler plate loading

        @unroll for k in 1:Nq3 # search
            @synchronize # perhaps moved up outside the loop
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1)) 
            # Extract Jacobian terms
            # ∂/∂x¹ = ∂ξ¹/ ∂x¹ ∂/∂ξ¹ + ∂ξ²/ ∂x¹ ∂/∂ξ² + ∂ξ³/ ∂x¹ ∂/∂ξ³
            # ∂/∂x² = ∂ξ¹/ ∂x² ∂/∂ξ¹ + ∂ξ²/ ∂x² ∂/∂ξ² + ∂ξ³/ ∂x² ∂/∂ξ³
            # ∂/∂x³ = ∂ξ¹/ ∂x³ ∂/∂ξ¹ + ∂ξ²/ ∂x³ ∂/∂ξ² + ∂ξ³/ ∂x³ ∂/∂ξ³
            # The integration by parts is then done with respect to the
            # (ξ¹, ξ², ξ³) coordinate system 
            # (which is why terms associated with the columns appear)
            # maybe load gradient into shared memory
            shared_flux[1, i, j] = state[ijk, e, 1] # local_flux[1]

            ξ3x1 = vgeo[ijk, _ξ3x1, e]
            ξ3x2 = vgeo[ijk, _ξ3x2, e]
            ξ3x3 = vgeo[ijk, _ξ3x3, e]

            @unroll for n in 1:Nq3
                # ξ3-grid lines
                ∂Q∂ξ³ = D[n, k] * shared_flux[1, i, j]
                local_tendency[n, 1] += ξ3x1 * ∂Q∂ξ³
                local_tendency[n, 2] += ξ3x2 * ∂Q∂ξ³
                local_tendency[n, 3] += ξ3x3 * ∂Q∂ξ³
            end
            
            @synchronize
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            ξ1x1 = vgeo[ijk, _ξ1x1, e]
            ξ1x2 = vgeo[ijk, _ξ1x2, e]
            ξ1x3 = vgeo[ijk, _ξ1x3, e]

            ξ2x1 = vgeo[ijk, _ξ2x1, e]
            ξ2x2 = vgeo[ijk, _ξ2x2, e]
            ξ2x3 = vgeo[ijk, _ξ2x3, e]

            @unroll for n in 1:Nq1
                # ξ1-grid lines
                ∂Q∂ξ¹ = D[i, n] * shared_flux[1, n, j]
                gradient[ijk, e, 1] += ξ1x1 * ∂Q∂ξ¹
                gradient[ijk, e, 2] += ξ1x2 * ∂Q∂ξ¹
                gradient[ijk, e, 3] += ξ1x3 * ∂Q∂ξ¹                                                                                 
            end

            @unroll for n in 1:Nq2
                ∂Q∂ξ² = D[j, n] * shared_flux[1, i, n]
                # ξ2-grid lines
                gradient[ijk, e, 1] += ξ2x1 * ∂Q∂ξ²
                gradient[ijk, e, 2] += ξ2x2 * ∂Q∂ξ²
                gradient[ijk, e, 3] += ξ2x3 * ∂Q∂ξ²                                                                                
            end
        end

        @unroll for k in 1:Nq3 
            # ξ3-grid lines
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            gradient[ijk, e, 1] += local_tendency[k, 1] 
            gradient[ijk, e, 2] += local_tendency[k, 2] 
            gradient[ijk, e, 3] += local_tendency[k, 3] 
        end
    end
end

"""
    launch_interface_gradients!(dg, state_prognostic, t; surface::Symbol, dependencies)

Launches horizontal and vertical kernels for computing the interface gradients.
The argument `surface` is either `:interior` or `:exterior`, which denotes whether
we are computing interface gradients on boundaries which are interior (exterior resp.)
to the _parallel_ boundary.
"""
function launch_interface_gradient!(
    grid::G,
    gradient,
    state;
    dependencies = nothing,
) where {G <: AbstractGrid}
    # MPI
    # @assert surface === :interior || surface === :exterior

    FT = eltype(state)
    device = array_device(state)
    dim = length(grid.D)
    N = polynomialorders(grid)
    
    # Assumes poly order is the same in every direction
    # TODO: copy from new kernel with variable polynomial order
    if dim == 1
        Np = (N[1] + 1)
        Nfp = 1
        nface = 2
    elseif dim == 2
        Np = (N[1] + 1) * (N[2] + 1)
        Nfp = (N[1] + 1)
        nface = 4
    elseif dim == 3
        Np = (N[1] + 1) * (N[2] + 1) * (N[3] + 1)
        Nfp = (N[1] + 1) * (N[2] + 1)
        nface = 6
    end
    gradient.realdata[:] .= 0   
    workgroup = Nfp
    elems = grid.interiorelems
    ndrange = Nfp * length(grid.interiorelems)

    comp_stream = dependencies

    #TODO: fix with appropriate function calls
    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction `dg.diffusion_direction`
    comp_stream = interface_gradient_kernel!(device, workgroup)(
        gradient.data,
        state.data,
        Val(dim),
        Val(N[1]),
        grid.vgeo,
        grid.sgeo,
        grid.vmap⁻,
        grid.vmap⁺,
        grid.elemtobndy,
        elems,
        nface,
        Np,
        ndrange = ndrange,
        dependencies = comp_stream,
    )
    return comp_stream
end

@doc """
    function interface_gradient_kernel!(
        gradient,
        state,
        ::Val{dim},
        ::Val{polyorder},
        direction, # we will probably want to get rid of this
        numerical_state_first_order,  # this is metadata, we need to specify what numerical state we are using
        tendency,
        vgeo,
        sgeo,
        vmap⁻,
        vmap⁺,
        elemtobndy,
        elems,
    )

Compute kernel for evaluating the interface tendencies for the
DG form:

∫ₑ ψ⋅ ∂q/∂t dV - ∫ₑ ∇ψ⋅F⃗ dV + ∮ₑ ψF⃗⋅n̂ dS,

or equivalently in matrix form:

dQ/dt = M⁻¹(MS + DᵀM F⃗ + ∑ᶠ LᵀMᶠF⃗ ).

This kernel computes the surface terms: M⁻¹ ∑ᶠ LᵀMᶠ(F)),
where M is the mass matrix, Mf is the face mass matrix, L is an interpolator
from volume to face, and Fⁱⁿᵛ⋆, Fᵛⁱˢᶜ⋆
are the numerical statees for the inviscid and viscous
statees, respectively.  interface_tendency!
"""
@kernel function interface_gradient_kernel!(
    gradient,
    state,
    ::Val{dim},
    ::Val{polyorder},
    vgeo,
    sgeo,
    vmap⁻,
    vmap⁺,
    elemtobndy,
    elems,
    nface,
    Np,
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(state)

        # faces: (1, 2, 3, 4, 5, 6)  = (West, East, South, North, Bottom, Top)
        # We think...
        faces = 1:nface

        Nq = N + 1
        Nqk = dim == 2 ? 1 : Nq
    end
    eI = @index(Group, Linear)
    n = @index(Local, Linear)

    e = @private Int (1,)
    
    e[1] = elems[eI]
    # goes over too many points right now
    for f in faces
        # Inter-element index
        e⁻ = e[1]
        normal_vector = SVector(
            sgeo[_n1, n, f, e⁻],
            sgeo[_n2, n, f, e⁻],
            sgeo[_n3, n, f, e⁻],
        )
        # Get surface mass, volume mass inverse

        # sM = surface mass matrix
        # vMI = volume mass inverse matrix
        sM, vMI = sgeo[_sM, n, f, e⁻], sgeo[_vMI, n, f, e⁻]

        # Ids corresponding to the workitem on the face
        # vmap⁻ and vmap⁺ are the linear indices
        id⁻, id⁺ = vmap⁻[n, f, e⁻], vmap⁺[n, f, e⁻]

        # neighbor element index
        e⁺ = ((id⁺ - 1) ÷ Np) + 1 # this is why 1 indexing is failure

        # vid's are cartesian indices of the GL points
        # needs to be modified by the faces?
        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1, ((id⁺ - 1) % Np) + 1

        # No Flux BC and Central Fluxes for the interfaces allows the kernel
        # to be extensible
        # TODO: make it no state on the boundary for now and add the state later
        # bctype = elemtobndy[f, e⁻]

        # gradient[vmap⁻] = (state[vmap⁻] + state[vmap⁺])/2
        # ∫dA F⃗⋅n̂ 
        # normal_vector[1] * (state[vmap⁻,1] + state[vmap⁺,1])/2
        # normal_vector[2] * (state[vmap⁻,2] + state[vmap⁺,2])/2
        # normal_vector[3] * (state[vmap⁻,3] + state[vmap⁺,3])/2
        # # ∂ᵗ∫dVρ = - ∫dS Φ⃗⋅n̂ = - state on boundary
        
        local_state_1 = normal_vector[1] * (state[vid⁺, e⁺] - state[vid⁻, e⁻])/2
        local_state_2 = normal_vector[2] * (state[vid⁺, e⁺] - state[vid⁻, e⁻])/2
        local_state_3 = normal_vector[3] * (state[vid⁺, e⁺] - state[vid⁻, e⁻])/2
        
        # Update RHS (in outer face loop): M⁻¹ Mfᵀ(Fⁱⁿᵛ⋆ ))
        gradient[vid⁻, e⁻, 1] += vMI * sM * local_state_1
        gradient[vid⁻, e⁻, 2] += vMI * sM * local_state_2
        gradient[vid⁻, e⁻, 3] += vMI * sM * local_state_3

        # @show vid⁻ e⁻ id⁻ local_state gradient[vid⁻, e⁻, 1] vMI sM f
        # Need to wait after even faces to avoid race conditions
        # Note: only need synchronization for F%2 == 0, but this crashed...
        @synchronize
    end
end

