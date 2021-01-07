using MPI
using ClimateMachine
using Logging
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.VTK: writevtk, writepvtu
using KernelAbstractions
using KernelAbstractions.Extras: @unroll

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

include("advection_diffusion_model.jl")

struct Pseudo1D{n, α, β, μ, δ} <: AdvectionDiffusionProblem end

function init_velocity_diffusion!(
    ::Pseudo1D{n, α, β},
    aux::Vars,
    geom::LocalGeometry,
) where {n, α, β}
    # Direction of flow is n with magnitude α
    aux.advection.u = α * n

    # diffusion of strength β in the n direction
    aux.diffusion.D = β * n * n'
end

function initial_condition!(
    ::Pseudo1D{n, α, β, μ, δ},
    state,
    aux,
    localgeo,
    t,
) where {n, α, β, μ, δ}
    ξn = dot(n, localgeo.coord)
    # ξT = SVector(localgeo.coord) - ξn * n
    state.ρ = exp(-(ξn - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
end
Dirichlet_data!(P::Pseudo1D, x...) = initial_condition!(P, x...)
function Neumann_data!(
    ::Pseudo1D{n, α, β, μ, δ},
    ∇state,
    aux,
    x,
    t,
) where {n, α, β, μ, δ}
    ξn = dot(n, x)
    ∇state.ρ =
        -(
            2n * (ξn - μ - α * t) / (4 * β * (δ + t)) *
            exp(-(ξn - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
        )
end


ClimateMachine.init()
ArrayType = ClimateMachine.array_type()

mpicomm = MPI.COMM_WORLD

polynomialorder = 4
base_num_elem = 4

FT = Float64
dim = 3
direction = EveryDirection
l = numlevels = 1
n =  SVector{3, FT}(
1 / sqrt(3),
1 / sqrt(3),
1 / sqrt(3),
)


α = FT(1)
β = FT(1 // 100)
μ = FT(-1 // 2)
δ = FT(1 // 10)
Ne = 2^(l - 1) * base_num_elem
brickrange = ntuple(
    j -> range(FT(-1); length = Ne + 1, stop = 1),
    dim,
)
periodicity = ntuple(j -> false, dim)
bc = ntuple(j -> (1, 2), dim)
topl = StackedBrickTopology(
    mpicomm,
    brickrange;
    periodicity = periodicity,
    boundary = bc,
)

fluxBC = true

grid = DiscontinuousSpectralElementGrid(
    topl,
    FloatType = FT,
    DeviceArray = ArrayType,
    polynomialorder = polynomialorder,
)
model = AdvectionDiffusion{dim}(Pseudo1D{n, α, β, μ, δ}(), flux_bc = fluxBC)
dg = DGModel(
    model,
    grid,
    RusanovNumericalFlux(),
    CentralNumericalFluxSecondOrder(),
    CentralNumericalFluxGradient(),
    direction = direction(),
)

Q = init_ode_state(dg, FT(0))

device = array_device(Q)
comp_stream = Event(device)

comp_stream = ClimateMachine.DGMethods.launch_volume_gradients!(
  dg,
  Q,
  0.0;
  dependencies = comp_stream,
)

wait(comp_stream)

import ClimateMachine.DGMethods: hyperdiff_indexmap



const _ξ1x1, _ξ2x1, _ξ3x1 = Grids._ξ1x1, Grids._ξ2x1, Grids._ξ3x1
const _ξ1x2, _ξ2x2, _ξ3x2 = Grids._ξ1x2, Grids._ξ2x2, Grids._ξ3x2
const _ξ1x3, _ξ2x3, _ξ3x3 = Grids._ξ1x3, Grids._ξ2x3, Grids._ξ3x3
const _M, _MI = Grids._M, Grids._MI
const _x1, _x2, _x3 = Grids._x1, Grids._x2, Grids._x3
const _JcV = Grids._JcV

const _n1, _n2, _n3 = Grids._n1, Grids._n2, Grids._n3
const _sM, _vMI = Grids._sM, Grids._vMI







function _pack_args!(args, head, FT, T)
  if FT == T
    push!(args, head)
    return nothing
  end
  if isprimitivetype(T)
    error("field type $T is incompatible with $FT")
  end
  for field in fieldnames(T)
    _pack_args!(args, :($head.$field), FT, fieldtype(T, field))
  end
end

@generated function pack(::Type{FT}, var) where {FT}
  args = []
  _pack_args!(args, :var, FT, var)
  :(SVector{$(length(args)),$FT}($(args...)))
end

packtype(::Type{FT}, ::Type{T}) where {FT,T} = Core.Compiler.return_type(pack,Tuple{Type{FT},T})



function _unpack_args(::Type{FT}, idx) where {FT<:Real}
  return :(v[$idx]), idx+1
end

function _unpack_args(::Type{NT}, idx) where {NT<:NamedTuple{fields}} where {fields}
  args = []
  for f in fields
    arg, idx = _unpack_args(fieldtype(NT,f), idx)
    push!(args,arg)
  end
  return :(NamedTuple{$fields}(($(args...),))), idx
end

function _unpack_args(::Type{SArray{S,ET,N,L}}, idx) where {S,ET,N,L}
  args = []
  for i in 1:L
    arg, idx = _unpack_args(ET, idx)
    push!(args,arg)
  end
  return :(SArray{$S}($(args...))), idx
end


@generated function unpack(::Type{T}, v) where {T}
  _unpack_args(T,1)[1]
end


#=
function _unpack_grad_args(::Type{FT}, idx) where {FT<:Real}
  return :(v[:,$idx]), idx+1
end

function _unpack_grad_args(::Type{NT}, idx) where {NT<:NamedTuple{fields}} where {fields}
  args = []
  for f in fields
    arg, idx = _unpack_grad_args(fieldtype(NT,f), idx)
    push!(args,arg)
  end
  return :(NamedTuple{$fields}(($(args...),))), idx
end

function _unpack_grad_args(::Type{SArray{S,ET,N,L}}, idx) where {S,ET,N,L}
  args = []
  for i in 1:L
    arg, idx = _unpack_grad_args(ET, idx)
    push!(args,arg)
  end
  return :(SArray{$S}($(args...))), idx
end


@generated function unpack_grad(::Type{T}, v) where {T}
  _unpack_grad_args(T,1)[1]
end
=#

using HybridArrays
import ClimateMachine.MPIStateArrays: vars, array_device

Hprog= HybridArray{Tuple{5,5,5,1,StaticArrays.Dynamic()}}(reshape(Q.data, (5,5,5,1,64)))
Haux = HybridArray{Tuple{5,5,5,15,StaticArrays.Dynamic()}}(reshape(dg.state_auxiliary.data, (5,5,5,15,64)))
Hgradpost = HybridArray{Tuple{5,5,5,3,StaticArrays.Dynamic()}}(reshape(dg.state_gradient_flux.data,(5,5,5,3,64)))

Base.ndims(::KernelAbstractions.ScratchArray{N}) where {N} = N

struct PackedArray{V,D,A,N} <: AbstractArray{V,N}
  data::A
end
PackedArray{V,D}(data::A) where {V,D,A} = PackedArray{V,D,A,ndims(data)-1}(data)
vars(::PackedArray{V}) where {V} = V
array_device(P::PackedArray) = array_device(P.data)
array_device(H::HybridArray) = array_device(H.data)

Base.getindex(P::PackedArray{V,1},i::Integer) where {V} = unpack(V,P.data[:,i])

Base.getindex(P::PackedArray{V,2},i::Integer) where {V} = unpack(V,P.data[i,:])
Base.getindex(P::PackedArray{V,3},i::Integer, j::Integer) where {V} = unpack(V,P.data[i,j,:])

Base.getindex(P::PackedArray{V,4},i::Integer,j::Integer,k::Integer,e::Integer) where {V} = unpack(V,P.data[i,j,k,:,e])
Base.getindex(P::PackedArray{V,4},i::Integer,j::Integer,k::Colon,  e::Integer) where {V} = PackedArray{V,2}(P.data[i,j,k,:,e])


# "Full" field
const FField{V,FT,N1,N2,N3,NV,A} = PackedArray{V,4,HybridArray{Tuple{N1,N2,N3,NV,StaticArrays.Dynamic()},FT,5,5,A}}

FField{V}(array) where {V} = PackedArray{V,4}(array)


Fprog = FField{vars(Q)}(Hprog)
Faux = FField{vars(dg.state_auxiliary)}(Haux)
Fgradpost = FField{vars(dg.state_gradient_flux)}(Hgradpost)


gradarg(state, aux, t) = (ρ = state.ρ,)
gradpost(∇vars, state, aux, t) = (σ = aux.diffusion.D * ∇vars.ρ,)


@kernel function volume_gradients!(
  gradarg::FnGradArg,
  gradpost::FnGradPost,
  Fprog::FField{ProgType,FT,Nq1,Nq2,Nq3,Nprog},
  Faux::FField{AuxType,FT,Nq1,Nq2,Nq3,Naux},
  Fgradpost::FField{GradType,FT,Nq1,Nq2,Nq3,Ngradpost},
  direction,
  Fvgeo,
  t,
  D,
  increment = false,
) where {FnGradArg,FnGradPost,ProgType,AuxType,GradType,FT,Nq1,Nq2,Nq3,Nprog,Naux,Ngradpost}
  @uniform begin
      TransformType = Core.Compiler.return_type(
        gradarg,
        Tuple{ProgType, AuxType, typeof(t)}
      )
      NGrad = length(packtype(FT, TransformType))
  end

  # Transformation from conservative variables to
  # primitive variables (i.e. ρu → u)
  shared_transform = @localmem FT (Nq1, Nq2, NGrad)

  local_transform_gradient = @private SVector{3,FT} (NGrad, Nq3)
  local_prog = @private FT (Nprog, Nq3)
  local_aux = @private FT (Naux, Nq3)
  Gξ3 = @private FT (NGrad, Nq3)

  @uniform L3prog = PackedArray{ProgType,2}(local_prog)
  @uniform L3aux = PackedArray{AuxType,2}(local_aux)

  # Grab the index associated with the current element `e` and the
  # horizontal quadrature indices `i` (in the ξ1-direction),
  # `j` (in the ξ2-direction) [directions on the reference element].
  # Parallelize over elements, then over columns
  e = @index(Group, Linear)
  i, j = @index(Local, NTuple)

  @inbounds begin

      @unroll for k in 1:Nq3
          # Initialize local gradient variables
          @unroll for s in 1:NGrad
              local_transform_gradient[s,k] = SVector{3,FT}(-0,-0,-0)
              Gξ3[s, k] = -zero(FT)
          end
          @unroll for s in 1:Nprog
            local_prog[s,k] = Fprog.data[i,j,k,s,e]
          end
          @unroll for s in 1:Naux
            local_aux[s,k] = Faux.data[i,j,k,s,e]
          end
      end

      # Compute G(q) and write the result into shared memory
      @unroll for k in 1:Nq3
          transform = gradarg(unpack(ProgType, local_prog[:,k]), unpack(AuxType, local_aux[:,k]), t)
          shared_transform[i, j, :] = pack(FT, transform)

          # Synchronize threads on the device
          @synchronize

          ξ1x = Fvgeo[i,j,k,e].∂ξ∂x[1,:]
          ξ2x = Fvgeo[i,j,k,e].∂ξ∂x[2,:]

          # Compute gradient of each state
          @unroll for s in 1:NGrad
              Gξ1 = Gξ2 = zero(FT)

              @unroll for n in 1:Nq1
                  # Smack G with the differentiation matrix
                  Gξ1 += D[i, n] * shared_transform[n, j, s]
                  if dim == 3 || (dim == 2 && direction isa EveryDirection)
                      Gξ2 += D[j, n] * shared_transform[i, n, s]
                  end
                  # Compute the gradient of G over the entire column
                  if dim == 3 && direction isa EveryDirection
                      Gξ3[s, n] += D[n, k] * shared_transform[i, j, s]
                  end
              end

              # Application of chain-rule in ξ1 and ξ2 directions,
              # ∂G/∂xi = ∂ξ1/∂xi * ∂G/∂ξ1, ∂G/∂xi = ∂ξ2/∂xi * ∂G/∂ξ2
              # to get a physical gradient
              local_transform_gradient[s, k] += ξ1x .* Gξ1

              if dim == 3 || (dim == 2 && direction isa EveryDirection)
                  local_transform_gradient[s, k] += ξ2x .* Gξ2
              end
          end

          # Synchronize threads on the device
          @synchronize
      end

      @unroll for k in 1:Nq3
          # Application of chain-rule in ξ3-direction: ∂G/∂xi = ∂ξ3/∂xi * ∂G/∂ξ3
          if dim == 3 && direction isa EveryDirection
              ξ3x = Fvgeo[i,j,k,e].∂ξ∂x[3,:]
              @unroll for s in 1:NGrad
                  local_transform_gradient[s, k] += ξ3x .* Gξ3[s, k]
              end
          end

          post = gradpost(
            unpack(TransformType, local_transform_gradient[:,k]),
            unpack(ProgType, local_prog[:,k]),
            unpack(AuxType, local_aux[:,k]),
            t
          )

          # Write out the result of the kernel to global memory
          if increment
            Fgradpost.data[i,j,k,:,e] += pack(FT, post)
          else
            Fgradpost.data[i,j,k,:,e] = pack(FT, post)
          end

      end
  end
end






function launch_volume_gradients!(
  gradarg::FnGradArg,
  gradpost::FnGradPost,
  Fprog::FField{ProgType,FT,Nq1,Nq2,Nq3,Nprog},
  Faux::FField{AuxType,FT,Nq1,Nq2,Nq3,Naux},
  Fgradpost::FField{GradType,FT,Nq1,Nq2,Nq3,Ngradpost},
  direction,
  grid,
  t;
  dependencies) where {FnGradArg,FnGradPost,ProgType,AuxType,GradType,FT,Nq1,Nq2,Nq3,Nprog,Naux,Ngradpost}


  # We assume (in 3-D) that both x and y directions
  # are discretized using the same polynomial order, Nq[1] == Nq[2].
  # In 2-D, the workgroup spans the entire set of quadrature points:
  # Nq[1] * Nq[2]
  ngroups = size(Fprog.data,5)
  workgroup = (Nq1, Nq2)
  ndrange = (Nq1 * ngroups, Nq2)
  device = array_device(Fprog.data)

  Fvgeo = FField{NamedTuple{(:∂ξ∂x,:M,:MI,:MH,:x,:JcV),Tuple{SMatrix{3,3,Float64,9},FT,FT,FT,SVector{3,FT},FT}}}(HybridArray{Tuple{5,5,5,16,StaticArrays.Dynamic()}}(reshape(grid.vgeo,(5,5,5,16,64))))

  # If the model direction is EveryDirection, we need to perform
  # both horizontal AND vertical kernel calls; otherwise, we only
  # call the kernel corresponding to the model direction `dg.diffusion_direction`
  if dg.diffusion_direction isa EveryDirection ||
     dg.diffusion_direction isa HorizontalDirection

      # We assume N₁ = N₂, so the same polyorder, quadrature weights,
      # and differentiation operators are used
      @assert Nq1 == Nq2
      dependencies = volume_gradients!(device, workgroup)(
        gradarg,
        gradpost,
        Fprog,
        Faux,
        Fgradpost,
        HorizontalDirection(),
        Fvgeo,
        t,
        grid.D[1];
        ndrange = ndrange,
        dependencies = dependencies,
      )
  end

  # Now we call the kernel corresponding to the vertical direction
  if dg.diffusion_direction isa EveryDirection ||
     dg.diffusion_direction isa VerticalDirection

      # Vertical polynomial degree and differentiation matrix
      dependencies = volume_gradients!(device, workgroup)(
        gradarg,
        gradpost,
        Fprog,
        Faux,
        Fgradpost,
        VerticalDirection(),
        Fvgeo,
        t,
        grid.D[3],
        !(dg.diffusion_direction isa VerticalDirection);
        ndrange = ndrange,
        dependencies = dependencies,
      )
  end
  return dependencies
end

wait(launch_volume_gradients!(
  gradarg,
  gradpost,
  Fprog,
  Faux,
  Fgradpost,
  EveryDirection(),
  grid,
  0.0;
  dependencies = comp_stream,
))
