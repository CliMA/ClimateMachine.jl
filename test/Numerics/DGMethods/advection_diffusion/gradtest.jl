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


function gradient_argument(
  m::AdvectionDiffusion,
  state,
  aux,
  t,
)
  (ρ = state.ρ,)
end
gradient_extract(m::AdvectionDiffusion, args...) = gradient_extract(m.diffusion, args...)
gradient_extract(::Diffusion{1}, ∇vars, state, aux, t) = (σ = aux.diffusion.D * ∇vars.ρ,)






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

using HybridArrays


reshape(Q.data, (5,5))

struct Field{V,A}
  data::A
end
Field{V}(data::A) where {V,A} = Field{V,A}(data)
MPIStateArrays.vars(::Field{V}) where {V} = V

@kernel function volume_gradients!(
  balance_law::BalanceLaw,
  ::Val{info},
  direction,
  state_prognostic,
  state_gradient_flux,
  Qhypervisc_grad,
  state_auxiliary,
  vgeo,
  t,
  D,
  ::Val{hypervisc_indexmap},
  elems,
  increment = false,
) where {info, hypervisc_indexmap}
  @uniform begin
      dim = info.dim
      FT = eltype(state_prognostic.data)

      PrognosticType = vars(state_prognostic)
      AuxiliaryType = vars(state_auxiliary)

      TransformType = Core.Compiler.return_type(
        gradient_argument,
        Tuple{typeof(balance_law), PrognosticType, AuxiliaryType, typeof(t)}
      )

      num_state_prognostic = length(packtype(FT, PrognosticType))
      ngradstate = length(packtype(FT, TransformType))
      num_state_auxiliary = length(packtype(FT, AuxiliaryType))

      num_state_gradient_flux = number_states(balance_law, GradientFlux())

      # Kernel assumes same polynomial order in both
      # horizontal directions (x, y)
      @inbounds Nq1 = info.Nq[1]
      @inbounds Nq2 = info.Nq[2]
      Nq3 = info.Nqk

      ngradtransformstate = num_state_prognostic

      local_transform = MArray{Tuple{ngradstate}, FT}(undef)
      local_state_gradient_flux =
          MArray{Tuple{num_state_gradient_flux}, FT}(undef)
  end

  # Transformation from conservative variables to
  # primitive variables (i.e. ρu → u)
  shared_transform = @localmem FT (Nq1, Nq2, ngradstate)

  local_state_prognostic = @private FT (ngradtransformstate, Nq3)
  local_state_auxiliary = @private FT (num_state_auxiliary, Nq3)
  local_transform_gradient = @private FT (3, ngradstate, Nq3)
  Gξ3 = @private FT (ngradstate, Nq3)

  # Grab the index associated with the current element `e` and the
  # horizontal quadrature indices `i` (in the ξ1-direction),
  # `j` (in the ξ2-direction) [directions on the reference element].
  # Parallelize over elements, then over columns
  e = @index(Group, Linear)
  i, j = @index(Local, NTuple)


  @inbounds @views begin
      @unroll for k in 1:Nq3
          # Initialize local gradient variables
          @unroll for s in 1:ngradstate
              local_transform_gradient[1, s, k] = -zero(FT)
              local_transform_gradient[2, s, k] = -zero(FT)
              local_transform_gradient[3, s, k] = -zero(FT)
              Gξ3[s, k] = -zero(FT)
          end

          # Load prognostic and auxiliary variables
          ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
          @unroll for s in 1:ngradtransformstate
              local_state_prognostic[s, k] = state_prognostic.data[ijk, s, e]
          end
          @unroll for s in 1:num_state_auxiliary
              local_state_auxiliary[s, k] = state_auxiliary.data[ijk, s, e]
          end
      end

      # Compute G(q) and write the result into shared memory
      @unroll for k in 1:Nq3
          local_transform = gradient_argument(
            balance_law,
            unpack(vars(state_prognostic), local_state_prognostic[:,k]),
            unpack(vars(state_auxiliary), local_state_auxiliary[:,k]),
            t,
          )
          local_transform_vec = pack(FT, local_transform)
          @unroll for s in 1:ngradstate
              shared_transform[i, j, s] = local_transform_vec[s]
          end

          # Synchronize threads on the device
          @synchronize

          ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
          ξ1x1, ξ1x2, ξ1x3 =
              vgeo[ijk, _ξ1x1, e], vgeo[ijk, _ξ1x2, e], vgeo[ijk, _ξ1x3, e]

          # Compute gradient of each state
          @unroll for s in 1:ngradstate
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
              local_transform_gradient[1, s, k] += ξ1x1 * Gξ1
              local_transform_gradient[2, s, k] += ξ1x2 * Gξ1
              local_transform_gradient[3, s, k] += ξ1x3 * Gξ1

              if dim == 3 || (dim == 2 && direction isa EveryDirection)
                  ξ2x1, ξ2x2, ξ2x3 = vgeo[ijk, _ξ2x1, e],
                  vgeo[ijk, _ξ2x2, e],
                  vgeo[ijk, _ξ2x3, e]
                  local_transform_gradient[1, s, k] += ξ2x1 * Gξ2
                  local_transform_gradient[2, s, k] += ξ2x2 * Gξ2
                  local_transform_gradient[3, s, k] += ξ2x3 * Gξ2
              end
          end

          # Synchronize threads on the device
          @synchronize
      end

      @unroll for k in 1:Nq3
          ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

          # Application of chain-rule in ξ3-direction: ∂G/∂xi = ∂ξ3/∂xi * ∂G/∂ξ3
          if dim == 3 && direction isa EveryDirection
              ξ3x1, ξ3x2, ξ3x3 = vgeo[ijk, _ξ3x1, e],
              vgeo[ijk, _ξ3x2, e],
              vgeo[ijk, _ξ3x3, e]
              @unroll for s in 1:ngradstate
                  local_transform_gradient[1, s, k] += ξ3x1 * Gξ3[s, k]
                  local_transform_gradient[2, s, k] += ξ3x2 * Gξ3[s, k]
                  local_transform_gradient[3, s, k] += ξ3x3 * Gξ3[s, k]
              end
          end

          if num_state_gradient_flux > 0
            l = gradient_extract(balance_law,
             unpack_grad(TransformType, local_transform_gradient[:,:,k]),
             unpack(vars(state_prognostic), local_state_prognostic[:,k]),
             unpack(vars(state_auxiliary), local_state_auxiliary[:,k]),
             t
            )
            local_state_gradient_flux = pack(FT, l)

              # Write out the result of the kernel to global memory
              @unroll for s in 1:num_state_gradient_flux
                  if increment
                      state_gradient_flux[ijk, s, e] +=
                          local_state_gradient_flux[s]
                  else
                      state_gradient_flux[ijk, s, e] =
                          local_state_gradient_flux[s]
                  end
              end
          end
      end
  end
end

function launch_volume_gradients!(dg, state_prognostic, t; dependencies)
  FT = eltype(state_prognostic)
  Qhypervisc_grad, _ = dg.states_higher_order

  # Workgroup is determined by the number of quadrature points
  # in the horizontal direction. For each horizontal quadrature
  # point, we operate on a stack of quadrature in the vertical
  # direction. (Iteration space is in the horizontal)
  info = ClimateMachine.DGMethods.basic_launch_info(dg)

  # We assume (in 3-D) that both x and y directions
  # are discretized using the same polynomial order, Nq[1] == Nq[2].
  # In 2-D, the workgroup spans the entire set of quadrature points:
  # Nq[1] * Nq[2]
  workgroup = (info.Nq[1], info.Nq[2])
  ndrange = (info.Nq[1] * info.nrealelem, info.Nq[2])
  comp_stream = dependencies

  # If the model direction is EveryDirection, we need to perform
  # both horizontal AND vertical kernel calls; otherwise, we only
  # call the kernel corresponding to the model direction `dg.diffusion_direction`
  if dg.diffusion_direction isa EveryDirection ||
     dg.diffusion_direction isa HorizontalDirection

      # We assume N₁ = N₂, so the same polyorder, quadrature weights,
      # and differentiation operators are used
      horizontal_polyorder = info.N[1]
      horizontal_D = dg.grid.D[1]
      comp_stream = volume_gradients!(info.device, workgroup)(
          dg.balance_law,
          Val(info),
          HorizontalDirection(),
          Field{vars(state_prognostic)}(state_prognostic.data),
          dg.state_gradient_flux.data,
          Qhypervisc_grad.data,
          Field{vars(dg.state_auxiliary)}(dg.state_auxiliary.data),
          dg.grid.vgeo,
          t,
          horizontal_D,
          Val(hyperdiff_indexmap(dg.balance_law, FT)),
          dg.grid.topology.realelems,
          ndrange = ndrange,
          dependencies = comp_stream,
      )
  end

  # Now we call the kernel corresponding to the vertical direction
  if dg.diffusion_direction isa EveryDirection ||
     dg.diffusion_direction isa VerticalDirection

      # Vertical polynomial degree and differentiation matrix
      vertical_polyorder = info.N[info.dim]
      vertical_D = dg.grid.D[info.dim]
      comp_stream = volume_gradients!(info.device, workgroup)(
          dg.balance_law,
          Val(info),
          VerticalDirection(),
          Field{vars(state_prognostic)}(state_prognostic.data),
          dg.state_gradient_flux.data,
          Qhypervisc_grad.data,
          Field{vars(dg.state_auxiliary)}(dg.state_auxiliary.data),
          dg.grid.vgeo,
          t,
          vertical_D,
          Val(hyperdiff_indexmap(dg.balance_law, FT)),
          dg.grid.topology.realelems,
          # If we are computing the volume gradient in every direction, we
          # need to increment into the appropriate fields _after_ the
          # horizontal computation.
          !(dg.diffusion_direction isa VerticalDirection),
          ndrange = ndrange,
          dependencies = comp_stream,
      )
  end
  return comp_stream
end

wait(launch_volume_gradients!(
         dg,
         Q,
         0.0;
         dependencies = comp_stream,
       ))
