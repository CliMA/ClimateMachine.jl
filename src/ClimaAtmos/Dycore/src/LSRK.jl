module LSRK

using Requires

@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
  using .CUDAnative.CUDAdrv

  include("LSRK_cuda.jl")
end

using ..CLIMAAtmosDycore
AD = CLIMAAtmosDycore
using Base: @kwdef

"""
    Parameters

Data structure containing the low storage RK parameters
"""
# {{{ Parameters
@kwdef struct Parameters # <: AD.AbstractTimeParameter
end
# }}}

"""
    Configuration

Data structure containing the low storage RK configuration
"""
# {{{ Configuration
struct Configuration{DFloat} # <: AD.AbstractTimeConfiguration
  "low storage RK coefficient vector A (rhs scaling)"
  RKA
  "low storage RK coefficient vector B (rhs add in scaling)"
  RKB
  "low storage RK coefficient vector C (time scaling)"
  RKC
  "Storage for RHS during the LSRK update"
  rhs
  function Configuration(params::Parameters,
                         mpicomm,
                         spacerunner::AD.AbstractSpaceRunner)
    rhs = similar(AD.getQ(spacerunner))
    fill!(rhs, 0)

    DFloat = eltype(rhs)
    RKA = (DFloat(0),
           DFloat(-567301805773)  / DFloat(1357537059087),
           DFloat(-2404267990393) / DFloat(2016746695238),
           DFloat(-3550918686646) / DFloat(2091501179385),
           DFloat(-1275806237668) / DFloat(842570457699 ))

    RKB = (DFloat(1432997174477) / DFloat(9575080441755 ),
           DFloat(5161836677717) / DFloat(13612068292357),
           DFloat(1720146321549) / DFloat(2090206949498 ),
           DFloat(3134564353537) / DFloat(4481467310338 ),
           DFloat(2277821191437) / DFloat(14882151754819))

    RKC = (DFloat(0),
           DFloat(1432997174477) / DFloat(9575080441755),
           DFloat(2526269341429) / DFloat(6820363962896),
           DFloat(2006345519317) / DFloat(3224310063776),
           DFloat(2802321613138) / DFloat(2924317926251))

    new{DFloat}(RKA, RKB, RKC, rhs)
  end
end
#}}}

"""
    State

Data structure containing the low storage RK state
"""
# {{{ State
mutable struct State{DFloat} # <: AD.AbstractTimeState
  """
  `DfloatReal` number specifying the time step

  If a function the arguments will be `(state, configuration, parameters)` and
  it should return a `Real` number for the time step

  no default value
  """
  dt::DFloat
  function State(config::Configuration, x...)
    DFloat = eltype(config.rhs)
    new{DFloat}(zero(DFloat))
  end
end

# }}}

"""
    Runner

Data structure containing the runner for the vanilla DG discretization of
the compressible Euler equations

"""
# {{{ Runner
struct Runner <: AD.AbstractTimeRunner
  params::Parameters
  config::Configuration
  state::State
  function Runner(mpicomm, spacerunner::AD.AbstractSpaceRunner; args...)
    params = Parameters(;args...)
    config = Configuration(params, mpicomm, spacerunner)
    state = State(config, params, spacerunner)
    new(params, config, state)
  end
end
AD.createrunner(::Val{:LSRK}, m, s; a...) = Runner(m, s; a...)
# }}}

AD.initstate!(runner::Runner, dt::Real) = runner.state.dt = dt

# {{{ run!
function AD.run!(runner::Runner, spacerunner::AD.AbstractSpaceRunner;
                 timeend::Real=Inf, stopaftertimeend=true,
                 numberofsteps::Integer=0, callbacks=()) where {SP, T<:State}

  @assert isfinite(timeend) || numberofsteps > 0

  params = runner.params
  config = runner.config
  state = runner.state

  t0 = AD.solutiontime(spacerunner)

  RKA = config.RKA
  RKB = config.RKB
  RKC = config.RKC
  dt = state.dt

  # Loop through an initialize callbacks (if they need it)
  foreach(callbacks) do cb
    try
      cb(true)
    catch
    end
  end

  step = 0
  time = t0
  rhs = config.rhs
  Q = AD.getQ(spacerunner)
  realelems = AD.getrealelems(spacerunner)
  while time < timeend
    step += 1
    for s = 1:length(RKA)
      AD.rhs!(rhs, spacerunner)

      # update solution and scale RHS
      update!(Val(size(Q,2)), Val(size(Q,1)), rhs, Q, realelems,
              RKA[s%length(RKA)+1], RKB[s], dt)
      time += RKC[s] * dt
      AD.setsolutiontime!(spacerunner, time)
    end
    time = t0 + step * dt
    AD.setsolutiontime!(spacerunner, time)

    # FIXME: Determine better way to handle postcallback behavior

    # Current behavior:
    #   retval = 1 exit after all callbacks
    #   retval = 2 exit immediately
    retval = 0
    for (i, cb) in enumerate(callbacks)
      # FIXME: Consider whether callbacks need anything, or if function closure
      #        can be used for everything
      thisretval = cb()
      thisretval = (thisretval == nothing) ? 0 : thisretval
      !(thisretval in (0, 1, 2)) &&
      error("callback #$(i) returned invalid value. It should return either:
            `nothing` (continue time stepping)
            `0`       (continue time stepping)
            `1`       (stop time stepping after all callbacks)
            `2`       (stop time stepping immediately)")
      retval = max(thisretval, retval)
      retval == 2 && return
    end
    retval == 1 && return

    # Figure out if we should stop
    if numberofsteps == step
      return
    end
    if !stopaftertimeend && (t0 + (step+1) * dt) > timeend
      return
    end
  end
end
# }}}

# {{{ Update solution (for all dimensions)
function update!(::Val{nstates}, ::Val{Np}, rhs::Array, Q, elems, rka, rkb,
                 dt) where {nstates, Np}
  @inbounds for e = elems, s = 1:nstates, i = 1:Np
    Q[i, s, e] += rkb * dt * rhs[i, s, e]
    rhs[i, s, e] *= rka
  end
end
# }}}

end
