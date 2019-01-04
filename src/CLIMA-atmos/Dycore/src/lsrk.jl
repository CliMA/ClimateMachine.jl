struct LSRKState
  dt
  RKA
  RKB
  RKC
  rhs
  function LSRKState(dt, state::State{DeviceArray}) where DeviceArray
    # Fourth-order, low-storage, Runge–Kutta scheme of Carpenter and Kennedy
    # (1994) ((5,4) 2N-Storage RK scheme.
    #
    # Ref:
    # @TECHREPORT{CarpenterKennedy1994,
    #   author = {M.~H. Carpenter and C.~A. Kennedy},
    #   title = {Fourth-order {2N-storage} {Runge-Kutta} schemes},
    #   institution = {National Aeronautics and Space Administration},
    #   year = {1994},
    #   number = {NASA TM-109112},
    #   address = {Langley Research Center, Hampton, VA},
    # }

    # FIXME: Consider whether we should really set DFloat here or not...
    DFloat = eltype(state.Q)
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

    # FIXME: Handle better for GPU?
    rhs = DeviceArray(zeros(DFloat, size(state.Q)))

    new(dt, RKA, RKB, RKC, rhs)
  end
end
function timestepinfocallback(lsrkstate::LSRKState, x)
  cb = EveryXSecondsCallback(x) do
    time = totaltime(cb)
    steps = numberofsteps(cb)
    average_step_time_ = time / steps
    average_stage_time = time / steps / length(lsrkstate.RKA)
    @info "LSRK Time Stepper Info" steps, average_step_time_ average_stage_time
    return 0
  end
end

run!(s, ts::LSRKState, p...) = lowstoraagerkrun!(s, ts, p...)

function lowstoraagerkrun!(state, lsrkstate::LSRKState, parameters,
                           configuration, timeend, callbacks = ())
  t0 = state.time

  RKA = lsrkstate.RKA
  RKB = lsrkstate.RKB
  RKC = lsrkstate.RKC
  dt = lsrkstate.dt

  # Loop through an initialize callbacks (if they need it)
  foreach(callbacks) do cb
    try
      cb(true)
    catch
    end
  end

  step = 0
  while state.time < timeend
    step += 1
    for s = 1:length(RKA)
      rhs!(lsrkstate.rhs, state, parameters, configuration)

      # update solution and scale RHS
      lowstorageRKupdate!(Val(parameters.dim), Val(parameters.N),
                          lsrkstate.rhs, state.Q, configuration.vgeo,
                          configuration.mesh.realelems,
                          RKA[s%length(RKA)+1], RKB[s], dt)
      state.time += RKC[s] * dt
    end
    state.time = t0 + step * dt

    # FIXME: Determine better way to handle postcallback behavior

    # Current behavior:
    #   retval = 1 exit after all callbacks
    #   retval = 2 exit immediately
    retval = 0
    for cb in callbacks
      # FIXME: Consider whether callbacks need anything, or if function closure
      #        can be used for everything
      retval = max(cb(), retval)
      retval == 2 && return
    end
    retval == 1 && return

  end
end

# {{{ Update solution (for all dimensions)
# FIXME Pass string intp requie for uuid?
function lowstorageRKupdate!(::Val{dim}, ::Val{N}, rhs::Array, Q, vgeo, elems,
                             rka, rkb, dt) where {dim, N}
  @inbounds for e = elems, s = 1:_nstate, i = 1:(N+1)^dim
    Q[i, s, e] += rkb * dt * rhs[i, s, e] * vgeo[i, _MJI, e]
    rhs[i, s, e] *= rka
  end
end
# }}}
