# I do not think that we need these now...
# abstract type AbstractSpaceParameter end
# abstract type AbstractTimeParameter end
# abstract type AbstractSpaceConfiguration end
# abstract type AbstractTimeConfiguration end
# abstract type AbstractSpaceState end
# abstract type AbstractTimeState end

"""
    Runner

Data structure containing the spatial and temporal Runner structures

"""
abstract type AbstractSpaceRunner end
abstract type AbstractTimeRunner end
struct Runner{SpaceType<:AbstractSpaceRunner, TimeType<:AbstractTimeRunner}
  "Runner structure for the spacial discretization"
  _space_::SpaceType
  "Runner structure for the temporal discretization"
  _time_::TimeType
  function Runner(mpicomm,
                  spacemethod::Symbol, spaceargs,
                  timemethod::Symbol, timeargs)
    space = createrunner(Val(spacemethod), mpicomm; spaceargs...)
    time = createrunner(Val(timemethod), mpicomm, space; timeargs...)
    Runner(space, time)
  end
  Runner(space, time) = new{typeof(space), typeof(time)}(space, time)
end

Base.getindex(r::Runner, s) = r[Symbol(s)]
function Base.getindex(r::Runner, s::Symbol)
  s == :space && return r._space_
  s == :time && return r._time_
  error("Runner can be accessed with \"space\" or \"time\" strings or Symbols")
end

createrunner(::Val{T}, a...) where T =
error("No implementation of `createrunner` method `$(T)`")

similarQ(space::AbstractSpaceRunner) =
error("no implementation of `similarQ` for `$(typeof(space))`")

gettime(r::Runner) = gettime(r[:space])
gettime(r::AbstractSpaceRunner) =
error("no implementation of `gettime` for `$(typeof(r))`")

settime!(r::Runner) = settime!(r[:space])
settime!(r::AbstractSpaceRunner) =
error("no implementation of `settime!` for `$(typeof(r))`")

initspacestate!(x, r::Runner; a...) = initstate!(x, r[:space]; a...)
inittimestate!(x, r::Runner; a...) = initstate!(x, r[:time]; a...)
initstate!(x, r::Union{AbstractSpaceRunner, AbstractTimeRunner}; a...) =
error("no implementation of `initstate!` for `$(typeof(r))`")

estimatedt(r::Runner; a...) = estimatedt(r[:space]; a...)
estimatedt(r::AbstractSpaceRunner; a...) =
error("no implementation of `estimatedt` for `$(typeof(r))`")

L2solutionnorm(r::Runner, x...; kw...) = L2solutionnorm(r[:space], x...; kw...)
L2solutionnorm(r::AbstractSpaceRunner, x...; kw...) =
error("no implementation of `L2solutionnorm` for `$(typeof(r))`")

writevtk(r::Runner, x...; kw...) = writevtk(r[:space], x...; kw...)
writevtk(r::AbstractSpaceRunner, x...; kw...) =
error("no implementation of `writevtk` for `$(typeof(r))`")

run!(r::Runner; kw...) = run!(r[:time], r[:space]; kw...)
run!(time::AbstractTimeRunner, space::AbstractSpaceRunner; kw...) =
error("no implementation of `run!` for `$(typeof(time))`")

getQ(r::Runner) = getQ(r[:space])
getQ(r::AbstractSpaceRunner) =
error("no implementation of `getQ` for `$(typeof(r))`")

getmesh(r::Runner) = getmesh(r[:space])
getmesh(r::AbstractSpaceRunner) =
error("no implementation of `getmesh` for `$(typeof(r))`")

rhs!(rhs, r::Runner) = rhs!(rhs, r[:space])
rhs!(rhs, r::AbstractSpaceRunner) =
error("no implementation of `rhs!` for `$(typeof(r))`")
