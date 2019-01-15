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
  s == :spacerunner && return r._space_
  s == :timerunner && return r._time_
  s == :time && return r._space_[:time]
  s == :Q && return r._space_[:Q]
  s == :hostQ && return r._space_[:hostQ]
  error("""
        getindex for the $(typeof(r)) supports:
        `:spacerunner` => gets the spatial discretization runner
        `:timerunner`  => gets the temporal discretization runner
        `:time`        => gets the spatial runners time
        `:Q`           => gets the spatial runners state Q
        `:hostQ`       => gets a host copy spatial runners state Q
        """)
end

createrunner(::Val{T}, a...) where T =
error("No implementation of `createrunner` method `$(T)`")

initspacestate!(f::Function, r::Runner; a...) = initstate!(r[:spacerunner], f;
                                                           a...)

inittimestate!(f::Function, r::Runner; a...) = initstate!(r[:timerunner], f;
                                                          a...)
inittimestate!(r::Runner, x; a...) = initstate!(r[:timerunner], x; a...)

initstate!(f::Function, r::Union{AbstractSpaceRunner, AbstractTimeRunner};
           a...) = initstate!(r, f; a...)

initstate!(r::Union{AbstractSpaceRunner, AbstractTimeRunner}, x; a...) =
error("no implementation of `initstate!` for `$(typeof(r))`")

estimatedt(r::Runner; a...) = estimatedt(r[:spacerunner]; a...)
estimatedt(r::AbstractSpaceRunner; a...) =
error("no implementation of `estimatedt` for `$(typeof(r))`")

L2solutionnorm(r::Runner, x...; kw...) = L2solutionnorm(r[:spacerunner], x...;
                                                        kw...)
L2solutionnorm(r::AbstractSpaceRunner, x...; kw...) =
error("no implementation of `L2solutionnorm` for `$(typeof(r))`")

writevtk(r::Runner, x...; kw...) = writevtk(r[:spacerunner], x...; kw...)
writevtk(r::AbstractSpaceRunner, x...; kw...) =
error("no implementation of `writevtk` for `$(typeof(r))`")

run!(r::Runner; kw...) = run!(r[:timerunner], r[:spacerunner]; kw...)
run!(time::AbstractTimeRunner, space::AbstractSpaceRunner; kw...) =
error("no implementation of `run!` for `$(typeof(time))`")

rhs!(rhs, r::Runner) = rhs!(rhs, r[:space])
rhs!(rhs, r::AbstractSpaceRunner) =
error("no implementation of `rhs!` for `$(typeof(r))`")
