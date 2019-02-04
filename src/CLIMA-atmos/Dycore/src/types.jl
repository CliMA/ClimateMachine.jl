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
  spacerunner::SpaceType
  "Runner structure for the temporal discretization"
  timerunner::TimeType
  function Runner(mpicomm,
                  spacemethod::Symbol, spaceargs,
                  timemethod::Symbol, timeargs)
    space = createrunner(Val(spacemethod), mpicomm; spaceargs...)
    time = createrunner(Val(timemethod), mpicomm, space; timeargs...)
    Runner(space, time)
  end
  Runner(space, time) = new{typeof(space), typeof(time)}(space, time)
end

createrunner(::Val{T}, a...) where T =
error("No implementation of `createrunner` method `$(T)`")

initspacestate!(f::Function, r::Runner; a...) = initstate!(r.spacerunner, f;
                                                           a...)

inittimestate!(f::Function, r::Runner; a...) = initstate!(r.timerunner, f;
                                                          a...)
inittimestate!(r::Runner, x; a...) = initstate!(r.timerunner, x; a...)

initstate!(f::Function, r::Union{AbstractSpaceRunner, AbstractTimeRunner};
           a...) = initstate!(r, f; a...)

initstate!(r::Union{AbstractSpaceRunner, AbstractTimeRunner}, x; a...) =
error("no implementation of `initstate!` for `$(typeof(r))`")

estimatedt(r::Runner; a...) = estimatedt(r.spacerunner; a...)
estimatedt(r::AbstractSpaceRunner; a...) =
error("no implementation of `estimatedt` for `$(typeof(r))`")

getrealelems(r::AbstractSpaceRunner) =
error("no implementation of `getrealelems` for `$(typeof(r))`")

solutiontime(r::AbstractSpaceRunner) =
error("no implementation of `solutiontime` for `$(typeof(r))`")

setsolutiontime!(r::AbstractSpaceRunner, v) =
error("no implementation of `setsolutiontime!` for `$(typeof(r))`")

getQ(r::AbstractSpaceRunner) =
error("no implementation of `getQ` for `$(typeof(r))`")
getstateid(r::AbstractSpaceRunner) =
error("no implementation of `getstateid` for `$(typeof(r))`")
getmoistid(r::AbstractSpaceRunner) =
error("no implementation of `getmoistid` for `$(typeof(r))`")
gettraceid(r::AbstractSpaceRunner) =
error("no implementation of `gettraceid` for `$(typeof(r))`")

L2solutionnorm(r::Runner, x...; kw...) = L2solutionnorm(r.spacerunner, x...;
                                                        kw...)
L2solutionnorm(r::AbstractSpaceRunner, x...; kw...) =
error("no implementation of `L2solutionnorm` for `$(typeof(r))`")

L2errornorm(r::Runner, x...; kw...) = L2errornorm(r.spacerunner, x...;
                                                        kw...)
L2errornorm(r::AbstractSpaceRunner, x...; kw...) =
error("no implementation of `L2errornorm` for `$(typeof(r))`")

writevtk(r::Runner, x...; kw...) = writevtk(r.spacerunner, x...; kw...)
writevtk(r::AbstractSpaceRunner, x...; kw...) =
error("no implementation of `writevtk` for `$(typeof(r))`")

run!(r::Runner; kw...) = run!(r.timerunner, r.spacerunner; kw...)
run!(time::AbstractTimeRunner, space::AbstractSpaceRunner; kw...) =
error("no implementation of `run!` for `$(typeof(time))`")

rhs!(rhs, r::Runner) = rhs!(rhs, r[:space])
rhs!(rhs, r::AbstractSpaceRunner) =
error("no implementation of `rhs!` for `$(typeof(r))`")
