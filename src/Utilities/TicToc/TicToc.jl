"""
    TicToc -- timing measurement

Low-overhead time interval measurement via minimally invasive macros.

"""
module TicToc

using Printf

export @tic, @toc, tictoc

# disable to reduce overhead
const tictoc_track_memory = true

if tictoc_track_memory

    mutable struct TimingInfo
        ncalls::Int
        time::UInt64
        allocd::Int64
        gctime::UInt64
        curr::UInt64
        mem::Base.GC_Num
    end
    TimingInfo() = TimingInfo(
        0,
        0,
        0,
        0,
        0,
        Base.GC_Num(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )

else # !tictoc_track_memory

    mutable struct TimingInfo
        ncalls::Int
        time::UInt64
        curr::UInt64
    end
    TimingInfo() = TimingInfo(0, 0, 0)

end # if tictoc_track_memory

const timing_infos = TimingInfo[]
const timing_info_names = Symbol[]
const atexit_function_registered = Ref(false)

# `@tic` helper
function _tic(nm)
    exti = Symbol("tictoc__", nm)
    global timing_info_names
    if exti in timing_info_names
        throw(ArgumentError("$(nm) already used in @tic"))
    end
    push!(timing_info_names, exti)
    @static if tictoc_track_memory
        quote
            global $(exti)
            $(exti).curr = time_ns()
            $(exti).mem = Base.gc_num()
        end
    else
        quote
            global $(exti)
            $(exti).curr = time_ns()
        end
    end
end

"""
    @tic nm

Indicate the start of the interval `nm`.
"""
macro tic(args...)
    na = length(args)
    if na != 1
        throw(ArgumentError("wrong number of arguments in @tic"))
    end
    ex = args[1]
    if !isa(ex, Symbol)
        throw(ArgumentError("need a name argument to @tic"))
    end
    return _tic(ex)
end

# `@toc` helper
function _toc(nm)
    exti = Symbol("tictoc__", nm)
    @static if tictoc_track_memory
        quote
            global $(exti)
            $(exti).time += time_ns() - $(exti).curr
            $(exti).ncalls += 1
            local diff = Base.GC_Diff(Base.gc_num(), $(exti).mem)
            $(exti).allocd += diff.allocd
            $(exti).gctime += diff.total_time
        end
    else
        quote
            global $(exti)
            $(exti).time += time_ns() - $(exti).curr
            $(exti).ncalls += 1
        end
    end
end

"""
    @toc nm

Indicate the end of the interval `nm`.
"""
macro toc(args...)
    na = length(args)
    if na != 1
        throw(ArgumentError("wrong number of arguments in @toc"))
    end
    ex = args[1]
    if !isa(ex, Symbol)
        throw(ArgumentError("need a name argument to @toc"))
    end
    return _toc(ex)
end

"""
    print_timing_info()

`atexit()` function, writes all information about every interval to
`stdout`.
"""
function print_timing_info()
    println("TicToc timing information")
    @static if tictoc_track_memory
        println("name,ncalls,tottime(ns),allocbytes,gctime")
    else
        println("name,ncalls,tottime(ns)")
    end
    for i in 1:length(timing_info_names)
        @static if tictoc_track_memory
            s = @sprintf(
                "%s,%d,%d,%d,%d",
                timing_info_names[i],
                timing_infos[i].ncalls,
                timing_infos[i].time,
                timing_infos[i].allocd,
                timing_infos[i].gctime
            )
        else
            s = @sprintf(
                "%s,%d,%d",
                timing_info_names[i],
                timing_infos[i].ncalls,
                timing_infos[i].time
            )
        end
        println(s)
    end
end

"""
    tictoc()

Call at program start (only once!) to set up the globals used by the
macros and to register the at-exit callback.
"""
function tictoc()
    global timing_info_names
    for nm in timing_info_names
        exti = Symbol(nm)
        isdefined(@__MODULE__, exti) && continue
        expr = quote
            const $exti = $TimingInfo()
        end
        eval(Expr(:toplevel, expr))
        push!(timing_infos, getfield(@__MODULE__, exti))
    end
    if parse(Int, get(ENV, "TICTOC_PRINT_RESULTS", "0")) == 1
        if !atexit_function_registered[]
            atexit(print_timing_info)
            atexit_function_registered[] = true
        end
    end
    return length(timing_info_names)
end

end # module
