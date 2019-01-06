mutable struct EveryXSecondsCallback
  starttime_ns
  nexttime
  step
  timetodo
  func
  function EveryXSecondsCallback(func, timetodo)
    starttime_ns = time_ns()
    nexttime = timetodo
    step = 0
    new(starttime_ns, nexttime, step, timetodo, func)
  end
end
function (cb::EveryXSecondsCallback)(initialize::Bool=false)
  if initialize
    cb.starttime = time_ns()
    cb.nextime = timetodo
    cb.step = 0
    try
      cb.func(true)
    catch
    end
    return 0
  end

  currtime_ns = time_ns()
  runtime = (currtime_ns - cb.starttime_ns) * 1e-9

  cb.step += 1
  if runtime < cb.nexttime
    return 0
  else
    cb.nexttime = runtime + cb.timetodo
    return cb.func()
  end
end
function numberofsteps(cb::EveryXSecondsCallback)
  cb.step
end
function totaltime(cb::EveryXSecondsCallback)
  currtime_ns = time_ns()
  (currtime_ns - cb.starttime_ns) * 1e-9
end
