using JLD2
using Plots

function timeplotter(datadict, keys = Tuple(keys(datadict)); kwargs...)
    tp = timeplot(datadict, keys[1]; kwargs...)
    for k in keys[2:end]
        timeplot!(datadict, k; kwargs...)
    end
    return tp
end
function timeplot(datadict, key; kwargs...)
    data = datadict[key]
    plot(data.dts, data.solvetimes;
        xscale = :log10, yscale = :log10, label = key,
        xlabel = "Timestep size (s)", ylabel = "Solve time (s)", title = "Solve Time v Timestep Size",
        legend = :outertopright, kwargs...)
end
function timeplot!(datadict, key; kwargs...)
    data = datadict[key]
    plot!(data.dts, data.solvetimes; label = key, kwargs...)
end

function rmseplotter(datadict, keys = Tuple(keys(datadict)); kwargs...)
    rp = rmseplot(datadict, keys[1]; kwargs...)
    for k in keys[2:end]
        rmseplot!(datadict, k; kwargs...)
    end
    return rp
end
function rmseplot(datadict, key; kwargs...)
    data = datadict[key]
    plot(data.dts, data.rmse;
        xscale = :log10, yscale = :log10, label = key,
        xlabel = "Timestep size (s)", ylabel = "RMSE", title = "RMSE v Timestep Size",
        legend = :outertopright, kwargs...)
end
function rmseplot!(datadict, key; kwargs...)
    data = datadict[key]
    plot!(data.dts, data.rmse; label = key, kwargs...)
end

function rmsetimeplotter(datadict, keys = Tuple(keys(datadict)); kwargs...)
    rtp = rmsetimeplot(datadict, keys[1]; kwargs...)
    for k in keys[2:end]
        rmsetimeplot!(datadict, k; kwargs...)
    end
    return rtp
end
function rmsetimeplot(datadict, key; kwargs...)
    data = datadict[key]
    plot(data.solvetimes, data.rmse;
        xscale = :log10, yscale = :log10, label = key,
        xlabel = "Solve time (s)", ylabel = "RMSE", title = "RMSE v Solve Time",
        legend = :outertopright, kwargs...)
end
function rmsetimeplot!(datadict, key; kwargs...)
    data = datadict[key]
    plot!(data.solvetimes, data.rmse; label = key, kwargs...)
end