
export_plot(z, all_data, ϕ_all, filename, ylabel) = nothing
export_plot_snapshot(z, all_data, ϕ_all, filename, ylabel) = nothing

# using Requires
# @init @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
#   using .Plots

"""
    plot_friendly_name(ϕ)

Get plot-friendly string, since many Unicode
characters do not render in plot labels.
"""
function plot_friendly_name(ϕ)
    s = ϕ
    s = replace(s, "ρ" => "rho")
    s = replace(s, "α" => "alpha")
    s = replace(s, "∂" => "partial")
    s = replace(s, "∇" => "nabla")
    return s
end

"""
    export_plot(z, all_data, ϕ_all, filename, ylabel)

Export plot of all variables, or all
available time-steps in `all_data`.
"""
function export_plot(z, all_data, ϕ_all, filename, ylabel)
    ϕ_all isa Tuple || (ϕ_all = (ϕ_all,))
    p = plot()
    for n in 0:(length(keys(all_data)) - 1)
        for ϕ in ϕ_all
            ϕ_string = String(ϕ)
            ϕ_name = plot_friendly_name(ϕ_string)
            ϕ_data = all_data[n][ϕ_string][:]
            plot!(ϕ_data, z, xlabel = ϕ_name, ylabel = ylabel)
        end
    end
    savefig(filename)
end

"""
    export_plot_snapshot(z, all_data, ϕ_all, filename, ylabel)

Export plot of all variables in `all_data`
"""
function export_plot_snapshot(z, all_data, ϕ_all, filename, ylabel)
    ϕ_all isa Tuple || (ϕ_all = (ϕ_all,))
    p = plot()
    for ϕ in ϕ_all
        ϕ_string = String(ϕ)
        ϕ_name = plot_friendly_name(ϕ_string)
        ϕ_data = all_data[ϕ_string][:]
        plot!(ϕ_data, z, xlabel = ϕ_name, ylabel = ylabel)
    end
    savefig(filename)
end

# end
