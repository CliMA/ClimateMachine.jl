using Plots
using KernelAbstractions: CPU
using ClimateMachine.MPIStateArrays: array_device
using ClimateMachine.BalanceLaws: Prognostic, Auxiliary, GradientFlux

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
    export_plot(z, all_data, ϕ_all, filename;
                xlabel, ylabel, time_data, round_digits, horiz_layout)

Export plot of all variables, or all
available time-steps in `all_data`.
"""
function export_plot(
    z,
    all_data::Array,
    ϕ_all,
    filename;
    xlabel,
    ylabel,
    time_data,
    round_digits = 2,
    horiz_layout = false,
)
    ϕ_all isa Tuple || (ϕ_all = (ϕ_all,))
    single_var = ϕ_all[1] == xlabel && length(ϕ_all) == 1
    p = plot()
    # Lims hack
    # See https://github.com/JuliaPlots/Plots.jl/issues/3100
    ε = sqrt(eps(eltype(time_data)))
    zero_variance = all(
        all(abs.(diff(data[String(ϕ)][:])) .< ε)
        for ϕ in ϕ_all, data in all_data
    )
    non_zero_data =
        all(maximum(data[String(ϕ)][:]) > ε for ϕ in ϕ_all, data in all_data)
    prescribe_lims = zero_variance && non_zero_data
    prescribe_lims && @warn "Plot Limits have been manually adjusted"

    for (t, data) in zip(time_data, all_data)
        for ϕ in ϕ_all
            ϕ_string = String(ϕ)
            ϕ_name = plot_friendly_name(ϕ_string)
            ϕ_data = data[ϕ_string][:]
            t_label = "t=$(round(t, digits=round_digits))"
            label = single_var ? t_label : "$ϕ_string, $t_label"
            args = horiz_layout ? (z, ϕ_data) : (ϕ_data, z)
            if prescribe_lims && !horiz_layout
                Δϕ_max = maximum(ϕ_data) - minimum(ϕ_data)
                ϕ_mean = sum(ϕ_data) / length(ϕ_data)
                xlims_min = ϕ_mean - ϕ_mean * 0.01
                xlims_max = ϕ_mean + ϕ_mean * 0.01
                plot!(
                    args...;
                    xlabel = xlabel,
                    ylabel = ylabel,
                    label = label,
                    xlims = (xlims_min, xlims_max),
                )
            else
                plot!(args...; xlabel = xlabel, ylabel = ylabel, label = label)
            end
        end
    end
    savefig(filename)
end

function save_binned_surface_plots(
    x,
    y,
    z,
    title,
    filename,
    n_plots = (3, 3),
    z_label_prefix = "z",
    n_digits = 5,
)
    n_z_partitions = prod(n_plots)
    z_min_global = min(z...)
    z_max_global = max(z...)
    Δz = (z_max_global - z_min_global) / n_z_partitions
    z_min = ntuple(i -> z_min_global + (i - 1) * Δz, n_z_partitions)
    z_max = ntuple(i -> z_min_global + (i) * Δz, n_z_partitions)
    p = []
    for i in 1:n_z_partitions
        mask = z_min[i] .<= z .<= z_max[i]
        x_i = x[mask]
        y_i = y[mask]
        sz_min = string(z_min[i])[1:min(n_digits, length(string(z_min[i])))]
        sz_max = string(z_max[i])[1:min(n_digits, length(string(z_max[i])))]
        p_i = plot(
            x_i,
            y_i,
            title = "$(title), in ($sz_min, $sz_max)",
            seriestype = :scatter,
            markersize = 5,
        )
        push!(p, p_i)
    end
    plot(p..., layout = n_plots, legend = false)
    savefig(filename)
end;

state_prefix(::Prognostic) = "prog_"
state_prefix(::Auxiliary) = "aux_"
state_prefix(::GradientFlux) = "grad_flux_"

"""
    plot_results(solver_config, all_data, time_data, output_dir)

Exports plots of states given
 - `solver_config` a `SolverConfiguration`
 - `all_data` an array of dictionaries, returned from `dict_of_nodal_states`
 - `time_data` an array of time values
 - `output_dir` output directory
"""
function export_state_plots(
    solver_config,
    all_data,
    time_data,
    output_dir;
    state_types = (Prognostic(), Auxiliary()),
    z = get_z(solver_config.dg.grid),
)
    FT = eltype(solver_config.Q)
    z = array_device(solver_config.Q) isa CPU ? z : Array(z)
    mkpath(output_dir)
    for st in state_types
        vs = vars_state(solver_config.dg.balance_law, st, FT)
        for fn in flattenednames(vs)
            file_name = state_prefix(st) * replace(fn, "." => "_")
            export_plot(
                z,
                all_data,
                (fn,),
                joinpath(output_dir, "$(file_name).png");
                xlabel = fn,
                ylabel = "z [m]",
                time_data = time_data,
                round_digits = 5,
            )
        end
    end
end
