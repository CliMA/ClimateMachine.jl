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
    export_plot(
        z,
        time_data,
        all_data::Array,
        ϕ_all,
        filename;
        xlabel,
        ylabel,
        time_units = "[s]",
        round_digits = 2,
        horiz_layout = false,
        xlims = (:auto, :auto),
    )

Export plot of all variables, or all
available time-steps in `all_data`.
"""
function export_plot(
    z,
    time_data,
    all_data::Array,
    ϕ_all,
    filename;
    xlabel,
    ylabel,
    time_units = "[s]",
    round_digits = 2,
    horiz_layout = false,
    xlims = (:auto, :auto),
)
    ϕ_all isa Tuple || (ϕ_all = (ϕ_all,))
    single_var = ϕ_all[1] == xlabel || length(ϕ_all) == 1
    p = plot()
    for (t, data) in zip(time_data, all_data)
        for ϕ in ϕ_all
            ϕ_string = String(ϕ)
            ϕ_data = data[ϕ_string][:]
            t_label = "t=$(round(t, digits=round_digits)) $time_units"
            label = single_var ? t_label : "$ϕ_string, $t_label"
            args = horiz_layout ? (z, ϕ_data) : (ϕ_data, z)
            plot!(
                args...;
                xlabel = xlabel,
                ylabel = ylabel,
                label = label,
                xlims = xlims,
            )
        end
    end
    savefig(filename)
end

"""
    export_contour(
        z,
        time_data,
        all_data::Array,
        ϕ,
        filename;
        xlabel = "time [s]",
        ylabel = "z [m]",
        label = String(ϕ)
    )

Export contour plots given
 - `z` Array of altitude. Note: this must not include duplicate nodal points.
 - `time_data` array of time data
 - `all_data` an array whose elements are populated by `dict_of_nodal_states`
 - `ϕ` variable to contour
 - `filename` file name to export to.
 - `xlabel` x-label
 - `ylabel` y-label
 - `label` contour labels
"""
function export_contour(
    z,
    time_data,
    all_data::Array,
    ϕ,
    filename;
    xlabel = "time [s]",
    ylabel = "z [m]",
    label = String(ϕ),
)
    ϕ_string = String(ϕ)
    ϕ_data = hcat([data[ϕ_string][:] for data in all_data]...)
    args = (time_data, z, ϕ_data)
    contourf(
        args...;
        xlabel = xlabel,
        ylabel = ylabel,
        label = label,
        c = :viridis,
    )
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
    export_state_plots(
        solver_config,
        all_data,
        time_data,
        output_dir;
        state_types = (Prognostic(), Auxiliary()),
        z = Array(get_z(solver_config.dg.grid)),
        xlims = (:auto, :auto),
    )

Export line plots of states given
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
    z = Array(get_z(solver_config.dg.grid)),
    xlims = (:auto, :auto),
    time_units = "[s]",
    ylabel = "z [m]",
)
    FT = eltype(solver_config.Q)
    mkpath(output_dir)
    for st in state_types
        vs = vars_state(solver_config.dg.balance_law, st, FT)
        for fn in flattenednames(vs)
            base_name = state_prefix(st) * replace(fn, "." => "_")
            file_name = joinpath(output_dir, "$(base_name).png")
            export_plot(
                z,
                time_data,
                all_data,
                (fn,),
                file_name;
                xlabel = fn,
                ylabel = ylabel,
                time_units = time_units,
                round_digits = 5,
                xlims = xlims,
            )
        end
    end
end

"""
    export_state_contours(
        solver_config,
        all_data,
        time_data,
        output_dir;
        state_types = (Prognostic(),),
        xlabel = "time [s]",
        ylabel = "z [m]",
        z = Array(get_z(solver_config.dg.grid; rm_dupes=true)),
    )

Call `export_contour` for every
state variable given `state_types`.
"""
function export_state_contours(
    solver_config,
    all_data,
    time_data,
    output_dir;
    state_types = (Prognostic(),),
    xlabel = "time [s]",
    ylabel = "z [m]",
    z = Array(get_z(solver_config.dg.grid; rm_dupes = true)),
)
    FT = eltype(solver_config.Q)
    mkpath(output_dir)
    for st in state_types
        vs = vars_state(solver_config.dg.balance_law, st, FT)
        for fn in flattenednames(vs)
            base_name = state_prefix(st) * replace(fn, "." => "_")
            filename = joinpath(output_dir, "cnt_$(base_name).png")
            label = string(replace(fn, "." => "_"))
            args = (z, time_data, all_data, fn, filename)
            export_contour(
                args...;
                xlabel = xlabel,
                ylabel = ylabel,
                label = label,
            )
        end
    end
end
