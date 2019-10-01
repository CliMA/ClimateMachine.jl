#### PlotFuncs

# Provides a set of plot functions that operate on StateVec.

using Requires

@init @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
  using .Plots
  export plot_state, plot_states
end

export nice_string

k_start = 1
k_stop_min = 1000
markershapes = Symbol[:utriangle, :circle, :rect, :star5,
                      :diamond, :hexagon, :cross, :xcross,
                      :dtriangle, :rtriangle, :ltriangle,
                      :pentagon, :heptagon, :octagon,
                      :star4, :star6, :star7, :star8,
                      :vline, :hline, :+, :x]
markershapes_net = Symbol[:hline, :+, :x]
markersize = [6, 4, 2]

function nice_string(name)
  friendly_name = string(name)
  friendly_name = replace(friendly_name, "θ" => "theta")
  friendly_name = replace(friendly_name, "ρ" => "rho")
  friendly_name = replace(friendly_name, "α" => "alpha")
  friendly_name = replace(friendly_name, "∇" => "grad")
  friendly_name = replace(friendly_name, "εδ" => "entr-detr")
  return friendly_name
end

@init @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin

"""
    plot_state(sv::StateVec,
                    grid::Grid,
                    name_s::Symbol = nothing,
                    directory::AbstractString,
                    i = 0,
                    include_ghost = false,
                    xlims::Union{Nothing, Tuple{R, R}} = nothing,
                    ylims::Union{Nothing, Tuple{R, R}} = nothing
                    ) where R

Save the plot variable along the z-direction in `StateVec` given the
grid, `grid`, variable name `name_s`, directory `directory`,
sub-domain `i`, and a `Bool`, `include_ghost`, indicating
to include include or exclude the ghost points.
"""
function plot_state(sv::StateVec,
                    grid::Grid,
                    directory::AbstractString,
                    name_s::Symbol;
                    i = 0,
                    include_ghost = true,
                    filename = nothing,
                    i_Δt = 1,
                    xlims::Union{Nothing, Tuple{R, R}} = nothing,
                    ylims::Union{Nothing, Tuple{R, R}} = nothing
                    ) where R
  domain_range = include_ghost ? over_elems(grid) : over_elems_real(grid)
  k_stop_local = min(length(domain_range), k_stop_min)
  domain_range = domain_range[k_start:k_stop_local]
  x = [grid.zc[k] for k in domain_range]
  name = nice_string(name_s)
  if i==0
    plot()
    for i in over_sub_domains(sv, name_s)
      name_i = nice_string(var_string(sv, name_s, i))
      y = [sv[name_s, k, i] for k in domain_range]
      plot!(y, x, markershapes = markershapes[i], label = name_i)
    end
    plot!(title = name * " vs z", xlabel = name, ylabel = "z")
  else
    name = nice_string(var_string(sv, name_s, i))
    y = [sv[name_s, k, i] for k in domain_range]
    plot(y, x, markershapes = markershapes[i], label = name)
    plot!(title = name * " vs z", xlabel = name, ylabel = "z")
  end
  if xlims != nothing; plot!(xlims = xlims); end
  if ylims != nothing; plot!(ylims = xlims); end
  filename == nothing && (filename = name)
  mkpath(directory)
  savefig(joinpath(directory, filename))
end

"""
    plot_states(sv::StateVec,
                grid::Grid,
                name_ids,
                directory::AbstractString,
                include_ghost = false,
                xlims::Union{Nothing, Tuple{R, R}} = nothing,
                ylims::Union{Nothing, Tuple{R, R}} = nothing
                ) where R

Save the plot variable along the z-direction in `StateVec` given the
grid, `grid`, variable name `name_s`, directory `directory`,
sub-domain `i`, and a `Bool`, `include_ghost`, indicating to
include include or exclude the ghost points.
"""
function plot_states(sv::StateVec,
                     grid::Grid,
                     directory::AbstractString,
                     name_ids;
                     include_ghost = true,
                     filename = nothing,
                     i_Δt = 1,
                     sources = false,
                     xlims::Union{Nothing, Tuple{R, R}} = nothing,
                     ylims::Union{Nothing, Tuple{R, R}} = nothing
                     ) where R
  if sources
    domain_range = over_elems_real(grid)
  else
    domain_range = include_ghost ? over_elems(grid) : over_elems_real(grid)
  end
  k_stop_local = min(length(domain_range), k_stop_min)
  domain_range = domain_range[k_start:k_stop_local]

  x = [grid.zc[k] for k in domain_range]
  plot()
  gm, en, ud, sd, al = allcombinations(DomainIdx(sv))
  if sources
    @inbounds for name_id in name_ids
      @inbounds for i in sd
        source_term_names = unique([s.name for k in domain_range for s in sv[name_id, k, i]])
        data = [Dict(s.name => s.value for s in sv[name_id, k, i]) for k in domain_range]
        @inbounds for s in source_term_names
          y = [x[s] for x in data]
          if any([abs(v) > eps(typeof(v)) for v in y])
            plot!(y, x, label = nice_string(s)*var_suffix(sv, name_id, i), markershapes = markershapes[i], markersize = markersize[i])
          end
        end
        y = [sum([x[s] for s in source_term_names]) for x in data]
        if any([abs(v) > eps(typeof(v)) for v in y])
          plot!(y, x, label = "net_"*nice_string(var_string(sv, name_id, i)), markershapes = markershapes_net[i], markersize = markersize[i])
        end
      end
    end
  else
    @inbounds for name_id in name_ids
      @inbounds for i in over_sub_domains(sv, name_id)
        y = [sv[name_id, k, i] for k in domain_range]
        plot!(y, x, label = nice_string(s)*var_suffix(sv, name_id, i), markershapes = markershapes[i], markersize = markersize[i])
      end
    end
  end
  filename == nothing && (filename = nice_string(name_ids[1]))
  plot!(title = nice_string(filename)*" vs z", xlabel = nice_string(filename), ylabel = "z")
  if xlims != nothing; plot!(xlims = xlims); end
  if ylims != nothing; plot!(ylims = xlims); end
  mkpath(directory)
  savefig(joinpath(directory, filename))
end

end # @require
