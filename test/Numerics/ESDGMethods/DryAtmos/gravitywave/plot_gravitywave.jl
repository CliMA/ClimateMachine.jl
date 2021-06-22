include("gravitywave.jl")
include("../../diagnostics.jl")

using FileIO
using JLD2: @load
using PyPlot
using PGFPlotsX
using LaTeXStrings
using Printf
using ClimateMachine.VariableTemplates

rcParams!(PyPlot.PyDict(PyPlot.matplotlib."rcParams"))

function gw_plots(datadir=joinpath("esdg_output", "gravitywave"))
  gw_plot_convergence(datadir)
  gw_plot_tht_perturbation(datadir)
end

gw_diagnostic_vars(FT) = @vars(δT::FT, w::FT)
function gw_nodal_diagnostics!(atmos, diag::Vars, state::Vars, aux::Vars, coord)
  FT = eltype(state)
  _MSLP::FT = MSLP(param_set)
  _R_d::FT = R_d(param_set)
  _cp_d::FT = cp_d(param_set)
  
  ρ = state.ρ
  ρu = state.ρu
  ρe = state.ρe
  Φ = aux.Φ

  w = ρu[2] / ρ

  p = pressure(ρ, ρu, ρe, Φ)
  T = p / (_R_d * ρ)
  #diag.θ = T * (_MSLP / p) ^ (_R_d / _cp_d)
  diag.δT = T - aux.ref_state.T
  diag.w = w
end

function gw_plot_convergence(datadir=joinpath("esdg_output", "gravitywave"))
  for (root, dir, files) in walkdir(datadir)

    files = filter(s->endswith(s, "jld2"), files)
    nfiles = length(files)
    nfiles == 0 && continue

    any(occursin.("convergence", files)) || continue
    
    datafile = files[end]
    @show datafile

    @load joinpath(root, datafile) convergence_data

    @pgf begin
      plotsetup = {
                xlabel = "Δx [km]",
                grid = "major",
                xmode = "log",
                ymode = "log",
                xticklabel="{\\pgfmathparse{exp(\\tick)/1000}\\pgfmathprintnumber[fixed,precision=3]{\\pgfmathresult}}",
                #xmax = 1,
                xtick = convergence_data[2].avg_dx,
                #ymin = 10. ^ -10 / 5,
                #ymax = 5,
                #ytick = 10. .^ -(0:2:10),
                legend_pos="south east",
                group_style= {group_size="2 by 2",
                              vertical_sep="2cm",
                              horizontal_sep="2cm"},
              }
     
      Ns = sort(collect(keys(convergence_data)))[1:end-1]
      s2title = Dict(1 => L"\rho",
                     2 => L"\rho u",
                     3 => L"\rho w",
                     5 => L"\rho e")

      for norm in (:l2, :linf)
        ylabel = norm === :l2 ?
                 L"L_{2}" * " error" :
                 L"L_{\infty}" * " error"
        fig = GroupPlot(plotsetup)
        for s in (1, 2, 3, 5)
          labels = []
          plots = []
          for N in Ns
            dxs = convergence_data[N].avg_dx
            if norm === :l2
              errs = convergence_data[N].l2_errors_state[:, s]
              rates = convergence_data[N].l2_rates_state[:, s]
            else
              errs = convergence_data[N].linf_errors_state[:, s]
              rates = convergence_data[N].linf_rates_state[:, s]
            end
            coords = Coordinates(dxs, errs)
            plot = PlotInc({}, coords)
            push!(plots, plot)
            push!(labels, "N$N " * @sprintf("(%.2f)", rates[end]))
          end
          legend = Legend(labels)
          push!(fig, {title=s2title[s], ylabel=ylabel}, plots..., legend)
        end
        savepath = joinpath(root, "gw_convergence_$(norm).pdf")
        pgfsave(savepath, fig)
      end
    end
  end
end

function gw_plot_tht_perturbation(datadir)
  linedata = Dict()
  for (root, dir, files) in walkdir(datadir)
    files = filter(s->endswith(s, "jld2"), files)
    nfiles = length(files)
    nfiles == 0 && continue

    any(occursin.("step", files)) || continue
    
    files = sort(files)
    datafile = files[end]
    data = load(joinpath(root, datafile))

    model = data["model"]
    state_prognostic = data["state_prognostic"]
    state_exact = data["state_exact"]
    state_auxiliary = data["state_auxiliary"]
    vgeo = data["vgeo"]


    state_diagnostic = create_diagnostic_state(gw_diagnostic_vars, state_prognostic)
    nodal_diagnostics!(state_diagnostic,
                       gw_nodal_diagnostics!,
                       gw_diagnostic_vars,
                       model,
                       state_prognostic,
                       state_auxiliary,
                       vgeo)
    
    state_diagnostic_exact = create_diagnostic_state(gw_diagnostic_vars, state_prognostic)
    nodal_diagnostics!(state_diagnostic_exact,
                       gw_nodal_diagnostics!,
                       gw_diagnostic_vars,
                       model,
                       state_exact,
                       state_auxiliary,
                       vgeo)

    @show datafile
    
    dim = 2
    N = data["N"]
    K = data["K"]
    vgeo = data["vgeo"]
    x, z, δT, w = interpolate_equidistant(state_diagnostic, vgeo, dim, N, K)
    x, z, δT_exact, w_exact = interpolate_equidistant(state_diagnostic_exact, vgeo, dim, N, K)

    # convert to km
    x ./= 1e3
    z ./= 1e3
   
    ioff()
    xticks = range(0, 300, length = 7)
    fig, ax = subplots(2, 1, figsize=(14, 14))
    
    for a in ax
      a.set_xlim([xticks[1], xticks[end]])
      a.set_xticks(xticks)
      a.set_xlabel(L"x" * " [km]")
      a.set_ylabel(L"z" * " [km]")
      a.set_aspect(15)
    end

    ll = 0.0036 / 100
    sl = 0.0006 / 100
    levels = vcat(-ll:sl:-sl, sl:sl:ll)
    ax[1].set_title("T perturbation [K]")
    norm = matplotlib.colors.TwoSlopeNorm(vmin=levels[1], vcenter=0, vmax=levels[end])
    cset = ax[1].contourf(x', z', δT', cmap=ColorMap("PuOr"), levels=levels, norm=norm)
    ax[1].contour(x', z', δT_exact', levels=levels, colors=("k",))
    cbar = colorbar(cset, ax = ax[1])
    
    #levels = 10
    ax[2].set_title("w [m/s]")
    #norm = matplotlib.colors.TwoSlopeNorm(vmin=levels[1], vcenter=0, vmax=levels[end])
    cset = ax[2].contourf(x', z', w', cmap=ColorMap("PuOr"), levels=levels)
    ax[2].contour(x', z', w_exact', levels=levels, colors=("k",))
    cbar = colorbar(cset, ax = ax[2])


    tight_layout()
    savefig(joinpath(root, "gw_T_perturbation.pdf"))
    close(fig)
    
    k = findfirst(z[1, :] .> 5)
    x_k = x[:, k]
    w_k = w[:, k]
    w_exact_k = w_exact[:, k]
    linedata[(N, K...)] = (x_k, w_k, w_exact_k)
  end


  @pgf begin

    fig = @pgf GroupPlot({group_style= {group_size="1 by 2", vertical_sep="1.5cm"},
                         xmin=0,
                         xmax= 300})
    x, w, w_exact = linedata[(3, 40, 10)]
    p1 = Plot({color="blue"}, Coordinates(x, w))
    p2 = Plot({}, Coordinates(x, w_exact))
    push!(fig, {xlabel="x [km]",
                ylabel="w [m/s]",
                width="10cm",
                height="5cm"},
               p1, p2)
    x, w, w_exact = linedata[(3, 40, 10)]
    p1 = Plot({color="blue"}, Coordinates(x, w))
    p2 = Plot({}, Coordinates(x, w_exact))
    push!(fig, {xlabel="x [km]",
                ylabel="w [m/s]",
                width="10cm",
                height="5cm"},
               p1, p2)
  end
  pgfsave(joinpath(datadir, "gw_line.pdf"), fig)
end
