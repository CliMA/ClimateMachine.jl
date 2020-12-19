using FileIO
using JLD2
using ClimateMachine
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "docs", "plothelpers.jl"));

output_dir = @__DIR__;

mkpath(output_dir);

stored_data = load(string(output_dir,"/sbl_edmf.jld2"))

# Get z-coordinate
z = stored_data["z"]
dons_arr = stored_data["dons_arr"]
time_data = stored_data["time_data"]

println(dons_arr[1].keys)
export_plot(
 z,
 time_data,
 dons_arr,
 ("ρu[1]",),
 joinpath(output_dir, "mom_u_plot_exp.png");
 xlabel = "ρu (kg / m^2 s)",
 ylabel = "z (m)",
 time_units = "(seconds)",
)
export_plot(
 z,
 time_data,
 dons_arr,
 ("ρu[2]",),
 joinpath(output_dir, "mom_v_plot_exp.png");
 xlabel = "ρv (kg / m^2 s)",
 ylabel = "z (m)",
 time_units = "(seconds)",
)
export_plot(
 z,
 time_data,
 dons_arr,
 ("ρu[3]",),
 joinpath(output_dir, "mom_w_plot_exp.png");
 xlabel = "ρw (kg / m^2 s)",
 ylabel = "z (m)",
 time_units = "(seconds)",
)
# Temperature and Buoyancy
export_plot(
 z,
 time_data,
 dons_arr,
 ("turbconv.updraft[1].ρaθ_liq",),
 joinpath(output_dir, "rhoathetal_upd_plot_exp.png");
 xlabel = "ρaθl_u (kg K / m^3)",
 ylabel = "z (m)",
 time_units = "(seconds)",
)
export_plot(
 z,
 time_data,
 dons_arr,
 ("moisture.θ_v",),
 joinpath(output_dir, "theta_v_plot_exp.png");
 xlabel = "θv_e (K)",
 ylabel = "z (m)",
 time_units = "(seconds)",
)
export_plot(
 z,
 time_data,
 dons_arr,
 ("turbconv.environment.buoyancy",),
 joinpath(output_dir, "env_buoy_plot_exp.png");
 xlabel = "env buoy",
 ylabel = "z (m)",
 time_units = "(seconds)",
)
# TKE
export_plot(
 z,
 time_data,
 dons_arr,
 ("turbconv.environment.ρatke",),
 joinpath(output_dir, "env_tke_plot_exp.png");
 xlabel = "ρaTKE (kg / m^1 s^2)",
 ylabel = "z (m)",
 time_units = "(seconds)",
)
export_plot(
 z,
 time_data,
 dons_arr,
 ("turbconv.environment.K_m",),
 joinpath(output_dir, "env_eddy_diff.png");
 xlabel = "K_m (m^2 s^-1)",
 ylabel = "z (m)",
 time_units = "(seconds)",
)
export_plot(
 z,
 time_data,
 dons_arr,
 ("turbconv.environment.l_mix",),
 joinpath(output_dir, "env_mix_length.png");
 xlabel = "l_mix (m)",
 ylabel = "z (m)",
 time_units = "(seconds)",
)
export_plot(
 z,
 time_data,
 dons_arr,
 ("turbconv.environment.buoy_prod",),
 joinpath(output_dir, "env_buoy_prod.png");
 xlabel = "w'b'_0",
 ylabel = "z (m)",
 time_units = "(seconds)",
)
# Covariances
export_plot(
 z,
 time_data,
 dons_arr,
 ("turbconv.environment.ρaθ_liq_cv",),
 joinpath(output_dir, "env_thetal_cov_plot_exp.png");
 xlabel = "ρaθ_liq_cv (kg K^2/ m^3)",
 ylabel = "z (m)",
 time_units = "(seconds)",
)
# Updraft variables
export_plot(
 z,
 time_data,
 dons_arr,
 ("turbconv.updraft[1].ρa",),
 joinpath(output_dir, "a_upd_plot_exp.png");
 xlabel = "ρa_u (kg / m^3)",
 ylabel = "z (m)",
 time_units = "(seconds)",
)
export_plot(
 z,
 time_data,
 dons_arr,
 ("turbconv.updraft[1].ρaw",),
 joinpath(output_dir, "rhoaw_upd_plot_exp.png");
 xlabel = "ρaw_u (kg / m^2 s)",
 ylabel = "z (m)",
 time_units = "(seconds)",
)

export_plot(
 z,
 time_data,
 dons_arr,
 ("turbconv.updraft[1].E_dyn",),
 joinpath(output_dir, "ent_dyn_plot_exp.png");
 xlabel = "E_dyn (kg / m^2 s)",
 ylabel = "z (m)",
 time_units = "(seconds)",
)
export_plot(
 z,
 time_data,
 dons_arr,
 ("turbconv.updraft[1].Δ_dyn",),
 joinpath(output_dir, "det_dyn_plot_exp.png");
 xlabel = "Δ_dyn (kg / m^2 s)",
 ylabel = "z (m)",
 time_units = "(seconds)",
)
export_plot(
 z,
 time_data,
 dons_arr,
 ("turbconv.updraft[1].E_trb",),
 joinpath(output_dir, "ent_trb_plot_exp.png");
 xlabel = "E_trb (kg / m^2 s)",
 ylabel = "z (m)",
 time_units = "(seconds)",
)
