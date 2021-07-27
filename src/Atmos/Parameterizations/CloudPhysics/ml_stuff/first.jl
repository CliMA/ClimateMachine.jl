import Pkg
# Pkg.add("StatsBase")
Pkg.add("Cloudy")#master
# Import modules
using Distributions  # probability distributions and associated functions
# using StatsBase
using LinearAlgebra
# using StatsPlots
# using Plots
using JLD2
using Random

# Import Calibrate-Emulate-Sample modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage
using EnsembleKalmanProcesses.DataStorage

# Import the module that runs Cloudy
include("/home/skadakia/clones/ClimateMachine.jl/test/Atmos/Parameterizations/AerosolActivation/runtests.jl")

rng_seed = 41
Random.seed!(rng_seed)

# homedir = pwd()
# figure_save_directory = homedir*"/output/"
# data_save_directory = homedir*"/output/"
# if ~isdir(figure_save_directory)
#     mkdir(figure_save_directory)
# end
# if ~isdir(data_save_directory)
#     mkdir(data_save_directory)
# end

###
###  Define the (true) parameters and their priors
###

# Define the parameters that we want to learn
# We assume that the true particle mass distribution is a Gamma 
# distribution with parameters N0_true, θ_true, k_true
par_names = ["mean_hygro", "gamma"]
n_par = length(par_names)
mean_hygro_true = 4  # number of particles (scaling factor for Gamma distribution)
gamma_true = 1  # scale parameter of Gamma distribution
# k_true = 9  # shape parameter of Gamma distribution

# Note that dist_true is a Cloudy distribution, not a Distributions.jl 
# distribution
ϕ_true = [mean_hygro_true, gamma_true]
dist_true = ParticleDistributions.GammaPrimitiveParticleDistribution(ϕ_true...)


###
###  Define priors for the parameters we want to learn
###

# Define constraints
lbound_mean_hygro = 0.4 * mean_hygro_true
lbound_gamma = 1.0e-1
# lbound_k = 1.0e-4
c1 = bounded_below(lbound_N0)
c2 = bounded_below(lbound_θ)
c3 = bounded_below(lbound_k)
constraints = [[c1], [c2], [c3]]

# We choose to use normal distributions to represent the prior distributions of
# the parameters in the transformed (unconstrained) space. i.e log coordinates
d1 = Parameterized(Normal(4.5, 1.0))  #truth is 5.19
d2 = Parameterized(Normal(0.0, 2.0))  #truth is 0.378
d3 = Parameterized(Normal(-1.0, 1.0)) #truth is -2.51
distributions = [d1, d2, d3]

priors = ParameterDistribution(distributions, constraints, par_names)


###
###  Define the data from which we want to learn the parameters
###

data_names = ["M0", "M1", "M2"]
moments = [0.0, 1.0, 2.0]
n_moments = length(moments)


###
###  Model settings
###

# Collision-coalescence kernel to be used in Cloudy
coalescence_coeff = 1/3.14/4/100
kernel_func = x -> coalescence_coeff
kernel = CoalescenceTensor(kernel_func, 0, 100.0)

# Time period over which to run Cloudy
tspan = (0., 1.0)

###
###  Generate (artificial) truth samples
###

model_settings_true = ModelSettings(kernel, dist_true, moments, tspan)
G_t = run_dyn_model(ϕ_true, model_settings_true)
n_samples = 100
y_t = zeros(length(G_t), n_samples)
# In a perfect model setting, the "observational noise" represents the 
# internal model variability. Since Cloudy is a purely deterministic model, 
# there is no straightforward way of coming up with a covariance structure 
# for this internal model variability. We decide to use a diagonal 
# covariance, with entries (variances) largely proportional to their 
# corresponding data values, G_t
Γy = convert(Array, Diagonal([100.0, 5.0, 30.0]))
μ = zeros(length(G_t))

# Add noise
for i in 1:n_samples
    y_t[:, i] = G_t .+ rand(MvNormal(μ, Γy))
end

truth = Observations.Obs(y_t, Γy, data_names)
truth_sample = truth.mean


###
###  Calibrate: Ensemble Kalman Inversion
###

N_ens = 50 # number of ensemble members
N_iter = 8 # number of EKI iterations
# initial parameters: N_par x N_ens
initial_par = construct_initial_ensemble(priors, N_ens; rng_seed)
ekiobj = EnsembleKalmanProcess(initial_par, truth_sample, truth.obs_noise_cov,
                               Inversion(), Δt=0.1)

# Initialize a ParticleDistribution with dummy parameters. The parameters 
# will then be set within `run_dyn_model`
dummy = ones(n_par)
dist_type = ParticleDistributions.GammaPrimitiveParticleDistribution(dummy...)
model_settings = DynamicalModel.ModelSettings(kernel, dist_type, moments, 
                                              tspan)
# EKI iterations
for n in 1:N_iter
    θ_n = get_u_final(ekiobj)
    # Transform parameters to physical/constrained space
    ϕ_n = mapslices(x -> transform_unconstrained_to_constrained(priors, x), θ_n; dims=1)
    # Evaluate forward map
    G_n = [run_dyn_model(ϕ_n[:, i], model_settings) for i in 1:N_ens]
    G_ens = hcat(G_n...)  # reformat
    EnsembleKalmanProcessModule.update_ensemble!(ekiobj, G_ens)
end

# EKI results: Has the ensemble collapsed toward the truth?
θ_true = transform_constrained_to_unconstrained(priors, ϕ_true)
println("True parameters (unconstrained): ")
println(θ_true)

println("\nEKI results:")
println(mean(get_u_final(ekiobj), dims=2))

u_stored= get_u(ekiobj, return_array=false)
g_stored= get_g(ekiobj, return_array=false)
# @save data_save_directory*"parameter_storage_eki.jld2" u_stored
# @save data_save_directory*"data_storage_eki.jld2" g_stored

#plots
gr(size=(1800, 600))

u_init = get_u_prior(ekiobj)
# for i in 1:N_iter
#     u_i = get_u(ekiobj, i)

#     p1 = plot(u_i[1,:], u_i[2,:], seriestype=:scatter,
#               xlims = extrema(u_init[1,:]), ylims=extrema(u_init[2,:]))
#     plot!(p1, [θ_true[1]], xaxis="u1", yaxis="u2", seriestype="vline",
#           linestyle=:dash, linecolor=:red, label=false,
#           title="EKI iteration = " * string(i))
#     plot!(p1, [θ_true[2]], seriestype="hline", linestyle=:dash,
#           linecolor=:red, label="optimum")

#     p2 = plot(u_i[2,:], u_i[3,:], seriestype=:scatter,
#               xlims=extrema(u_init[2,:]), ylims=extrema(u_init[3,:]))
#     plot!(p2, [θ_true[2]], xaxis="u1", yaxis="u2", seriestype="vline",
#           linestyle=:dash, linecolor=:red, label=false,
#           title="EKI iteration = " * string(i))
#     plot!(p2, [θ_true[3]], seriestype="hline", linestyle=:dash,
#           linecolor=:red, label="optimum")

#     p3 = plot(u_i[3,:], u_i[1,:], seriestype=:scatter,
#               xlims=extrema(u_init[3,:]), ylims=extrema(u_init[1,:]))
#     plot!(p3, [θ_true[3]], xaxis="u1", yaxis="u2",
#           seriestype="vline", linestyle=:dash, linecolor=:red, label=false,
#           title="EKI iteration = " * string(i))
#     plot!(p3, [θ_true[1]], seriestype="hline", linestyle=:dash,
#           linecolor=:red, label="optimum")

#     p = plot(p1, p2, p3, layout=(1,3))
#     display(p)
#     sleep(0.5)
# end