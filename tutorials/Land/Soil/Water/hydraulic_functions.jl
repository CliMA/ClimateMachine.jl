# # Hydraulic functions and soil parameters

# This tutorial shows how to specify and explore the hydraulic functions
# used in Richard's equation and how to choose a soil type. In particular,
# we show how to choose the formalism for matric potential and hydraulic
# conductivity, and how to make the hydraulic conductivity account for
# the presence of ice as well as how the temperature dependence of the
# viscosity of liquid water.

# # Preliminary setup

# - load external packages

using Plots

# - load ClimateMachine modules

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SoilWaterParameterizations

FT = Float32
# # Choose general soil parameters for your soil type.
# For the water equation, the user needs to choose a porosity,
#  and a specific storage. All values
# are given in SI/mks units. Note that we neglect the residual pore
# space in the ClimateMachine model. Below we choose parameters for
# sandy loam (Bonan, 2019, Chapter 8, page 120), except for that of
# specific storage.
ν = FT(0.41)
S_s = FT(1e-3)

# # van Genuchten formalism
# ClimateMachine's Land model use a constrained van Genuchten (1980)
# formalism with two free parameters, `α` and `n`. The third parameter
# is computed from `n`. Of these, only `α` carries units, of inverse meters.
# These choices govern both the moisture dependency
# of the hydraulic conductivity and the expression for the matric potential.
# Note that they also are a function of the soil type!
# Here we also choose to make the conductivity independent of the volumetric
# ice fraction and temperature.
vg_α = FT(7.5)
vg_n = FT(1.89)
Ksat_vg = FT(4.42 / (3600 * 100))

viscosity_factor = ConstantViscosity{FT}()
impedance_factor = NoImpedance{FT}()
moisture_factor = MoistureDependent{FT}()
hydraulics = vanGenuchten{FT}(α = vg_α, n = vg_n)

# In the Climate Machine code, we make use of a feature of Julia called
# multiple dispatch. With multiple dispatch, a function  can have
# multiple methods (ways of executing), depending on the type of the
# variables passed in. Which method is executed is decided in the moment,
# when a particular set of arugments is passed and the function is called.
# For example, we have defined a new type of object
# in the ClimateMachine code for the `viscosity_factor`, the `impedance_factor`,
# the `hydraulics` model, and the `moisture_factor`, and based on the
# choices the user makes for these, the correct conductivity and matric potential
# value will be returned - for that choice of types.

# As one concrete example:
# `hydraulics` is of type vanGenuchten{Float32} based on our choice of FT:
#    `julia> typeof(hydraulics)`
#    `vanGenuchten{Float32}`
# and the supertype of vanGenucthen{Float32} is
#    `julia> supertype(vanGenuchten{FT})`
#    `AbstractHydraulicsModel{Float32}`
# The function `matric_potential` will execute different methods
# depending on if we pass a hydraulics model of type vanGenuchten or
# e.g. BrooksCorey! In both cases, it will return the correct value
# for `ψ`. The same is true for the `hydraulic_conductivity` function,
# except it also dispatches on the type of the impedance factor and
# viscosity factor supplied. It returns `K`,
# aside from the factor of `Ksat`, which we need to multiply by.

# This feature is nice because it allows us to have a single function call
# execute different behaviors base on choices the user makes at the
# driver level, without changing any source code.
# One byproduct of this flexibility is that the function `hydraulic_conductivity`
# requires all the arguments it could possibly need passed to it, which is
# why here we must supply a value `T` and `θ_ice`, even though they are not used.
T = FT(0.0)
θ_ice = FT(0.0)
# an array of values for the effective liquid saturation
S_l = FT.(0.01:0.01:0.99)

K = Ksat_vg .* hydraulic_conductivity.(Ref(impedance_factor), Ref(viscosity_factor), Ref(moisture_factor), Ref(hydraulics), Ref(θ_ice), Ref(ν), Ref(T), S_l)
# The matric potential, on the other hand, only depends on the hydraulics model choice and the
# effective saturation. However, it does use multiple dispatch on the `hydraulics` type.
ψ = matric_potential.(Ref(hydraulics), S_l)

# Let's plot these functions.
plot(S_l, log10.(K), xlabel = "effective saturation", ylabel = "Log10(K)", label = "K")
plot(S_l, log10.(-ψ), xlabel = "effective saturation", ylabel = "Log10(|ψ|)", label = "|ψ|")
# The huge range in both `K` and `ψ` as `S_l`, as well as the steep slope in
# `ψ` near saturation and completely dry soil, are the reason why Richard's equation
# is such a challenging numerical problem.

# # van Genuchten formalism with icy impedance and temperature effects
# Let's see how the curves change when we include the effects of temperature
# and ice on the hydraulic conductivity.
viscosity_factor_T = TemperatureDependentViscosity{FT}()
T = FT(300.0)
K_T = Ksat .* hydraulic_conductivity.(Ref(impedance_factor), Ref(viscosity_factor_T), Ref(moisture_factor), Ref(hydraulics), Ref(θ_ice), Ref(ν), Ref(T), S_l)
ice_impedance_I = IceImpedance{FT}()
θ_ice = FT(0.3)
S_l_accounting_for_ice = FT.(0.01:0.01:0.7) # note that the total volumetric water fraction cannot exceed unity.
K_ice = Ksat .* hydraulic_conductivity.(Ref(ice_impedance_I), Ref(viscosity_factor), Ref(moisture_factor), Ref(hydraulics), Ref(θ_ice), Ref(ν), Ref(T), S_l_accounting_for_ice)
plot(S_l, log10.(K), xlabel = "effective saturation", ylabel = "Log10(K)", label = "Base case")
plot!(S_l, log10.(K_T), label = "Temperature Dependent Viscosity")
plot!(S_l_accounting_for_ice, log10.(K_ice), label = "Ice Impedance")


# # Brooks and Corey formalism
# To choose a Brooks and Corey formalism for the hydraulics model,
# the user needs to specify `ψ_b`, the matric potential at saturation,
# a value for `Ksat`, and a constant `m`. Both `ψ_b` and `Ksat` carry units!
# The parameters below are also for sandy loam.
# This would also be used in the matric potential.
ψ_sat = -0.09
mval = 0.228
Ksat_bc = FT(56.28 / (3600 * 100))

hydraulics_bc = BrooksCorey{FT}(ψb = ψ_sat, m = mval)
T = FT(0.0)
θ_ice = FT(0.0)
# an array of values for the effective liquid saturation
S_l = FT.(0.01:0.01:0.99)

K_bc = Ksat_bc .* hydraulic_conductivity.(Ref(impedance_factor), Ref(viscosity_factor), Ref(moisture_factor), Ref(hydraulics_bc), Ref(θ_ice), Ref(ν), Ref(T), S_l)
plot(S_l, log10.(K), xlabel = "effective saturation", ylabel = "Log10(K)", label = "van Genuchten")
plot!(S_l, log10.(K_bc), xlabel = "effective saturation", ylabel = "Log10(K)", label = "Brooks and Corey")

# # Other features
# The user also has the choice of making the conductivity constant by choosing
# a `MoistureIndependent{FT}()` moisture factor. This is useful for debugging!
moisture_factor_nothing = MoistureIndependent{FT}()
K_constant = Ksat .* hydraulic_conductivity.(Ref(impedance_factor), Ref(viscosity_factor), Ref(moisture_factor_nothing), Ref(hydraulics), Ref(θ_ice), Ref(ν), Ref(T), S_l)
#    `julia> unique(K_constant)`
#    `1-element Array{Float32,1}:`
#    ` 1.2277777f-5`
# Note that choosing this option does not meant the matric potential
# is constant, as a hydraulics model is still required and employed.


# And, lastly, you might be wondering why we left `Ksat` out of the function
# for `hydraulic_conductivity`. It turns out it is also useful for debugging
# to be able to turn off the diffusion of water, by setting `K_sat = 0`.

