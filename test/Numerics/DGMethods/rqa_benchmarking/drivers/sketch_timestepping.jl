## Details above then ...

########
# Option 1
########
model = DryAtmosModel(
    physics = physics,
    boundary_conditions = (5, 6),
    initial_conditions = (ρ = ρ₀ᶜᵃʳᵗ, ρu = ρu⃗₀ᶜᵃʳᵗ, ρe = ρeᶜᵃʳᵗ),
    numerics = (
        flux = RusanovNumericalFlux(),
    ),
    parameters = parameters,
)

linear_model = DryAtmosLinearESDGModel(
    physics = linear_physics,
    boundary_conditions = (5, 6),
    initial_conditions = (ρ = ρ₀ᶜᵃʳᵗ, ρu = ρu⃗₀ᶜᵃʳᵗ, ρe = ρeᶜᵃʳᵗ),
    numerics = (
        flux = RusanovNumericalFlux(),
    ),
    parameters = parameters,
)

dx = min_node_distance(grid.numerical)
cfl = 7.5 # maybe 2
Δt = cfl * dx / 330.0
start_time = 0
end_time = 30 * 86400 # Δt
method = ARK2GiraldoKellyConstantinescu 


simulation = Simulation(
    (model, linear_esdg_model);
    grid = grid,
    timesteppers = (explicit, implicit, )
    timestepper = (method = method, timestep = Δt),
    time        = (start = start_time, finish = end_time),
);

#=
Advantages:
1. Current Framework
2. no extra specification necesary
3. Typically only one timestepping method is used in practice
Disadvantages:
1. Tied into some baked in assumptions about what's explicit and whats implicit
2. Tied to a particular method (here IMEX)
=#

########
# Option 2
########

model = DryAtmosModel(
    physics = physics,
    boundary_conditions = (5, 6),
    initial_conditions = (ρ = ρ₀ᶜᵃʳᵗ, ρu = ρu⃗₀ᶜᵃʳᵗ, ρe = ρeᶜᵃʳᵗ),
    numerics = (
        flux = RusanovNumericalFlux(),
    ),
    parameters = parameters,
)

linear_model = linearize(model)

dx = min_node_distance(grid.numerical)
cfl = 7.5 # maybe 2
Δt = cfl * dx / 330.0
start_time = 0
end_time = 30 * 86400 # Δt
method = ARK2GiraldoKellyConstantinescu
# Explicit(advection, rate = 1), Explicit(soundwave_horizontal, rate = 10), Explicit(radiation, rate = 0.1), Explicit(microphysics, rate = 5)
simulation = Simulation(
    (Explicit(model), Implicit(linear_esdg_model));
    grid = grid,
    timestepper = (method = method, timestep = Δt),
    time        = (start = start_time, finish = end_time),
);

#=
Advantages:
1. Clearer what model is explicit vs implicit
2. Automate construction of linear model
3. Can do error checking or warning
Disadvantages:
1. More code necessary
2. Structception
3. Easier to see how to generalize with multirate, Explicit(model, rate = Fast())
=#

########
# Option 3
########

model = DryAtmosModel(
    physics = physics,
    boundary_conditions = (5, 6),
    initial_conditions = (ρ = ρ₀ᶜᵃʳᵗ, ρu = ρu⃗₀ᶜᵃʳᵗ, ρe = ρeᶜᵃʳᵗ),
    numerics = (
        flux = RusanovNumericalFlux(),
    ),
    parameters = parameters,
)

linear_model = linearize(model)

dx = min_node_distance(grid.numerical)
cfl = 7.5 # maybe 2
Δt = cfl * dx / 330.0
start_time = 0
end_time = 30 * 86400 # Δt

timestepper = IMEX(implicit = linear_model, 
                   explicit = model, 
                   method = ARK2GiraldoKellyConstantinescu,
                   Δt = Δt,
                   start_time = 0,
                   end_time = 30 * 86400)

simulation = Simulation(
    models(timestepper);
    grid = grid,
    timestepper = timestepper,
);

#=
Advantages:
1. Clearer what model is explicit vs implicit
2. Automate construction of linear model
3. Can do error checking or warning
Disadvantages:
1. More code necessary
2. Structception
3. Perhaps too much information at once
4. Redundancy in having models as a part of simulation 
=#
