# # Tutorials
# A suite of concrete examples are provided here as a guidance for constructing experiments.
# ## Balance Law
# An introduction on components within a balance law is provided.
# ## Atmos
# Showcase drivers for atmospheric modelling in GCM, single stack, and LES simulations are provided.
# - Dry Idealzed GCM:  The Held-Suarez configuration is used as a guidance to create a driver that runs a simple GCM simulation.
# - Single Element Stack:  The Burgers Equations with a passive tracer is used as a guidance to run the simulation on a single element stack.
# - LES Experiment:  The dry rising bubble case is used as a quigance in creating an LES driver.
# - Topography:  Experiments of dry flow over prescirbe topography (Agnesi mountain) are provided for:
#     * Linear Hydrostatic Mountain
#     * Linear Non-Hydrostatic Mountain
# ## Ocean
# A showcase for Ocean model is still under construction.
# ## Land
# Examples are provided in constructing balance law and solving for fundemental equations in land modelling.
# - Heat:  A tutorial shows how to create a HeatModel to solve the heat equation and visualize the outputs.
# - Soil:  Examples of solving fundemental equations in the soil model are provided.
#     * Hydraulic Functions:  a tutorial to specify the hydraulic function in the Richard's equation.
#     * Soil Heat Equations:  a tutorial for solving the heat equation in the soil.
#     * Coupled Water and Heat:  a tutorial for solving interactive heat and wateri in the soil model.
# ## Numerics (need to be moved to How-to-Guide)
# - System Solvers:  Two numerical methods to solve the linear system Ax=b are provided.
#     * Conjugate Gradient
#     * Batched Generalized Minimal Residual
# - DG Methods
#     * Filters
# ## Diagnostics
# A diagnostic tool that can
# - generate statistics for MPIStateArrays
# - validate with reference values
# for debugging purposes.
