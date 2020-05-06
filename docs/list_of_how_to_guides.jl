####
#### Defines list of how-to-guides
####

how_to_guides = Any[
    "Home" => "HowToGuides/Contributing/CONTRIBUTING.md",
    "Arrays" => Any[
    # "Home" => "HowToGuides/Arrays/index.md"
    ],
    "Common" => Any[
        "SurfaceFluxes" => "HowToGuides/Common/SurfaceFluxes.md",
        "MoistThermodynamics" => "HowToGuides/Common/MoistThermodynamics.md",
    ],
    "Atmos" => Any[
        "AtmosModel" => "HowToGuides/Atmos/AtmosModel.md",
        "Rising Bubble LES" => "HowToGuides/Atmos/Model/risingbubble.md",
        "Microphysics" => "HowToGuides/Atmos/Microphysics.md",
        "Turbulence" => "HowToGuides/Atmos/Model/turbulence.md",
        "Tracers" => "HowToGuides/Atmos/Model/tracers.md",
    ],
    "Ocean" => Any[
    # "Home" => "HowToGuides/Ocean/index.md"
    ],
    "Land" => Any[
    # "Home" => "HowToGuides/Land/index.md"
    ],
    "Diagnostics" => Any["Variables" => "HowToGuides/Diagnostics/DiagnosticVariables.md"],
    "Numerics" => Any[
        "Meshes" => Any[
        # "Home" => "HowToGuides/Numerics/Meshes/index.md",
        ],
        "DG methods" => Any["How to make a balance law" => "HowToGuides/Numerics/DGmethods/how_to_make_a_balance_law.md",],
        "ODE Solvers" => Any["Time-integration" => "HowToGuides/Numerics/ODESolvers/Timestepping.md",],
        "Linear Solvers" => Any["Iterative Solvers" => "HowToGuides/Numerics/LinearSolvers/IterativeSolvers.md",],
    ],
]
