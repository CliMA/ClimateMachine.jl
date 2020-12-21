####
#### Defines list of how-to-guides
####

how_to_guides = Any[
    "Common" => Any[
        "Thermodynamics" => "HowToGuides/Common/Thermodynamics.md",
        "Universal Functions" => "HowToGuides/Common/UniversalFunctions.md",
    ],
    "Atmos" => Any[
        "Temperature profiles" => "HowToGuides/Atmos/TemperatureProfiles.md",
        "Reference profiles" => "HowToGuides/Atmos/AtmosReferenceState.md",
        "Moisture model" => "HowToGuides/Atmos/MoistureModelChoices.md",
        "Precipitation model" => "HowToGuides/Atmos/PrecipitationModelChoices.md",
        "How to create an experiment with moisture and precipitation" => "HowToGuides/Atmos/MoistureAndPrecip.md",
    ],
    "Ocean" => Any[
    # "Home" => "HowToGuides/Ocean/index.md"
    ],
    "Land" => Any[
    # "Home" => "HowToGuides/Land/index.md"
    ],
    "Numerics" => Any[
        "Meshes" => Any[
        # "Home" => "HowToGuides/Numerics/Meshes/index.md",
        ],
        "DG methods" => Any["How to make a balance law" => "HowToGuides/Numerics/DGMethods/how_to_make_a_balance_law.md",],
        "ODE Solvers" => Any["Time-integration" => "HowToGuides/Numerics/ODESolvers/Timestepping.md",],
        "System Solvers" => Any["Iterative Solvers" => "HowToGuides/Numerics/SystemSolvers/IterativeSolvers.md",],
    ],
    "Diagnostics" => Any["Using Diagnostics" => "HowToGuides/Diagnostics/UsingDiagnostics.md",],
]
