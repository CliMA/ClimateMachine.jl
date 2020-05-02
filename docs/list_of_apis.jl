####
#### Defines list of Application Programming Interface (APIs)
####

apis = Any[
    "Home" => "APIs/index.md",
    "Atmos" => Any[
        "AtmosModel" => "APIs/Atmos/AtmosModel.md"
        "Microphysics" => "APIs/Atmos/Microphysics.md"
    ],
    "Ocean" => Any[],
    "Land" => Any[],
    "Common" => Any["Surface Fluxes" => "APIs/Common/SurfaceFluxes.md"],
    "Arrays" => "APIs/Arrays/Arrays.md",
    "Diagnostics" => Any[
        "List of variables" => "APIs/Diagnostics/Diagnostics.md"
        "Input/Output" => "APIs/Diagnostics/InputOutput.md"
    ],
    "Numerics" => Any[
        "Meshes" => "APIs/Numerics/Meshes/Mesh.md",
        "LinearSolvers" => "APIs/Numerics/LinearSolvers/LinearSolvers.md",
        "ODESolvers" => "APIs/Numerics/ODESolvers/ODESolvers.md",
        "Balance Law" => "APIs/Numerics/DGmethods/BalanceLawOverview.md",
    ],
]
