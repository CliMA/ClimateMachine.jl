####
#### Defines list of Application Programming Interface (APIs)
####

apis = [
    "Home" => "APIs/index.md",
    "Driver" => "APIs/Driver/index.md",
    "Atmos" => [
        "AtmosModel" => "APIs/Atmos/AtmosModel.md",
        "Microphysics" => "APIs/Atmos/Microphysics.md",
        "Temperature Profiles" => "APIs/Atmos/TemperatureProfiles.md",
    ],
    "Ocean" =>
        ["Hydrostatic Boussinesq" => "APIs/Ocean/HydrostaticBoussinesq.md"],
    "Land" => [],
    "Common" => [
        "Surface Fluxes" => "APIs/Common/SurfaceFluxes.md",
        "Thermodynamics" => "APIs/Common/Thermodynamics.md",
    ],
    "Arrays" => "APIs/Arrays/Arrays.md",
    "Diagnostics" => [
        "List of variables" => "APIs/Diagnostics/Diagnostics.md",
        "Input/Output" => "APIs/Diagnostics/InputOutput.md",
        "State Check" => "APIs/Diagnostics/StateCheck.md",
    ],
    "Input/Output" => [
        "List of variables" => "APIs/Diagnostics/Diagnostics.md",
        "Input/Output" => "APIs/Diagnostics/InputOutput.md",
        "State Check" => "APIs/Diagnostics/StateCheck.md",
    ],
    "Numerics" => [
        "Meshes" => "APIs/Numerics/Meshes/Mesh.md",
        "SystemSolvers" => "APIs/Numerics/SystemSolvers/SystemSolvers.md",
        "ODESolvers" => "APIs/Numerics/ODESolvers/ODESolvers.md",
        "Balance Law" => "APIs/Numerics/DGMethods/BalanceLawOverview.md",
        "Numerical Fluxes" => "APIs/Numerics/DGMethods/NumericalFluxes.md",
    ],
    "Utilities" => [
        "Artifact Wrappers" => "APIs/Utilities/ArtifactWrappers.md",
        "Variable Templates" => "APIs/Utilities/VariableTemplates.md",
        "Single Stack Utilities" => "APIs/Utilities/SingleStackUtils.md",
        "Checkpoint" => "APIs/Utilities/Checkpoint.md",
        "Tic Toc" => "APIs/Utilities/TicToc.md",
    ],
]
