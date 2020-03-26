using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra
using Interpolations
using DelimitedFiles
using CLIMA
using CLIMA.Atmos
using CLIMA.ConfigTypes
using CLIMA.Diagnostics
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates
using Dierckx
using CLIMA.Parameters
const clima_dir = dirname(pathof(CLIMA))
include(joinpath(clima_dir, "..", "Parameters", "Parameters.jl"))

import CLIMA.DGmethods:
    vars_state,
    vars_aux,
    vars_integrals,
    vars_reverse_integrals,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    integral_load_aux!,
    integral_set_aux!,
    reverse_integral_load_aux!,
    reverse_integral_set_aux!

import CLIMA.DGmethods: boundary_state!
import CLIMA.Atmos: flux_diffusive!

# -------------------- Radiation Model -------------------------- #
vars_state(::RadiationModel, FT) = @vars()
vars_aux(::RadiationModel, FT) = @vars()
vars_integrals(::RadiationModel, FT) = @vars()
vars_reverse_integrals(::RadiationModel, FT) = @vars()

function atmos_nodal_update_aux!(
    ::RadiationModel,
    ::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function preodefun!(::RadiationModel, aux::Vars, state::Vars, t::Real) end
function integral_load_aux!(
    ::RadiationModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) end
function integral_set_aux!(::RadiationModel, aux::Vars, integ::Vars) end
function reverse_integral_load_aux!(
    ::RadiationModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) end
function reverse_integral_set_aux!(::RadiationModel, aux::Vars, integ::Vars) end
function flux_radiation!(
    ::RadiationModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end



# ------------------------ Begin Radiation Model ---------------------- #
"""
  DYCOMSRadiation <: RadiationModel

Stevens et. al (2005) approximation of longwave radiative fluxes in DYCOMS.
Analytical description as a function of the liquid water path and inversion height zᵢ

* Stevens, B. et. al. (2005) "Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine Stratocumulus". Mon. Wea. Rev., 133, 1443–1462, https://doi.org/10.1175/MWR2930.1
"""
struct DYCOMSRadiation{FT} <: RadiationModel
    "mass absorption coefficient `[m^2/kg]`"
    κ::FT
    "Troposphere cooling parameter `[m^(-4/3)]`"
    α_z::FT
    "Inversion height `[m]`"
    z_i::FT
    "Density"
    ρ_i::FT
    "Large scale divergence `[s^(-1)]`"
    D_subsidence::FT
    "Radiative flux parameter `[W/m^2]`"
    F_0::FT
    "Radiative flux parameter `[W/m^2]`"
    F_1::FT
end

vars_aux(m::DYCOMSRadiation, FT) = @vars(Rad_flux::FT)

vars_integrals(m::DYCOMSRadiation, FT) = @vars(attenuation_coeff::FT)
function integral_load_aux!(
    m::DYCOMSRadiation,
    integrand::Vars,
    state::Vars,
    aux::Vars,
)
    FT = eltype(state)
    integrand.radiation.attenuation_coeff = state.ρ * m.κ * aux.moisture.q_liq
end
function integral_set_aux!(m::DYCOMSRadiation, aux::Vars, integrand::Vars)
    integrand = integrand.radiation.attenuation_coeff
    aux.∫dz.radiation.attenuation_coeff = integrand
end

vars_reverse_integrals(m::DYCOMSRadiation, FT) = @vars(attenuation_coeff::FT)
function reverse_integral_load_aux!(
    m::DYCOMSRadiation,
    integrand::Vars,
    state::Vars,
    aux::Vars,
)
    FT = eltype(state)
    integrand.radiation.attenuation_coeff = state.ρ * m.κ * aux.moisture.q_liq
end
function reverse_integral_set_aux!(
    m::DYCOMSRadiation,
    aux::Vars,
    integrand::Vars,
)
    aux.∫dnz.radiation.attenuation_coeff = integrand.radiation.attenuation_coeff
end

function flux_radiation!(
    m::DYCOMSRadiation,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(flux)
    z = altitude(atmos.orientation, aux)
    Δz_i = max(z - m.z_i, -zero(FT))
    # Constants
    upward_flux_from_cloud = m.F_0 * exp(-aux.∫dnz.radiation.attenuation_coeff)
    upward_flux_from_sfc = m.F_1 * exp(-aux.∫dz.radiation.attenuation_coeff)
    free_troposphere_flux =
        m.ρ_i *
        FT(cp_d) *
        m.D_subsidence *
        m.α_z *
        cbrt(Δz_i) *
        (Δz_i / 4 + m.z_i)
    F_rad =
        upward_flux_from_sfc + upward_flux_from_cloud + free_troposphere_flux
    ẑ = vertical_unit_vector(atmos.orientation, aux)
    flux.ρe += F_rad * ẑ
end
function preodefun!(m::DYCOMSRadiation, aux::Vars, state::Vars, t::Real) end
# -------------------------- End Radiation Model ------------------------ #

"""
  Initial Condition for DYCOMS_RF01 LES
@article{doi:10.1175/MWR2930.1,
author = {Stevens, Bjorn and Moeng, Chin-Hoh and Ackerman,
          Andrew S. and Bretherton, Christopher S. and Chlond,
          Andreas and de Roode, Stephan and Edwards, James and Golaz,
          Jean-Christophe and Jiang, Hongli and Khairoutdinov,
          Marat and Kirkpatrick, Michael P. and Lewellen, David C. and Lock, Adrian and
          Maeller, Frank and Stevens, David E. and Whelan, Eoin and Zhu, Ping},
title = {Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine Stratocumulus},
journal = {Monthly Weather Review},
volume = {133},
number = {6},
pages = {1443-1462},
year = {2005},
doi = {10.1175/MWR2930.1},
URL = {https://doi.org/10.1175/MWR2930.1},
eprint = {https://doi.org/10.1175/MWR2930.1}
}
"""
function init_tc!(bl, state, aux, (x, y, z), args...)
    FT = eltype(state)
    spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit, spl_rhoinit, spl_ppiinit, spl_thetainit=args[2]
    # interpolate data
    data_t = FT(spl_tinit(x,y,z))
    data_q = FT(spl_qinit(z))
    data_u = FT(spl_uinit(x,y,z))
    data_v = FT(spl_vinit(x,y,z))
    data_p = FT(spl_pinit(x,y,z))
    data_rho = FT(spl_rhoinit(x,y,z))
    data_pi = FT(spl_ppiinit(x,y,z))
    data_theta = FT(spl_thetainit(x,y,z))
    u = data_u 
    v = data_v
    w = FT(0)
    if (z>17000)
      u =0
      v= 0
    end
    if (data_t<0)
      @info data_t
    end
    q_pt = PhasePartition(data_q)
    ρ = data_rho
    e_int = internal_energy(data_t,q_pt)
    e_kin = FT(1 / 2) * FT((u^2 + v^2 + w^2))
    e_pot = gravitational_potential(bl.orientation, aux)
    E = ρ * total_energy(e_kin,e_pot,data_t,q_pt)
    state.ρ = data_rho
    state.ρu = SVector(ρ * u, ρ * v, FT(0))
    state.ρe = E
    state.moisture.ρq_tot = ρ * data_q
    state.moisture.ρq_liq = FT(0)
    state.moisture.ρq_ice = FT(0)
    return nothing
end

function read_sounding()
    #read in the original squal sounding
    fsounding  = open(joinpath(@__DIR__, "../sounding_gabersek.dat"))
    #fsounding  = open(joinpath(@__DIR__, "../sounding_gabersek_3deg_warmer.dat"))
    sounding = readdlm(fsounding)
    close(fsounding)
    (nzmax, ncols) = size(sounding)
    if nzmax == 0
        error("SOUNDING ERROR: The Sounding file is empty!")
    end
    return (sounding, nzmax, ncols)
end

function spline_int()

  # ----------------------------------------------------
  # GET DATA FROM INTERPOLATED ARRAY ONTO VECTORS
  # This driver accepts data in 6 column format
  # ----------------------------------------------------
  (sounding, _, ncols) = read_sounding()

  # WARNING: Not all sounding data is formatted/scaled
  # the same. Care required in assigning array values
  # height theta qv    u     v     pressure
  zinit, tinit, qinit, u_init, v_init, pinit  =
      sounding[:, 1], sounding[:, 2], 0.001 .* sounding[:, 3], sounding[:, 4], sounding[:, 5], sounding[:, 6]
  t_anom = 2
  RMW = 100000
  X = -810000:5000:810000
  Y = -810000:5000:810000
  theta = zeros(length(X),length(X),length(zinit))
  thetav = zeros(length(X),length(X),length(zinit))
  pressure = zeros(length(X),length(X),length(zinit))
  ppi = zeros(length(X),length(X),length(zinit))
  temp = zeros(length(X),length(X),length(zinit))
  rho = zeros(length(X),length(X),length(zinit))
  anom = zeros(length(zinit))
  for i in 1:length(X)
    for j in 1:length(X)
      for k in 1:length(zinit)
        anom[k] = t_anom * exp(-(pinit[k]-40000)^2 / 2 / (11000^2))
        theta[i,j,k] = tinit[k] + anom[k] * exp(-( X[i]^2 + Y[j]^2)/(2*RMW^2))
        thetav[i,j,k] = theta[i,j,k] * (1 + 0.61 * qinit[k])
      end
    end
  end
  f = 5e-5
  RMW =  100000
  maxz = length(zinit)
  tvinit = zeros(maxz)
  piinit = zeros(maxz)
  tvinit[1] = tinit[1] * (1.0 + 0.61 *qinit[1])
  piinit[1] = 1
  for k in 2:maxz
    tvinit[k] = tinit[k] * (1.0 + 0.61 *qinit[k])
    piinit[k] = piinit[k-1] - grav / (1004 * 0.5 *(tvinit[k] + tvinit[k-1])) * (zinit[k] - zinit[k-1])
  end
  for i in 1:length(X)
    for j in 1:length(X)
      pressure[i,j,maxz] = pinit[maxz]
      ppi[i,j,maxz] = piinit[maxz]
      temp[i,j,maxz] = tinit[maxz] * piinit[maxz]
      rho[i,j,maxz] =  pinit[maxz]/(287.4 * tvinit[maxz]/piinit[maxz])
    end
  end
  for i in 1:length(X)
    for j in 1:length(X)
      for k in maxz-1:-1 :1
        ppi[i,j,k] = ppi[i,j,k+1] + grav / (1004 * 0.5 *(thetav[i,j,k] + thetav[i,j,k+1])) * (zinit[k+1] - zinit[k])
        pressure[i,j,k] = 101515 * ppi[i,j,k]^(1004/287.4)
        temp[i,j,k] = theta[i,j,k] * ppi[i,j,k]
        rho[i,j,k] = pressure[i,j,k]/(287.4 * thetav[i,j,k]/ppi[i,j,k])
      end
    end
  end
  Z = collect(zinit)
  knots=(X,Y,Z)
  gradpx = zeros(length(X),length(X),length(zinit))
  gradpy = zeros(length(X),length(X),length(zinit))
  

  uinit = zeros(length(X),length(X),length(zinit))
  vinit = zeros(length(X),length(X),length(zinit))
  for i in 1:length(X)
    for j in 1:length(X)
      for k in 1:length(zinit)
        if (i == 1) || (j == 1) || (i == length(X)) || (j == length(X))
          uinit[i,j,k] = 0
          vinit[i,j,k] = 0
        
        else
          gradpx[i,j,k] = (pressure[i+1,j,k] - pressure[i,j,k])/5000
          gradpy[i,j,k] = (pressure[i,j+1,k] - pressure[i,j,k])/5000
          if (gradpx[i,j,k] == 0)
             uinit[i,j,k] = 0
          else
             uinit[i,j,k] = -1 / (f * rho[i,j,k]) * gradpx[i,j,k]
          end
          if (gradpy[i,j,k] == 0)
            vinit[i,j,k] = 0
          else
          vinit[i,j,k] = 1 / (f * rho[i,j,k]) * gradpy[i,j,k]
          end
          if (uinit[i,j,k] > 50)
             @info uinit[i,j,k], gradpx[i,j,k]
          end
        end
      end
    end
  end
  #------------------------------------------------------
  # GET SPLINE FUNCTION
  #------------------------------------------------------
  itp = interpolate(knots, pressure, Gridded(Linear()) )
  spl_pinit = itp
  itp = interpolate(knots, temp, Gridded(Linear()))
  spl_tinit = itp
  itp = interpolate(knots, rho, Gridded(Linear()))
  spl_rhoinit = itp
  itp = interpolate(knots, ppi, Gridded(Linear()))
  spl_ppiinit = itp
  itp = interpolate(knots, theta, Gridded(Linear()))
  spl_thetainit = itp
  itp = interpolate(knots, thetav, Gridded(Linear()))
  spl_thetavinit = itp
  itp = interpolate(knots, uinit, Gridded(Linear()))
  spl_uinit = itp
  itp = interpolate(knots, vinit, Gridded(Linear()))
  spl_vinit = itp


  #spl_tinit    = Spline1D(zinit, tinit; k=1)
  spl_qinit    = Spline1D(zinit, qinit; k=1)
  #spl_uinit    = Spline1D(zinit, uinit; k=1)
  #spl_vinit    = Spline1D(zinit, vinit; k=1)
  #spl_pinit    = Spline1D(zinit, pinit; k=1)
  return spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit, spl_rhoinit, spl_ppiinit, spl_thetainit
end

function config_tc(FT, N, resolution, xmax, ymax, zmax,xmin,ymin)
    # Reference state
    T_min = FT(289)
    T_s = FT(290.4)
    Γ_lapse = FT(grav / cp_d)
    T = LinearTemperatureProfile(T_min, T_s, Γ_lapse)
    rel_hum = FT(0)
    ref_state = HydrostaticState(T, rel_hum)

    # Radiation model
    κ = FT(85)
    α_z = FT(1)
    z_i = FT(840)
    ρ_i = FT(1.13)

    D_subsidence = FT(3.75e-6)

    F_0 = FT(70)
    F_1 = FT(22)
    radiation = DYCOMSRadiation{FT}(κ, α_z, z_i, ρ_i, D_subsidence, F_0, F_1)

    # Sources
    f_coriolis = FT(5e-5)
    u_geostrophic = FT(7.0)
    v_geostrophic = FT(-5.5)
    w_ref = FT(0)
    u_relaxation = SVector(u_geostrophic, v_geostrophic, w_ref)
    # Sponge
    c_sponge = 1
    # Rayleigh damping
    zsponge = FT(17000.0)
    rayleigh_sponge =
        RayleighSponge{FT}(zmax, zsponge, c_sponge, u_relaxation, 2)
    # Geostrophic forcing
    geostrophic_forcing =
        GeostrophicForcing{FT}(f_coriolis, u_geostrophic, v_geostrophic)

    # Boundary conditions
    # SGS Filter constants
    C_smag = FT(0.21) # 0.21 for stable testing, 0.18 in practice
    C_drag = FT(0.0011)
    LHF = FT(50)
    SHF = FT(10)
    ics = init_tc!

    source = (
        Gravity(),
        rayleigh_sponge,
    )

    model = AtmosModel{FT}(
        AtmosLESConfigType;
	ref_state = NoReferenceState(),
	moisture = NonEquilMoist{FT}(),
        turbulence = SmagorinskyLilly{FT}(C_smag),
        source = source,
        boundarycondition = (
            AtmosBC(
                momentum = (Impenetrable(DragLaw(
                    (state, aux, t, normPu) -> C_drag + 4 * 1e-5 * normPu,
                ))),
		energy =  PrescribedEnergyFlux((state, aux, t) -> LHF + SHF),#BulkFormulationEnergy((state, aux, t, normPu) -> C_drag + 4 * 1e-5 * normPu),
		moisture = BulkFormulationMoisture((state, aux, t, normPu) -> C_drag + 4 * 1e-5 * normPu),
                ),
            
            AtmosBC(),
	
        ),
        init_state = ics,
        param_set = ParameterSet{FT}(),
    )

    ode_solver =
        CLIMA.ExplicitSolverType(solver_method = LSRK144NiegemannDiehlBusch)

    config = CLIMA.AtmosLESConfiguration(
        "DYCOMS",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        init_tc!,
	xmin = xmin,
	ymin = ymin,
        solver_type = ode_solver,
        model = model,
    )
    return config
end

function config_diagnostics(driver_config)
    interval = 10000 # in time steps
    dgngrp = setup_atmos_default_diagnostics(interval, driver_config.name)
    return CLIMA.setup_diagnostics([dgngrp])
end

function main()
    CLIMA.init()

    FT = Float64

    # DG polynomial order
    N = 4

    # Domain resolution and size
    Δh = FT(5000)
    Δv = FT(500)
    resolution = (Δh, Δh, Δv)

    xmax = FT(800000)
    ymax = FT(800000)
    zmax = FT(25000)
    xmin = FT(-800000)
    ymin = FT(-800000)

    t0 = FT(0)
    timeend = FT(20)
    spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit, spl_rhoinit, spl_ppiinit, spl_thetainit = spline_int()
    driver_config = config_tc(FT, N, resolution, xmax, ymax, zmax,xmin,ymin)
    solver_config =
        CLIMA.setup_solver(t0, timeend, driver_config,(spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit, spl_rhoinit, spl_ppiinit, spl_thetainit), init_on_cpu = true, Courant_number = 0.45)
    dgn_config = config_diagnostics(driver_config)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init = false)
        Filters.apply!(solver_config.Q, (6,7,8), solver_config.dg.grid, TMARFilter())
        nothing
    end

    result = CLIMA.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbtmarfilter,),
        check_euclidean_distance = true,
    )
end

main()
