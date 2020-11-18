using RRTMGP.GrayBCs:                    GrayLwBCs
using RRTMGP.GrayAngularDiscretizations: AngularDiscretization
using RRTMGP.GrayAtmosphericStates:      GrayAtmosphericState
using RRTMGP.GrayFluxes:                 GrayFlux
using RRTMGP.GraySources:                source_func_longwave_gray_atmos
using RRTMGP.GrayAtmos:                  GrayRRTMGP
using RRTMGP.GrayRTESolver:              gray_atmos_lw!
using RRTMGP.GrayOptics:                 GrayOneScalar, GrayTwoStream
using ..Mesh.Grids

export RadiationModel, NoRadiation, GrayRadiation

abstract type RadiationModel end

vars_state(::RadiationModel, ::AbstractStateType, FT) = @vars()

function atmos_nodal_update_auxiliary_state!(
    ::RadiationModel,
    ::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function integral_set_auxiliary_state!(::RadiationModel, integ::Vars, aux::Vars) end
function integral_load_auxiliary_state!(
    ::RadiationModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) end
function reverse_integral_set_auxiliary_state!(
    ::RadiationModel,
    integ::Vars,
    aux::Vars,
) end
function reverse_integral_load_auxiliary_state!(
    ::RadiationModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) end
function flux_radiation!(
    ::RadiationModel,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end

struct NoRadiation <: RadiationModel end

# This has to be mutable so that the model can be defined after construction.
mutable struct GrayRadiation{NGaussAngles, NStreams} <: RadiationModel
    RRTMGPmodel::GrayRRTMGP
    latitude
    sfc_emissivity

    GrayRadiation(;
        ngaussangles::Int = 1,
        nstreams::Int = 1,
    ) = new{ngaussangles, nstreams}()
end

function vars_state(::GrayRadiation, ::VerticallyFlattened, FT)
    @vars begin
        z::FT
        pressure::FT
        temperature::FT
        flux::FT
    end
end

function atmos_init_state_vertically_flattened!(
    rad::GrayRadiation{NGaussAngles, NStreams},
    m::AtmosModel,
    vf,
    aux,
    grid
) where {NGaussAngles, NStreams}
    DA = arraytype(grid)

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1
    Nqj = dim == 2 ? 1 : Nq
    elems = grid.topology.realelems
    nvertelem = grid.topology.stacksize
    horzelems = fld1(first(elems), nvertelem):fld1(last(elems), nvertelem)
    nhorzelem = length(horzelems)
    vgeo = grid.vgeo

    FT = eltype(vgeo)
    rad.latitude = similar(DA, FT, Nq * Nqj * nhorzelem)
    rad.sfc_emissivity = similar(DA, FT, Nq * Nqj * nhorzelem)

    for i in 1:Nq
        for j in 1:Nqj
            for eh in horzelems
                hindex = i + Nq * (j - 1) + Nq * Nqj * (eh - 1)

                # Fill in all but the top z-coordinates.
                for ev in 1:nvertelem
                    e = ev + (eh - 1) * nvertelem
                    for k in 1:Nq - 1
                        ijk = i + Nq * ((j - 1) + Nqj * (k - 1))
                        vindex = k + (Nq - 1) * (ev - 1)
                        vf[1][vindex, hindex] = vgeo[ijk, Grids._x3, e]
                    end
                end

                # Fill in the top z-coordinates.
                ev = nvertelem
                k = Nq
                e = ev + (eh - 1) * nvertelem
                ijk = i + Nq * ((j - 1) + Nqj * (k - 1))
                vindex = k + (Nq - 1) * (ev - 1)
                vf[1][vindex, hindex] = vgeo[ijk, Grids._x3, e]

                # Fill in the latitudes.
                rad.latitude[hindex] = latitude(
                    m,
                    Vars{vars_state(m, Auxiliary(), FT)}(view(aux, ijk, :, e)),
                )

                # Need to get surface emissivity from actual boundary conditions.
                rad.sfc_emissivity[hindex] = 1
            end
        end
    end

    as = GrayAtmosphericState(
        nvertelem - 1,
        nhorzelem,
        vf[2],
        vf[3],
        vf[1],
        rad.latitude,
        DA,
    )
    OPC = NStreams == 1 ? GrayOneScalar : GrayTwoStream
    optical_props = OPC(FT, nhorzelem, nvertelem - 1, DA)
    sf = source_func_longwave_gray_atmos(FT, nhorzelem, nvertelem - 1, 1, OPC, DA)
    bcs = GrayLwBCs(DA, rad.sfc_emissivity)
    gray_flux = GrayFlux(nhorzelem, nvertelem - 1, nvertelem, FT, DA)
    ang_disc = AngularDiscretization(FT, NGaussAngles, DA)
    rad.RRTMGPmodel = GrayRRTMGP{
        FT,
        Int,
        DA{FT, 1},
        DA{FT, 2},
        DA{FT, 3},
        Bool,
        typeof(optical_props),
        typeof(sf),
        typeof(bcs),
    }(
        as,
        optical_props,
        sf,
        bcs,
        gray_flux,
        ang_disc,
    )
end

function atmos_nodal_update_vertically_flattened_state!(
    ::GrayRadiation,
    m::AtmosModel,
    vf::Vars,
    state::Vars,
    aux::Vars,
    ::Val{false}
)
    vf.radiation.pressure = pressure(m, m.moisture, state, aux)
    vf.radiation.temperature = temperature(m, m.moisture, state, aux)
end

function atmos_nodal_update_vertically_flattened_state!(
    ::GrayRadiation,
    m::AtmosModel,
    vf::Vars,
    state::Vars,
    aux::Vars,
    ::Val{true}
)
    vf.radiation.pressure += pressure(m, m.moisture, state, aux)
    vf.radiation.temperature += temperature(m, m.moisture, state, aux)
    vf.radiation.pressure /= 2
    vf.radiation.temperature /= 2
end

function atmos_update_vertically_flattened_state!(rad::GrayRadiation)
    gray_atmos_lw!(rad.RRTMGPmodel, max_threads = 256)
end