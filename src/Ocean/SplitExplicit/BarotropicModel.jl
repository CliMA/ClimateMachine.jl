import CLIMA.DGmethods:
    initialize_fast_state!,
    pass_tendency_from_slow_to_fast!,
    cummulate_fast_solution!,
    reconcile_from_fast_to_slow!

struct BarotropicModel{M} <: AbstractOceanModel
    baroclinic::M
    function BarotropicModel(baroclinic::M) where {M}
        return new{M}(baroclinic)
    end
end

function vars_state(m::BarotropicModel, T)
    @vars begin
        η::T
        U::SVector{2, T}
        η̄::T              # running averge of η
        Ū::SVector{2, T}  # running averge of U
    end
end

function init_state!(m::BarotropicModel, Q::Vars, A::Vars, coords, t)
    return nothing
end

function vars_aux(m::BarotropicModel, T)
    @vars begin
        Gᵁ::SVector{2, T}
        weights_η::T
        weights_U::T
        Ū::SVector{2, T}
    end
end

function init_aux!(m::BarotropicModel, A::Vars, geom::LocalGeometry)
    return ocean_init_aux!(m, m.baroclinic.problem, A, geom)
end

vars_gradient(m::BarotropicModel, T) = @vars()
vars_diffusive(m::BarotropicModel, T) = @vars()
vars_integrals(m::BarotropicModel, T) = @vars()

@inline flux_diffusive!(::BarotropicModel, args...) = nothing

@inline function flux_nondiffusive!(
    m::BarotropicModel,
    F::Grad,
    Q::Vars,
    A::Vars,
    t::Real,
)
    @inbounds begin
        U = Q.U
        η = Q.η
        H = m.baroclinic.problem.H
        I = LinearAlgebra.I

        F.η += U
        F.U += grav * H * η * I
    end
end

@inline function source!(m::BarotropicModel, S::Vars, Q::Vars, A::Vars, t::Real)
    @inbounds begin
        S.U += A.Gᵁ
    end
end

@inline function initialize_fast_state!(
    slow::OceanModel,
    fast::BarotropicModel,
    Qslow,
    Qfast,
    dgSlow,
    dgFast,
)
    Qfast.η̄ .= -0
    Qfast.Ū .= @SVector [-0, -0]

    # copy η and U from 3D equation
    # to calculate U we need to do an integral of u from the 3D
    indefinite_stack_integral!(dgSlow, slow, Qslow, dgSlow.auxstate, 0)

    ### copy results of integral to 2D equation
    boxy_∫u = reshape(dgSlow.auxstate.∫u, Nq^2, Nq, nelemv, nelemh)
    flat_∫u = @view boxy_∫u[:, end, end, :]
    Qfast.U .= flat_∫u

    boxy_η = reshape(Qslow.η, Nq^2, Nq, nelemv, nelemh)
    flat_η = @view boxy_η[:, end, end, :]
    Qfast.η .= flat_η

    return nothing
end

@inline function cummulate_fast_solution(
    fast::BarotropicModel,
    Qfast,
    fast_time,
    fast_dt,
    total_fast_step,
)

    #- might want to use some of the weighting factors: weights_η & weights_U
    #- should account for case where fast_dt < fast.param.dt
    total_fast_step += 1
    # Qfast.η̄ += Qfast.η
    Qfast.Ū += Qfast.U

    return nothing
end

@inline function pass_tendency_from_slow_to_fast!(
    slow::OceanModel,
    fast::BarotropicModel,
    dgSlow,
    dgFast,
    Qfast,
    dQslow,
)
    # integrate the tendency
    tendency_dg = dgSlow.modeldata.tendency_dg
    update_aux!(tendency_dg, tendency_dg.bl, dQslow, 0)

    ### copying ∫du from newdg into Gᵁ of dgFast
    boxy_∫du = reshape(tendency_dg.auxstate.∫du, Nq^2, Nq, nelemv, nelemh)
    flat_∫du = @view boxy_∫du[:, end, end, :]
    dgFast.A.Gᵁ .= flat_∫du

    return nothing
end

@inline function reconcile_from_fast_to_slow!(
    slow::OceanModel,
    fast::BarotropicModel,
    dgSlow,
    dgFast,
    dQslow,
    Qslow,
    Qfast,
    total_fast_step,
)
    # need to calculate int_u using integral kernels
    # u_slow := u_slow + (1/H) * (u_fast - \int_{-H}^{0} u_slow)

    # Compute: \int_{-H}^{0} u_slow)
    ### need to make sure this is stored into aux.∫u
    indefinite_stack_integral!(dgSlow, slow, Qslow, dgSlow.auxstate, t)

    ### substract ∫u from U and divide by H
    boxy_∫u = reshape(dgSlow.auxstate.∫u, Nq^2, Nq, nelemv, nelemh)
    flat_∫u = @view boxy_∫u[:, end, end, :]

    ### apples is a place holder for 1/H * (Ū - ∫u)
    apples = gFast.auxstate.Ū
    # apples .= 1/H * (Qfast.Ū - flat_∫u)
    apples .= 1 / H * (Qfast.Ū / total_fast_step - flat_∫u)

    ### need to reshape these things for the broadcast
    boxy_du = reshape(dQslow.u, Nq, Nq, Nq, nelemv, nelemh)
    boxy_u = reshape(Qslow.u, Nq, Nq, Nq, nelemv, nelemh)
    boxy_apples = reshape(apples, Nq, Nq, 1, 1, nelemh)

    ## this works, we tested it
    ## copy the 2D contribution down the 3D solution
    boxy_u .+= boxy_apples
    boxy_du .+= boxy_apples

    ### copy 2D eta over to 3D model
    boxy_η_3D = reshape(Qslow.η, Nq, Nq, Nq, nelemv, nelemh)
    # boxy_η̄_2D = reshape(Qfast.η̄, Nq, Nq,  1,      1, nelemh)
    # for now, with our simple weights, we just take the final value:
    boxy_η̄_2D = reshape(Qfast.η, Nq, Nq, 1, 1, nelemh)
    boxy_η_3D .= boxy_η̄_2D

    return nothing
end
