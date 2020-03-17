using KernelAbstractions.Extras: @unroll

@kernel function stage_update!(
    ::NaiveVariant,
    Q,
    Qstages,
    Lstages,
    Rstages,
    Qhat,
    RKA_explicit,
    RKA_implicit,
    dt,
    ::Val{is},
    ::Val{split_nonlinear_linear},
    slow_δ,
    slow_dQ,
) where {is, split_nonlinear_linear}
    i = @index(Global, Linear)
    @inbounds begin
        Qhat_i = Q[i]
        Qstages_is_i = Q[i]

        if slow_δ !== nothing
            Rstages[is - 1][i] += slow_δ * slow_dQ[i]
        end

        @unroll for js in 1:(is - 1)
            R_explicit = dt * RKA_explicit[is, js] * Rstages[js][i]
            L_explicit = dt * RKA_explicit[is, js] * Lstages[js][i]
            L_implicit = dt * RKA_implicit[is, js] * Lstages[js][i]
            Qhat_i += (R_explicit + L_implicit)
            Qstages_is_i += R_explicit
            if split_nonlinear_linear
                Qstages_is_i += L_explicit
            else
                Qhat_i -= L_explicit
            end
        end
        Qstages[is][i] = Qstages_is_i
        Qhat[i] = Qhat_i
    end
end

@kernel function stage_update!(
    ::LowStorageVariant,
    Q,
    Qstages,
    Rstages,
    Qhat,
    Qtt,
    RKA_explicit,
    RKA_implicit,
    dt,
    ::Val{is},
    ::Val{split_nonlinear_linear},
    slow_δ,
    slow_dQ,
) where {is, split_nonlinear_linear}
    i = @index(Global, Linear)
    @inbounds begin
        Qhat_i = Q[i]
        Qstages_is_i = -zero(eltype(Q))

        if slow_δ !== nothing
            Rstages[is - 1][i] += slow_δ * slow_dQ[i]
        end

        @unroll for js in 1:(is - 1)
            if split_nonlinear_linear
                rkcoeff = RKA_implicit[is, js] / RKA_implicit[is, is]
            else
                rkcoeff =
                    (RKA_implicit[is, js] - RKA_explicit[is, js]) /
                    RKA_implicit[is, is]
            end
            commonterm = rkcoeff * Qstages[js][i]
            Qhat_i += commonterm + dt * RKA_explicit[is, js] * Rstages[js][i]
            Qstages_is_i -= commonterm
        end
        Qstages[is][i] = Qstages_is_i
        Qhat[i] = Qhat_i
        Qtt[i] = Qhat_i
    end
end

@kernel function solution_update!(
    ::NaiveVariant,
    Q,
    Lstages,
    Rstages,
    RKB,
    dt,
    ::Val{Nstages},
    ::Val{split_nonlinear_linear},
    slow_δ,
    slow_dQ,
    slow_scaling,
) where {Nstages, split_nonlinear_linear}
    i = @index(Global, Linear)
    @inbounds begin
        if slow_δ !== nothing
            Rstages[Nstages][i] += slow_δ * slow_dQ[i]
        end
        if slow_scaling !== nothing
            slow_dQ[i] *= slow_scaling
        end

        @unroll for is in 1:Nstages
            Q[i] += RKB[is] * dt * Rstages[is][i]
            if split_nonlinear_linear
                Q[i] += RKB[is] * dt * Lstages[is][i]
            end
        end
    end
end

@kernel function solution_update!(
    ::LowStorageVariant,
    Q,
    Rstages,
    RKB,
    dt,
    ::Val{Nstages},
    slow_δ,
    slow_dQ,
    slow_scaling,
) where {Nstages}
    i = @index(Global, Linear)
    @inbounds begin
        if slow_δ !== nothing
            Rstages[Nstages][i] += slow_δ * slow_dQ[i]
        end
        if slow_scaling !== nothing
            slow_dQ[i] *= slow_scaling
        end

        @unroll for is in 1:Nstages
            Q[i] += RKB[is] * dt * Rstages[is][i]
        end
    end
end
