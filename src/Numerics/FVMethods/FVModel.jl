
"""
Fifth order WENO reconstruction on nonuniform grids
Implemented based on
Wang, Rong, Hui Feng, and Raymond J. Spiteri.
"Observations on the fifth-order WENO method with non-uniform meshes."
Applied Mathematics and Computation 196.1 (2008): 433-447.

size(Δh) = 5
size(u) = (num_state_primitive, 5)
construct left/right face states of cell[3]
h1     h2     h3     h4      h5
|--i-2--|--i-1--|--i--|--i+1--|--i+2--|
without hat : i - 1/2
with hat    : i + 1/2
r = 0, 1, 2 cells to the left, => P_r^{i}
I_{i-r}, I_{i-r+1}, I_{i-r+2}
P_r(x) =  ∑_{j=0}^{2} C_{rj}(x) u_{i - r + j}  (use these 3 cell averaged values)
C_{rj}(x) = B_{rj}(x) h_{3-r+j}                (i = 3)
C''_{rj}(x) = B''_{rj}(x) h_{3-r+j}                (i = 3)
P''_r(x)    =  ∑_{j=0}^{2} C''_{rj}(x) u_{i - r + j}  (use these 3 cell averaged values)
=  ∑_{j=0}^{2} B''_{rj}(x) h_{3-r+j}  u_{i - r + j}  (use these 3 cell averaged values)

"""
function weno_reconstruction!(
    state_primitive_top::Vars,
    state_primitive_bottom::Vars,
    cell_states_primitive::NTuple{5, Vars},
    cell_weights::SVector{5, FT},
) where {FT}


    num_state_primitive = length(parent(state_primitive_top))
    h1, h2, h3, h4, h5 = cell_weights

    b̂ = zeros(FT, 3, 3)
    b̂[3, 3] = 1 / (h1 + h2 + h3) + 1 / (h2 + h3) + 1 / h3
    b̂[3, 2] = b̂[3, 3] - (h1 + h2 + h3) * (h2 + h3) / ((h1 + h2) * h2 * h3)
    b̂[3, 1] = b̂[3, 2] + (h1 + h2 + h3) * h3 / (h1 * h2 * (h2 + h3))
    b̂[2, 3] = (h2 + h3) * h3 / ((h2 + h3 + h4) * (h3 + h4) * h4)
    b̂[2, 2] = b̂[2, 3] + 1 / (h2 + h3) + 1 / h3 - 1 / h4
    b̂[2, 1] = b̂[2, 2] - ((h2 + h3) * h4) / (h2 * h3 * (h3 + h4))
    b̂[1, 3] = -(h3 * h4) / ((h3 + h4 + h5) * (h4 + h5) * h5)
    b̂[1, 2] = b̂[1, 3] + h3 * (h4 + h5) / ((h3 + h4) * h4 * h5)
    b̂[1, 1] = b̂[1, 2] + 1 / h3 - 1 / h4 - 1 / (h4 + h5)

    b = zeros(FT, 3, 3)
    b[3, 3] = 1 / (h5 + h4 + h3) + 1 / (h4 + h3) + 1 / h3
    b[3, 2] = b[3, 3] - (h5 + h4 + h3) * (h4 + h3) / ((h5 + h4) * h4 * h3)
    b[3, 1] = b[3, 2] + (h5 + h4 + h3) * h3 / (h5 * h4 * (h4 + h3))
    b[2, 3] = (h4 + h3) * h3 / ((h4 + h3 + h2) * (h3 + h2) * h2)
    b[2, 2] = b[2, 3] + 1 / (h4 + h3) + 1 / h3 - 1 / h2
    b[2, 1] = b[2, 2] - ((h4 + h3) * h2) / (h4 * h3 * (h3 + h2))
    b[1, 3] = -(h3 * h2) / ((h3 + h2 + h1) * (h2 + h1) * h1)
    b[1, 2] = b[1, 3] + h3 * (h2 + h1) / ((h3 + h2) * h2 * h1)
    b[1, 1] = b[1, 2] + 1 / h3 - 1 / h2 - 1 / (h2 + h1)


    # at i - 1/2, i + 1/2
    P = zeros(FT, num_state_primitive, 2, 3)
    for r in 0:2
        for j in 0:2
            P[:, 1, 3 - r] +=
                b[r + 1, j + 1] *
                cell_weights[3 + r - j] *
                parent(cell_states_primitive[3 + r - j])
            P[:, 2, r + 1] +=
                b̂[r + 1, j + 1] *
                cell_weights[3 - r + j] *
                parent(cell_states_primitive[3 - r + j])
        end
    end


    # build the second derivative part in smoothness measure
    d2B = zeros(FT, 3, 3)
    for r in 0:2
        d2B[r + 1, 3] =
            6.0 / (
                (
                    cell_weights[3 - r] +
                    cell_weights[4 - r] +
                    cell_weights[5 - r]
                ) *
                (cell_weights[4 - r] + cell_weights[5 - r]) *
                cell_weights[5 - r]
            )
        d2B[r + 1, 2] =
            d2B[r + 1, 3] -
            6.0 / (
                (cell_weights[3 - r] + cell_weights[4 - r]) *
                cell_weights[4 - r] *
                cell_weights[5 - r]
            )
        d2B[r + 1, 1] =
            d2B[r + 1, 2] +
            6.0 / (
                cell_weights[3 - r] *
                cell_weights[4 - r] *
                (cell_weights[4 - r] + cell_weights[5 - r])
            )
    end

    d2P = zeros(FT, num_state_primitive, 3)
    for r in 0:2
        for j in 0:2
            d2P[:, r + 1] +=
                d2B[r + 1, j + 1] *
                cell_weights[3 - r + j] *
                parent(cell_states_primitive[3 - r + j])
        end
    end

    IS2 = h3^4 * d2P .^ 2

    # build the first derivative part in smoothness measure

    d1B = zeros(FT, 3, 3, 3) # xi-1/2 xi, xi+1/2; r, j
    d1B[1, 3, 3] = 2 * (h1 + 2 * h2) / ((h1 + h2 + h3) * (h2 + h3) * h3)
    d1B[1, 3, 2] = d1B[1, 3, 3] - 2 * (h1 + 2 * h2 - h3) / ((h1 + h2) * h2 * h3)
    d1B[1, 3, 1] = d1B[1, 3, 2] + 2 * (h1 + h2 - h3) / (h1 * h2 * (h2 + h3))

    d1B[1, 2, 3] = 2 * (h2 - h3) / ((h2 + h3 + h4) * (h3 + h4) * h4)
    d1B[1, 2, 2] = d1B[1, 2, 3] - 2 * (h2 - h3 - h4) / ((h2 + h3) * h3 * h4)
    d1B[1, 2, 1] = d1B[1, 2, 2] + 2 * (h2 - 2 * h3 - h4) / (h2 * h3 * (h3 + h4))

    # bug in the paper
    d1B[1, 1, 3] = -2 * (2 * h3 + h4) / ((h3 + h4 + h5) * (h4 + h5) * h5)
    d1B[1, 1, 2] = d1B[1, 1, 3] + 2 * (2 * h3 + h4 + h5) / ((h3 + h4) * h4 * h5)
    d1B[1, 1, 1] =
        d1B[1, 1, 2] - 2 * (2 * h3 + 2 * h4 + h5) / (h3 * h4 * (h4 + h5))

    for r in 0:2
        for j in 0:2
            d1B[2, r + 1, j + 1] =
                d1B[1, r + 1, j + 1] + 0.5 * h3 * d2B[r + 1, j + 1]
            d1B[3, r + 1, j + 1] = d1B[1, r + 1, j + 1] + h3 * d2B[r + 1, j + 1]
        end
    end
    d1P = zeros(FT, num_state_primitive, 3, 3)   # xi-1/2 xi, xi+1/2; r
    for i in 1:3
        for r in 0:2
            for j in 0:2
                d1P[:, i, r + 1] +=
                    d1B[i, r + 1, j + 1] *
                    cell_weights[3 - r + j] *
                    parent(cell_states_primitive[3 - r + j])
            end
        end
    end

    IS1 =
        h3^2 * (d1P[:, 1, :] .^ 2 + 4 * d1P[:, 2, :] .^ 2 + d1P[:, 3, :] .^ 2) /
        6.0


    IS = IS1 + IS2

    # high order test
    # IS .= 1.0

    d = zeros(FT, 3)
    d[3] =
        (h3 + h4) * (h3 + h4 + h5) /
        ((h1 + h2 + h3 + h4) * (h1 + h2 + h3 + h4 + h5))
    d[2] =
        (h1 + h2) * (h3 + h4 + h5) * (h1 + 2 * h2 + 2 * h3 + 2 * h4 + h5) /
        ((h1 + h2 + h3 + h4) * (h2 + h3 + h4 + h5) * (h1 + h2 + h3 + h4 + h5))
    d[1] = h2 * (h1 + h2) / ((h2 + h3 + h4 + h5) * (h1 + h2 + h3 + h4 + h5))

    d̂ = zeros(FT, 3)
    d̂[3] = h4 * (h4 + h5) / ((h1 + h2 + h3 + h4) * (h1 + h2 + h3 + h4 + h5))
    d̂[2] =
        (h1 + h2 + h3) * (h4 + h5) * (h1 + 2 * h2 + 2 * h3 + 2 * h4 + h5) /
        ((h1 + h2 + h3 + h4) * (h2 + h3 + h4 + h5) * (h1 + h2 + h3 + h4 + h5))
    d̂[1] =
        (h2 + h3) * (h1 + h2 + h3) /
        ((h2 + h3 + h4 + h5) * (h1 + h2 + h3 + h4 + h5))

    ϵ = 1.0e-6
    α = d' ./ (ϵ .+ IS) .^ 2
    α̂ = d̂' ./ (ϵ .+ IS) .^ 2

    w = α ./ sum(α, dims = 2)
    ŵ = α̂ ./ sum(α̂, dims = 2)


    # at  i - 1/2,  i + 1/2
    state_primitive_top_arr = parent(state_primitive_top)
    state_primitive_bottom_arr = parent(state_primitive_bottom)
    for i in 1:num_state_primitive
        state_primitive_bottom_arr[i] += w[i, :]' * P[i, 1, :]
        state_primitive_top_arr[i] += ŵ[i, :]' * P[i, 2, :]
    end
end



"""
Third order WENO reconstruction on nonuniform grids
Implemented based on
Cravero, Isabella, and Matteo Semplice.
"On the accuracy of WENO and CWENO reconstructions of third order on nonuniform meshes."
Journal of Scientific Computing 67.3 (2016): 1219-1246.

"""
function weno_reconstruction!(
    state_primitive_top::Vars,
    state_primitive_bottom::Vars,
    cell_states_primitive::NTuple{3, Vars},
    cell_weights::SVector{3, FT},
) where {FT}

    num_state_primitive = length(parent(state_primitive_top))
    h1, h2, h3 = cell_weights


    β, γ = h1 / h2, h3 / h2
    C⁺_l, C⁺_r = γ / (1 + β + γ), (1 + β) / (1 + β + γ)
    C⁻_l, C⁻_r = (1 + β) / (1 + β + γ), γ / (1 + β + γ)


    dP = zeros(FT, num_state_primitive, 2)
    dP[:, 1] =
        2 *
        (parent(cell_states_primitive[2]) - parent(cell_states_primitive[1])) /
        (h1 + h2)
    dP[:, 2] =
        2 *
        (parent(cell_states_primitive[3]) - parent(cell_states_primitive[2])) /
        (h2 + h3)


    # at i - 1/2, i + 1/2, r = 0, 1
    P = zeros(FT, num_state_primitive, 2, 2)
    for r in 0:1
        P[:, 1, r + 1] =
            parent(cell_states_primitive[2]) - dP[:, r + 1] * h2 / 2.0
        P[:, 2, r + 1] =
            parent(cell_states_primitive[2]) + dP[:, r + 1] * h2 / 2.0
    end

    # IS = int h2 *P'^2 dx, P  is a linear function
    IS = h2^2 * dP .^ 2

    # high order test
    # IS .= 1.0

    d = [(1 + γ) / (1 + β + γ); β / (1 + β + γ)]
    d̂ = [γ / (1 + β + γ); (1 + β) / (1 + β + γ)]

    ϵ = 1.0e-6
    α = d' ./ (ϵ .+ IS) .^ 2
    α̂ = d̂' ./ (ϵ .+ IS) .^ 2

    w = α ./ sum(α, dims = 2)
    ŵ = α̂ ./ sum(α̂, dims = 2)

    # at  i - 1/2,  i + 1/2
    state_primitive_top_arr = parent(state_primitive_top)
    state_primitive_bottom_arr = parent(state_primitive_bottom)
    for i in 1:num_state_primitive
        state_primitive_bottom_arr[i] += w[i, :]' * P[i, 1, :]
        state_primitive_top_arr[i] += ŵ[i, :]' * P[i, 2, :]
    end

end




"""
Classical Second order FV reconstruction on nonuniform grids
Van Leer Limiter is used
Implemented based on https://en.wikipedia.org/wiki/Flux_limiter
"""
function limiter(
    Δ⁺::Array{FT, 1},
    Δ⁻::Array{FT, 1},
    num_state::IT,
) where {FT, IT}
    Δ = zeros(FT, num_state)
    for s in 1:num_state
        if Δ⁺[s] * Δ⁻[s] > 0.0
            Δ[s] = FT(2) * Δ⁺[s] * Δ⁻[s] / (Δ⁺[s] + Δ⁻[s])
        end
    end
    return Δ
end
function fv_reconstruction!(
    state_primitive_top::Vars,
    state_primitive_bottom::Vars,
    cell_states_primitive::NTuple{3, Vars},
    cell_weights::SVector{3, FT},
) where {FT}

    num_state_primitive = length(parent(state_primitive_top))
    Δz⁻, Δz, Δz⁺ = cell_weights


    Δu⁺ = (parent(cell_states_primitive[3]) .- parent(cell_states_primitive[2]))
    Δu⁻ = (parent(cell_states_primitive[2]) .- parent(cell_states_primitive[1]))

    ∂state =
        2.0 * limiter(Δu⁺ / (Δz⁺ + Δz), Δu⁻ / (Δz⁻ + Δz), num_state_primitive)


    u⁺ = parent(state_primitive_top)
    u⁻ = parent(state_primitive_bottom)

    u⁺ .= parent(cell_states_primitive[2]) .+ ∂state * Δz / 2.0
    u⁻ .= parent(cell_states_primitive[2]) .- ∂state * Δz / 2.0
end


"""
First order (constant) FV reconstruction on nonuniform grids
Mainly used for Boundary conditions
"""
function const_reconstruction!(
    state_primitive_top::Vars,
    state_primitive_bottom::Vars,
    cell_states_primitive::NTuple{1, Vars},
    cell_weights::SVector{1, FT},
) where {FT}

    state_primitive_top_arr = parent(state_primitive_top)
    state_primitive_bottom_arr = parent(state_primitive_bottom)

    state_primitive_top_arr .= parent(cell_states_primitive[1])
    state_primitive_bottom_arr .= parent(cell_states_primitive[1])
end
