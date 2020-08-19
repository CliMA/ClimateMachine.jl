using GaussQuadrature, Plots
using ClimateMachine.Mesh.Elements
using ClimateMachine
gr(size = (400,400))
n = 4
r,w = GaussQuadrature.legendre(n+1, both)
T = eltype(r)
a, b = GaussQuadrature.legendre_coefs(T, n)
x = collect(range(-1,1,length = 100))
V = GaussQuadrature.orthonormal_poly(x, a, b)
p = []
push!(p,plot(x, V[:,1], ylims = (-2,2), 
    title = string(1-1) * "'th",
    legend = false)
)
for i in 2:n+1
    push!(p,
    plot(x, V[:,i], ylims = (-2,2), 
    title = string(i-1) * "'th", 
    legend = false)
    )
end
display(plot(p...))
## Show decomposition
f(x) = sin(π*x) * exp(x) / 2
ry = @. f(r)
xy = @. f(x)
plot(r,ry)
plot!(x, xy)
𝒯 = GaussQuadrature.orthonormal_poly(r, a, b)
spectrum = inv(𝒯) * f.(r)
index = collect(range(0, n,length = n+1))
scatter(index, log10.(abs.(spectrum)), 
xlabel = "Legendre Index", 
ylabel = "Log10(Abs(Amplitude))", 
label = false,
title = "Legendre Modal Amplitudes")
##
function plot_approximation(V, spectrum, i; exact = true)
    p1 = plot(x, V[:,1:i] * spectrum[1:i], ylims = (-1.1,1.1), 
        title = "Modes 0 through " * string(i-1),
        label = "approximation", legend = :bottomright)
    if exact
        plot!(x, xy, color = :red, label = "truth")
    end
    return p1
end

p = []
for i in 1:n+1
    push!(p,
        plot_approximation(V, spectrum, i)
    )
end
display(plot(p...))
for i in 1:n+1
    display(p[i])
    sleep(0.3)
end

## Mass Matrix and Stiffness Matrix
Mi =  𝒯 * 𝒯'
M = inv(Mi)
D = ClimateMachine.Mesh.Elements.spectralderivative(r)
Sᵀ = (M * D)'

## Show that derivatives are not accurate for polynomials of order n+1
# but accurate for polynomials below that order
L = hcat([r .^ i for i in 1:n+1]...)
DL = hcat([i * r .^(i-1) for i in 1:n+1]...)
D * L - DL
## Compute Differentiation Matrix (numerically unstable for large N)
L  = hcat([r .^ i for i in 0:n]...)
DL = hcat([i * r .^ (i-1) for i in 0:n]...)
D*L - DL
D - DL / L # check computation of differentiation matrix
