using ClimateMachine.Mesh.Grids: polynomialorder

#####
##### CartesianDomain
#####

struct CartesianDomain{FT, G}
    grid::G
    Np::Int
    Ne::NamedTuple{(:x, :y, :z), NTuple{3, Int}}
    x::NTuple{2, FT}
    y::NTuple{2, FT}
    z::NTuple{2, FT}
    Lx::FT
    Ly::FT
    Lz::FT
end

Base.eltype(::CartesianDomain{FT}) where {FT} = FT

Base.show(io::IO, domain::CartesianDomain{FT, G}) where {FT, G} = print(
    io,
    "CartesianDomain{$FT, $(G.name.wrapper)}:",
    '\n',
    "    Np = ",
    domain.Np,
    ", Ne = ",
    domain.Ne,
    '\n',
    @sprintf(
        "    x = (%.2e, %.2e), y = (%.2e, %.2e), z = (%.2e, %.2e)",
        domain.x[1],
        domain.x[2],
        domain.y[1],
        domain.y[2],
        domain.z[1],
        domain.z[2]
    ),
    '\n',
    @sprintf(
        "    Lx = %.2e, Ly = %.2e, Lz = %.2e",
        domain.Lx,
        domain.Ly,
        domain.Lz
    )
)



name_it(Ne::NamedTuple{(:x, :y, :z)}) = Ne
name_it(Ne) = (x = Ne[1], y = Ne[2], z = Ne[3])

function CartesianDomain(grid, Ne)
    Ne = name_it(Ne)

    # Unwind volume geometry
    volume_geometry = grid.vgeo

    # Check number of elements
    prod(Ne) === size(volume_geometry, 3) ||
    error("prod(Ne) must match the total number of grid elements.")

    Np = polynomialorder(grid)

    x = view(volume_geometry, :, grid.x1id, :)
    y = view(volume_geometry, :, grid.x2id, :)
    z = view(volume_geometry, :, grid.x3id, :)

    xlims = (minimum(x), maximum(x))
    ylims = (minimum(y), maximum(y))
    zlims = (minimum(z), maximum(z))

    Lx = xlims[2] - xlims[1]
    Ly = ylims[2] - ylims[1]
    Lz = zlims[2] - zlims[1]

    return CartesianDomain(grid, Np, Ne, xlims, ylims, zlims, Lx, Ly, Lz)
end
