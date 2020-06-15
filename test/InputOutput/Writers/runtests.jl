using Dates
using NCDatasets
using OrderedCollections
using Test
using ClimateMachine.Writers

@testset "Writers" begin
    odims = OrderedDict(
        "x" => (collect(1:5), Dict()),
        "y" => (collect(1:5), Dict()),
        "z" => (collect(1010:10:1050), Dict()),
    )
    ovartypes = OrderedDict(
        "v1" => (("x", "y", "z"), Float64, Dict()),
        "v2" => (("x", "y", "z"), Float64, Dict()),
    )
    vals1 = rand(5, 5, 5)
    vals2 = rand(5, 5, 5)

    nc = NetCDFWriter()
    nfn, _ = mktemp()
    nfull = full_name(nc, nfn)

    init_data(nc, nfn, odims, ovartypes)
    append_data(nc, OrderedDict("v1" => vals1, "v2" => vals2), 2.0)

    NCDataset(nfull, "r") do nds
        xdim = nds["x"][:]
        ydim = nds["y"][:]
        zdim = nds["z"][:]
        @test xdim == odims["x"][1]
        @test ydim == odims["y"][1]
        @test zdim == odims["z"][1]
        @test try
            adim = nds["a"][:]
            adim == ones(5)
            false
        catch e
            true
        end
        t = nds["time"][:]
        v1 = nds["v1"][:]
        v2 = nds["v2"][:]
        @test length(t) == 1
        @test t[1] == DateTime(1900, 1, 1, 0, 0, 2)
        @test v1[:, :, :, 1] == vals1
        @test v2[:, :, :, 1] == vals2
    end
end
