using CLIMA.ArtifactWrappers
using NCDatasets
using Test

# Get soundings dataset folder:
soundings_dataset = ArtifactWrapper(
    joinpath(@__DIR__, "Artifacts.toml"),
    "soundings",
    ArtifactFile[
        ArtifactFile(
            url = "https://caltech.box.com/shared/static/rjnvt2dlw7etm1c7mmdfrkw5gnfds5lx.nc",
            filename = "sounding_gabersek.nc",
        ),
        ArtifactFile(
            url = "https://caltech.box.com/shared/static/th1a0p5bduej2hx6lisphlpo7pm2wvys.nc",
            filename = "sounding_WKR88.nc",
        ),
        ArtifactFile(
            url = "https://caltech.box.com/shared/static/etz2f6qn45m5x4ow1nwx2mcams9fno9c.nc",
            filename = "sounding_JCP2013_with_pressure.nc",
        ),
        ArtifactFile(
            url = "https://caltech.box.com/shared/static/90um61xxjtxh45cwg2d2t5mucl8cbphe.nc",
            filename = "sounding_JCP2013.nc",
        ),
        ArtifactFile(
            url = "https://caltech.box.com/shared/static/satr1neltu8pedvukn125ywr5eks8m0r.nc",
            filename = "sounding_GC1991.nc",
        ),
        ArtifactFile(
            url = "https://caltech.box.com/shared/static/3kmfcqrlf9scx9166c2fx0g1c36i124l.nc",
            filename = "sounding_blend.nc",
        ),
    ],
)
data_folder = get_data_folder(soundings_dataset)

@testset "Get soundings data" begin
    @test isfile(joinpath(data_folder, "sounding_gabersek.nc"))
    @test isfile(joinpath(data_folder, "sounding_GC1991.nc"))
    @test isfile(joinpath(data_folder, "sounding_WKR88.nc"))
    @test isfile(joinpath(data_folder, "sounding_JCP2013.nc"))
    @test isfile(joinpath(data_folder, "sounding_blend.nc"))
    @test isfile(joinpath(data_folder, "sounding_JCP2013_with_pressure.nc"))
end
