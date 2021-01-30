module TestArtifactWrappers

using Test
using ClimateMachine.ArtifactWrappers

@testset "ArtifactFile" begin
    af = ArtifactFile(url = "url", filename = "filename")
    @test af.url == "url"
    @test af.filename == "filename"
end

@testset "ArtifactWrapper" begin
    mktempdir() do path

        tempfile = joinpath(path, "foo.txt")

        open(tempfile, "w") do io
            print(io, "foo")
        end

        af = ArtifactFile(url = "file://$tempfile", filename = "foo")

        # Caches artifact:
        aw = ArtifactWrapper(path, true, "test_art_wrap", ArtifactFile[af])
        f = get_data_folder(aw)
        @test isfile(joinpath(f, "foo"))

        # Download only:
        aw = ArtifactWrapper(path, false, "test_art_wrap", ArtifactFile[af])

        # Test that "artifact_" was prepended:
        @test occursin("artifact_", aw.artifact_dir)

        f = get_data_folder(aw)
        @test isfile(joinpath(f, "foo"))
    end
end

end
