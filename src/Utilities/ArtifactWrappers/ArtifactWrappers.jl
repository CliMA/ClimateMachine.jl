module ArtifactWrappers

using Pkg.Artifacts
using DocStringExtensions: FIELDS

export ArtifactWrapper, ArtifactFile, get_data_folder

"""
    ArtifactFile

A single data file to be downloaded,
containing both the url and the name to
use locally.

# Fields
$(FIELDS)
"""
Base.@kwdef struct ArtifactFile
    "URL pointing to data to be downloaded"
    url::AbstractString = ""
    "Local name used for downloaded online data"
    filename::AbstractString = ""
end

"""
    ArtifactWrapper

A set of data files to be downloaded,
grouped by `data_name`. Example:

```
dataset =
  ArtifactWrapper("Artifacts.toml", "MyDataSet",
    ArtifactFile[
    ArtifactFile(url="https://..../SomeNetCDF1.nc",
              filename="experiment1.nc"),
    ArtifactFile(url="https://..../SomeNetCDF2.nc",
              filename="experiment2.nc"),
    ]
    )
```

# Fields
$(FIELDS)
"""
struct ArtifactWrapper
    "Path to the used Artifacts.toml"
    artifact_toml::AbstractString
    "Unique name of dataset"
    data_name::AbstractString
    "Array of `ArtifactFile`'s, grouped by this dataset"
    artifact_files::Vector{ArtifactFile}
end

"""
    get_data_folder(art_wrap::ArtifactWrapper)

Get local folder of dataset
defined in `art_wrap`.

Example (extending [`ArtifactWrapper`](@ref) example):

```
dataset_path = get_data_folder(dataset)
```
"""
function get_data_folder(art_wrap::ArtifactWrapper)

    # Query the `Artifacts.toml` file for the hash bound to the name data_name
    # (returns `nothing` if no such binding exists)
    data_hash = artifact_hash(art_wrap.data_name, art_wrap.artifact_toml)

    # If the name was not bound, or the hash it was bound to does not exist, create it!
    if data_hash == nothing || !artifact_exists(data_hash)
        # create_artifact() returns the content-hash of the artifact directory once we're finished creating it
        data_hash = create_artifact() do artifact_dir
            # We create the artifact by simply downloading a few files into the new artifact directory
            filenames = [af.filename for af in art_wrap.artifact_files]
            urls = [af.url for af in art_wrap.artifact_files]
            for (url, filename) in zip(urls, filenames)
                download("$(url)", joinpath(artifact_dir, filename))
            end
        end

        # Now bind that hash within our `Artifacts.toml`.  `force = true` means that if it already exists,
        # just overwrite with the new content-hash.  Unless the source files change, we do not expect
        # the content hash to change, so this should not cause unnecessary version control churn.
        bind_artifact!(
            art_wrap.artifact_toml,
            art_wrap.data_name,
            data_hash,
            force = true,
        )
    end

    # Get the path of the dataset, either newly created or previously generated.
    # this should be something like `~/.julia/artifacts/dbd04e28be047a54fbe9bf67e934be5b5e0d357a`
    dataset_path = artifact_path(data_hash)
    return dataset_path
end

end # module
