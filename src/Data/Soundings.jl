module Soundings

using DataDeps, MD5

export @datadep_str

function __init__()
  register(DataDep(
    "Gabersek sounding",
    """
	Dataset: Gabersek sounding
	Website: https://figshare.com/articles/Gabersek_sounding/9868307
	Author: Simon Byrne
	Date of Publication: 2019-09-17T22:02:18Z
	License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

	TBA.

	Please cite this dataset: Byrne, Simon (2019): Gabersek sounding. figshare. Dataset.
	""",
	Any["https://ndownloader.figshare.com/files/17695022"],
	[(md5, "974b66b39f34144372441a41052e744f")]
))

end


end
