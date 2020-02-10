using DataDeps

"""
    extract_targz(file)

Platform-independent file extraction
"""
function extract_targz(file)
  if Sys.iswindows()
    run(pipeline(`7z x -tgzip -so $file`, `7z x -si -ttar`))
  else
    run(`tar -xzf $file`)
  end
end

function data_folder_moist_thermo()
  register(DataDep("MoistThermoConstructorTestData",
                   "Data to test moist thermodynamic constructors",
                   "https://caltech.box.com/shared/static/0741fg18sav94jyt94j4lyymiao4eobw.gz",
                   "d826e1cf57ba8bfcdccf2fb2b3f264c508e7929032a2b6639b379a8d1b1d5dfd",
                   post_fetch_method=extract_targz))
  datafolder = datadep"MoistThermoConstructorTestData"
  return datafolder
end
