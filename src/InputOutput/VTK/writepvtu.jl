using ..TicToc

"""
    writepvtu(pvtuprefix, vtkprefixes, fieldnames, FT)

Write a pvtu file with the prefix 'pvtuprefix' for the collection of vtk files
given by 'vtkprefixes' using names  of fields 'fieldnames'. The data in the
`vtu` files is of type `FT`.
"""
function writepvtu(pvtuprefix, vtkprefixes, fieldnames, FT)
    open(pvtuprefix * ".pvtu", "w") do pvtufile
        write(
            pvtufile,
            """
            <?xml version="1.0"?>
            <VTKFile type="PUnstructuredGrid" version="0.1" compressor="vtkZLibDataCompressor" byte_order="LittleEndian">
              <PUnstructuredGrid GhostLevel="0">
                <PPoints>
                  <PDataArray type="$FT" Name="Position" NumberOfComponents="3" format="binary"/>
                </PPoints>
                <PPointData>
            """,
        )

        for name in fieldnames
            write(
                pvtufile,
                """
                      <PDataArray type="$FT" Name="$name" format="binary"/>
                """,
            )
        end

        write(
            pvtufile,
            """
                </PPointData>
            """,
        )

        for name in vtkprefixes
            write(
                pvtufile,
                """
                    <Piece Source="$name.vtu"/>
                """,
            )
        end

        write(
            pvtufile,
            """
              </PUnstructuredGrid>
            </VTKFile>
            """,
        )
    end
    return nothing
end
