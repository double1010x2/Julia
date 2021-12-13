# Use baremodule to shave off a few KB from the serialized `.ji` file
baremodule METIS_jll
using Base
using Base: UUID
import JLLWrappers

JLLWrappers.@generate_main_file_header("METIS")
JLLWrappers.@generate_main_file("METIS", UUID("d00139f3-1899-568f-a2f0-47f597d42d70"))
end  # module METIS_jll
