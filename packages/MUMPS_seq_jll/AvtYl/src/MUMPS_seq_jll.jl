# Use baremodule to shave off a few KB from the serialized `.ji` file
baremodule MUMPS_seq_jll
using Base
using Base: UUID
import JLLWrappers

JLLWrappers.@generate_main_file_header("MUMPS_seq")
JLLWrappers.@generate_main_file("MUMPS_seq", UUID("d7ed1dd3-d0ae-5e8e-bfb4-87a502085b8d"))
end  # module MUMPS_seq_jll
