# Use baremodule to shave off a few KB from the serialized `.ji` file
baremodule Osi_jll
using Base
using Base: UUID
import JLLWrappers

JLLWrappers.@generate_main_file_header("Osi")
JLLWrappers.@generate_main_file("Osi", UUID("7da25872-d9ce-5375-a4d3-7a845f58efdd"))
end  # module Osi_jll
