# Use baremodule to shave off a few KB from the serialized `.ji` file
baremodule CImGui_jll
using Base
using Base: UUID
import JLLWrappers

JLLWrappers.@generate_main_file_header("CImGui")
JLLWrappers.@generate_main_file("CImGui", UUID("7dd61d3b-0da5-5c94-bbf9-a0296c6e3925"))
end  # module CImGui_jll
