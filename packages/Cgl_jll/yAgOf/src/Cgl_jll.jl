# Use baremodule to shave off a few KB from the serialized `.ji` file
baremodule Cgl_jll
using Base
using Base: UUID
import JLLWrappers

JLLWrappers.@generate_main_file_header("Cgl")
JLLWrappers.@generate_main_file("Cgl", UUID("3830e938-1dd0-5f3e-8b8e-b3ee43226782"))
end  # module Cgl_jll
