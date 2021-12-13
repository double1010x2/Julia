# Use baremodule to shave off a few KB from the serialized `.ji` file
baremodule ASL_jll
using Base
using Base: UUID
import JLLWrappers

JLLWrappers.@generate_main_file_header("ASL")
JLLWrappers.@generate_main_file("ASL", UUID("ae81ac8f-d209-56e5-92de-9978fef736f9"))
end  # module ASL_jll
