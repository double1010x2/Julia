# Use baremodule to shave off a few KB from the serialized `.ji` file
baremodule Qt5Base_jll
using Base
using Base: UUID
import JLLWrappers

JLLWrappers.@generate_main_file_header("Qt5Base")
JLLWrappers.@generate_main_file("Qt5Base", UUID("ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"))
end  # module Qt5Base_jll
