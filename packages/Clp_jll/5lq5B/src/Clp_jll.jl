# Use baremodule to shave off a few KB from the serialized `.ji` file
baremodule Clp_jll
using Base
using Base: UUID
import JLLWrappers

JLLWrappers.@generate_main_file_header("Clp")
JLLWrappers.@generate_main_file("Clp", UUID("06985876-5285-5a41-9fcb-8948a742cc53"))
end  # module Clp_jll
